"""`role-set write`: the runner wired to the real writer and critic, or to the mocks for CI.

Refused before anything is sent: an endpoint off the loopback, a writer not on
the licence-checked list, a cloud tag, a critic step without a grant or with a
stale watchdog heartbeat, a pack that has drifted from its manifest, a missing
gates package. `--mock` runs the whole pipeline with no model; its output is
marked and is never data.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from role_set.assemble import trait_name
from role_set.exemplars import ExemplarError, load_exemplars
from role_set.gates import Gates, GatesUnavailable, load_gates
from role_set.harvest import load_pool
from role_set.guards import GuardError, require_grant, require_heartbeat, require_local_endpoint, require_model_allowed, require_ollama_digest
from role_set.mock import MockCritic, MockWriter, mock_gates
from role_set.pack import PackError, verify
from role_set.plan import Slot
from role_set.prompts import load_pack
from role_set.runner import RunHalted, Runner
from role_set.stuckrule import StuckRuleError
from role_set.stuckrule import load as load_stuck
from role_set.transport import chat, ollama_chat, unload

# The writer prompt (lesson, kinds, amendments, slot brief, material, exemplars) measures about 5,800 tokens with the
# real pack; 16,384 leaves room for the reply and its thinking.
WRITER_NUM_CTX = 16384


class WriteRefused(RuntimeError):
    """Something a run needs is missing or wrong. Nothing was sent."""


@dataclass(frozen=True)
class WriteConfig:
    plan: Path
    pack: Path
    out: Path
    endpoint: str | None = None
    model: str | None = None
    writers_file: Path = Path("writers.json")
    critic_endpoint: str | None = None
    critic_model: str | None = None
    grant: str | None = None
    heartbeat: Path | None = None
    concurrency: int = 1
    mock: bool = False
    no_critic: bool = False
    pool: Path | None = None
    exemplar_repo: Path | None = None
    exemplar_commit: str | None = None
    exemplar_files: tuple[str, ...] = ()
    stuck_repo: Path | None = None  # aspire-si checkout for the frozen stuck rule; defaults to the exemplar repo
    stuck_commit: str | None = None  # defaults to the exemplar commit
    stop_after_minutes: float | None = None  # no new slot starts after this; running slots finish


def load_plan(path: Path) -> tuple[list[Slot], dict[str, tuple[str, str]], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = [Slot(**d) for d in data["slots"]]
    sides: dict[str, tuple[str, str]] = {}
    for s in slots:
        seen = sides.setdefault(s.lesson, (s.side, s.side))
        if s.side not in seen:
            sides[s.lesson] = (seen[0], s.side)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return slots, sides, sha


def trait_names_from_pack(pack: dict[str, str]) -> dict[str, str]:
    names: dict[str, str] = {}
    for key, text in pack.items():
        m = re.fullmatch(r"traits/(T\d\d)\.md", key)
        if not m:
            continue
        heading = next((line for line in text.splitlines() if line.startswith("#")), "")
        try:
            names[m.group(1)] = trait_name(heading)
        except Exception:  # noqa: BLE001 - a malformed heading is reported, not fatal
            names[m.group(1)] = m.group(1)
    return names


def allowed_writers(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {w["served_as"]: w for w in data.get("writers", [])}


def allowed_critics(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {c["served_as"]: c for c in data.get("critics", [])}


def build(cfg: WriteConfig, *, now=time.time, digest_opener=None):
    """Everything the runner needs, with every refusal raised before a byte is sent."""

    slots, sides, plan_sha = load_plan(cfg.plan)
    try:
        manifest = verify(cfg.pack)
    except PackError as e:
        raise WriteRefused(str(e)) from e
    pack = load_pack(cfg.pack)
    trait_names = trait_names_from_pack(pack)
    pins: dict = {"plan_sha256": plan_sha, "pack_sha256": manifest.pack_sha256, "pack_commit": manifest.commit}

    if cfg.mock:
        writer = MockWriter()
        critic = None if cfg.no_critic else MockCritic(answers=writer.answers)
        try:
            gates: Gates = load_gates()
        except GatesUnavailable:
            gates = mock_gates()
        pins.update({"writer_model": "mock", "critic_model": None if cfg.no_critic else "mock", "mock": True})
        return slots, sides, pack, trait_names, gates, writer, critic, pins, None

    try:
        gates = load_gates()
        endpoint = require_local_endpoint(cfg.endpoint or "")
        writers = allowed_writers(cfg.writers_file)
        model = require_model_allowed(cfg.model or "", sorted(writers))
    except (GatesUnavailable, GuardError) as e:
        raise WriteRefused(str(e)) from e
    entry = writers[model]
    rehearsal = bool(entry.get("rehearsal_only"))
    if rehearsal and not entry.get("digest"):
        raise WriteRefused(f"REFUSED: {model} is rehearsal-only and has no digest pin; a rehearsal writer is local and pinned by digest")
    pins.update({"writer_model": model, "writer_repo": entry.get("repo"), "writer_revision": entry.get("revision"),
                 "writer_base_revision": entry.get("base_revision"), "writer_licence": entry.get("licence"), "rehearsal": rehearsal})

    def card(what: str) -> None:
        require_grant(cfg.grant, uses_card=True)
        if cfg.heartbeat is None:
            raise GuardError(f"REFUSED: the {what} on the 5090 needs --heartbeat <watchdog heartbeat file>")
        require_heartbeat(cfg.heartbeat, now())

    unloads: list = []
    if entry.get("digest"):
        # A writer served by the local Ollama (the rehearsal): on the 5090, so it needs the grant; pinned by digest
        # like the critic; and called on the native API, because the OpenAI-style endpoint ignores num_ctx and the
        # writer prompt runs past Ollama's default context.
        try:
            card("local writer")
            w_digest = require_ollama_digest(endpoint, model, entry["digest"], opener=digest_opener)
        except GuardError as e:
            raise WriteRefused(str(e)) from e
        pins.update({"writer_digest": w_digest, "grant": cfg.grant})
        unloads.append((endpoint, model))

        def writer(messages, seed=0):
            return ollama_chat(endpoint, model, messages, seed=seed, num_ctx=WRITER_NUM_CTX)
    else:
        def writer(messages, seed=0):
            return chat(endpoint, model, messages, seed=seed)

    critic = None
    if not cfg.no_critic:
        try:
            card("critic")
            c_endpoint = require_local_endpoint(cfg.critic_endpoint or "")
            c_model = cfg.critic_model or ""
            if not c_model or "cloud" in c_model.lower():
                raise GuardError("REFUSED: a local critic model is required, never a cloud tag")
            critics = allowed_critics(cfg.writers_file)
            if c_model not in critics:
                raise GuardError(f"REFUSED: {c_model} is not on the licence-checked critic list")
            digest = require_ollama_digest(c_endpoint, c_model, critics[c_model]["digest"], opener=digest_opener)
        except GuardError as e:
            raise WriteRefused(str(e)) from e
        pins.update({"critic_model": c_model, "critic_digest": digest, "grant": cfg.grant})

        def critic(messages, **sampling):
            return ollama_chat(c_endpoint, c_model, messages, **sampling)

        unloads.append((c_endpoint, c_model))

    cleanup = None
    if unloads:
        def cleanup():
            for e, m in unloads:
                unload(e, m)

    return slots, sides, pack, trait_names, gates, writer, critic, pins, cleanup


def output_dir(base: Path, pins: dict, *, mock: bool) -> Path:
    """Where a run writes. Mock output goes under mock/ and a rehearsal writer's under rehearsal/, so neither is
    ever confused with data; require_rehearsal_dir refuses any other place for a rehearsal writer."""

    out = base / "mock" if mock else (base / "rehearsal" if pins.get("rehearsal") else base)
    require_rehearsal_dir(out, pins)
    return out


def require_rehearsal_dir(out: Path, pins: dict) -> None:
    if pins.get("rehearsal") and out.name != "rehearsal":
        raise WriteRefused(f"REFUSED: {pins.get('writer_model')} is rehearsal-only; it writes only under a rehearsal/ directory, never {out.name}/")


def run(cfg: WriteConfig, *, now=time.time) -> dict:
    slots, sides, pack, trait_names, gates, writer, critic, pins, cleanup = build(cfg, now=now)
    out = output_dir(cfg.out, pins, mock=cfg.mock)
    excerpts = load_pool(cfg.pool) if cfg.pool else None
    exemplars = None
    if cfg.exemplar_repo and cfg.exemplar_commit and cfg.exemplar_files:
        try:
            exemplars = load_exemplars(cfg.exemplar_repo, cfg.exemplar_commit, list(cfg.exemplar_files))
            pins["exemplar_commit"] = cfg.exemplar_commit
        except ExemplarError as e:
            raise WriteRefused(str(e)) from e
    stuck = None
    repo, commit = cfg.stuck_repo or cfg.exemplar_repo, cfg.stuck_commit or cfg.exemplar_commit
    if repo and commit:
        try:
            stuck = load_stuck(repo, commit)
            pins["stuck_rule_sha256"] = stuck.rule_sha256
        except StuckRuleError as e:
            raise WriteRefused(str(e)) from e
    runner = Runner(gates=gates, pack=pack, trait_names=trait_names, writer=writer, critic=critic, out_dir=out, concurrency=cfg.concurrency, pins=pins,
                    excerpts=excerpts, exemplars=exemplars, stuck=stuck)
    halted = None
    budget = cfg.stop_after_minutes * 60 if cfg.stop_after_minutes else None
    try:
        outcomes = runner.run(slots, stop_after_s=budget)
    except RunHalted as e:
        halted = str(e)
        outcomes = []
    finally:
        if cleanup:
            cleanup()
    return {
        "out_dir": out.name,
        "accepted": runner.status.accepted,
        "dropped": runner.status.dropped,
        "attempted": runner.status.attempted,
        "halted": halted,
        "lesson_halts": runner.status.lesson_halts,
        "skipped": runner.status.skipped,
        "batch_check": runner.batch_check,
        "rehearsal": bool(pins.get("rehearsal")),
        "mock": cfg.mock,
        "slots": len(slots),
        "outcomes": len(outcomes),
    }
