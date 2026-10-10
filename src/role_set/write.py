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
from role_set.transport import chat, ollama_chat, unload


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
    pins.update({"writer_model": model, "writer_repo": entry.get("repo"), "writer_revision": entry.get("revision"),
                 "writer_base_revision": entry.get("base_revision"), "writer_licence": entry.get("licence")})

    def writer(messages, seed=0):
        return chat(endpoint, model, messages, seed=seed)

    critic = None
    cleanup = None
    if not cfg.no_critic:
        try:
            require_grant(cfg.grant, uses_card=True)
            if cfg.heartbeat is None:
                raise GuardError("REFUSED: the critic on the 5090 needs --heartbeat <watchdog heartbeat file>")
            require_heartbeat(cfg.heartbeat, now())
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

        def critic(messages):
            return ollama_chat(c_endpoint, c_model, messages)

        def cleanup():
            unload(c_endpoint, c_model)

    return slots, sides, pack, trait_names, gates, writer, critic, pins, cleanup


def run(cfg: WriteConfig, *, now=time.time) -> dict:
    slots, sides, pack, trait_names, gates, writer, critic, pins, cleanup = build(cfg, now=now)
    out = cfg.out
    if cfg.mock:
        out = out / "mock"  # never confused with data
    excerpts = load_pool(cfg.pool) if cfg.pool else None
    exemplars = None
    if cfg.exemplar_repo and cfg.exemplar_commit and cfg.exemplar_files:
        try:
            exemplars = load_exemplars(cfg.exemplar_repo, cfg.exemplar_commit, list(cfg.exemplar_files))
            pins["exemplar_commit"] = cfg.exemplar_commit
        except ExemplarError as e:
            raise WriteRefused(str(e)) from e
    runner = Runner(gates=gates, pack=pack, trait_names=trait_names, writer=writer, critic=critic, out_dir=out, concurrency=cfg.concurrency, pins=pins,
                    excerpts=excerpts, exemplars=exemplars)
    halted = None
    try:
        outcomes = runner.run(slots, sides)
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
        "mock": cfg.mock,
        "slots": len(slots),
        "outcomes": len(outcomes),
    }
