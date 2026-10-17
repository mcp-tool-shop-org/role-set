"""`role-set write`: the runner wired to the real writer and critic, or to the mocks for CI.

Refused before anything is sent: an endpoint off the loopback, a model not on
the writer list or listed without its terms record, weights or a llama-server binary
that differ from their pins, a critic of the writer's own family, a missing
partition or machine line, a step off a rented host (Colab or a pod) without a
grant or with a stale watchdog heartbeat, a pack that has drifted from its manifest, a missing gates
package. `--mock` runs the whole pipeline with no model; its output is
marked and is never data.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from role_set import domains, exclusion, proven, waves
from role_set.assemble import trait_name
from role_set.exemplars import ExemplarError, load_exclusions, load_exemplars
from role_set.gates import Gates, GatesUnavailable, load_gates
from role_set.guards import (
    GuardError,
    require_grant,
    require_heartbeat,
    require_local_endpoint,
    require_model_allowed,
)
from role_set.harvest import load_pool
from role_set.mock import MockCritic, MockWriter, mock_gates
from role_set.pack import PackError, verify
from role_set.plan import Slot
from role_set.prompts import load_pack
from role_set.runner import RunHalted, Runner, interleave
from role_set.stuckrule import StuckRuleError
from role_set.stuckrule import load as load_stuck
from role_set.transport import chat

# The writer prompt (lesson, kinds, amendments, slot brief, material, exemplars) measures about 5,800 tokens with the
# real pack; 16,384 leaves room for the reply and its thinking.
WRITER_NUM_CTX = 16384
# The writer samples at 0.7 (transport.chat's default); a local rehearsal writer at Ollama's 0 gave every retry of a
# slot the same draft, so the rehearsal could not show what a retry recovers.
WRITER_TEMPERATURE = 0.7
# The critic reads at 0 unless a check asks otherwise (the trait leg's three samples), as it did on Ollama.
CRITIC_TEMPERATURE = 0.0


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
    exemplar_exclude: Path | None = None  # roles/<role>/exemplar-exclude.json: reviewed ids never shown as exemplars
    stuck_repo: Path | None = None  # aspire-si checkout for the frozen stuck rule; defaults to the exemplar repo
    stuck_commit: str | None = None  # defaults to the exemplar commit
    stop_after_minutes: float | None = None  # no new slot starts after this; running slots finish
    closed_form: dict | None = None  # the gates' closed-form settings for this run; None takes the gates' defaults
    training: tuple[Path, ...] = ()  # the whole training set's data files (with .models.jsonl sidecars), for an apply plan
    partition: str | None = None  # production or trial:<name>, on every receipt (BUILD-SPEC B6)
    machine: Path | None = None  # the wave script's CUDA machine line (JSON)
    writer_gguf_sha256: str | None = None  # measured by the wave script before launch
    critic_gguf_sha256: str | None = None
    server_sha256: str | None = None  # the llama-server binary, measured by the wave script
    wave: str = "inline"  # inline, or one of write, critique, rewrite (BUILD-SPEC B1)
    exclusion_state: Path | None = None  # earlier batches' exclusion-state.json (BUILD-SPEC B4); None for the first batch
    drafts: Path | None = None  # a critique or rewrite wave's input: the previous wave's wave-drafts.jsonl


def _training(cfg: WriteConfig, slots: list[Slot], gates, pins: dict) -> tuple[list[dict], dict]:
    """The whole training set, for an apply plan's exclusions (aspire-si#127): its rows, and its sidecar models through
    the gates' own load_models (each line's row_sha256 checked). An apply plan without it is refused; the files' sha256
    go on the run's pins."""

    if not any(s.split == "apply" for s in slots):
        return [], {}
    if not cfg.training:
        raise WriteRefused("REFUSED: an apply plan needs --training, the whole training set's data files: apply items may "
                           "reuse no trained material or wording (aspire-si#127)")
    rows: list[dict] = []
    models: dict = {}
    pins["training"] = []
    for path in cfg.training:
        text = path.read_text(encoding="utf-8")
        rows += [json.loads(x) for x in text.splitlines() if x.strip()]
        models.update(gates.load_models(path))
        pins["training"].append({"file": path.name, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    return rows, models


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


def resolve_critic(listed: proven.ProvenList, name: str | None, endpoint: str | None, gguf_sha256: str | None,
                   writer: dict | None = None, *, allow_comparison: bool = False) -> tuple[str, str, dict]:
    """A run's critic: on the proven list, not comparison-only, another family than the writer, its weights as
    pinned, on the loopback. Returns the endpoint, the name and its pins."""

    c_endpoint = require_local_endpoint(endpoint or "")
    c_model = (name or "").strip()
    if not c_model or any(m in c_model.lower() for m in proven.CLOUD_MARKERS):
        raise GuardError("REFUSED: a local critic model is required, never a cloud tag")
    if c_model not in listed.critics:
        raise GuardError(f"REFUSED: {c_model} is not on the proven critic list")
    entry = listed.critics[c_model]
    if entry.get("role") == "critic-comparison" and not allow_comparison:
        raise GuardError(f"REFUSED: {c_model} is listed for critic-control comparison only, not as a run's critic")
    if writer is not None:
        proven.require_other_family(writer, entry)
    return c_endpoint, c_model, proven.require_pinned(entry, gguf_sha256)


def critic_caller(endpoint: str, model: str):
    """The critic on llama-server's OpenAI-compatible /v1, read at 0 by default."""

    def critic(messages, temperature: float = CRITIC_TEMPERATURE, seed: int = 0, **_):
        return chat(endpoint, model, messages, temperature=temperature, seed=seed)

    return critic


def _require_critic_policy(cfg: WriteConfig, partition: str) -> None:
    """Every counted row has its second reader (Argon §9.4 #1): --no-critic is for trial and mock runs only."""

    if cfg.no_critic and (partition == "production" or cfg.wave == "critique"):
        raise GuardError("REFUSED: --no-critic never runs on partition production or in a critique wave; every "
                         "counted row is read by a critic")


def build(cfg: WriteConfig, *, now=time.time):
    """Everything the runner needs, with every refusal raised before a byte is sent."""

    slots, sides, plan_sha = load_plan(cfg.plan)
    if not cfg.mock and any(domains.bears_domain(s) and not s.domain for s in slots):
        raise WriteRefused("REFUSED: the plan has no domains; plan it with --domains (BUILD-SPEC B2): a real batch sets every item with material in a pinned NAICS domain")
    try:
        manifest = verify(cfg.pack)
    except PackError as e:
        raise WriteRefused(str(e)) from e
    pack = load_pack(cfg.pack)
    trait_names = trait_names_from_pack(pack)
    pins: dict = {"plan_sha256": plan_sha, "pack_sha256": manifest.pack_sha256, "pack_commit": manifest.commit}

    header = None
    if cfg.wave not in ("inline", *waves.WAVES):
        raise WriteRefused(f"REFUSED: --wave must be inline, write, critique or rewrite, got {cfg.wave!r}")
    if cfg.wave in ("critique", "rewrite"):
        if cfg.drafts is None:
            raise WriteRefused(f"REFUSED: a {cfg.wave} wave needs --drafts, the previous wave's {waves.DRAFTS_FILE}")
        try:
            header, _ = waves.read(cfg.drafts)
        except waves.WaveError as e:
            raise WriteRefused(str(e)) from e
        pins["drafts_sha256"] = waves.file_sha256(cfg.drafts)
    elif cfg.drafts is not None:
        raise WriteRefused("REFUSED: --drafts is for a critique or rewrite wave")
    pins.setdefault("drafts_sha256", None)
    pins["wave"] = cfg.wave
    try:
        prior = exclusion.load(cfg.exclusion_state)
    except exclusion.ExclusionError as e:
        raise WriteRefused(str(e)) from e
    if plan_sha[:16] in prior.batches:
        raise WriteRefused(f"REFUSED: this plan's batch {plan_sha[:16]} is already in the exclusion state; a rerun starts "
                           "from the state before it")
    # The plan was made against one exclusion state; the run checks against the same one (Argon §9.4 #15).
    plan_data = json.loads(cfg.plan.read_text(encoding="utf-8"))
    planned = plan_data.get("exclusion_state_sha256")
    if planned and planned != prior.sha256:
        raise WriteRefused(f"REFUSED: the plan was made against exclusion state {planned[:12]}, and --exclusion-state is "
                           f"{prior.sha256[:12]}; pass the state the plan was made from")
    # A batch pin: every wave of the batch checks against the same earlier batches.
    pins["exclusion_state_sha256"] = prior.sha256
    pins["exclusion_batches"] = len(prior.batches)
    # The wave pilot (R&D, 2026-10-17): trial rows only, no exemplars (B7's receipt check isn't needed until batch 2),
    # and it never enters the exclusion state.
    pins["pilot"] = bool(plan_data.get("pilot"))
    if pins["pilot"]:
        if cfg.exemplar_repo or cfg.exemplar_commit or cfg.exemplar_files or cfg.exemplar_exclude:
            raise WriteRefused("REFUSED: a pilot plan runs without exemplars; drop the --exemplar-* arguments")
        if not cfg.mock and not (cfg.partition or "").startswith("trial:"):
            raise WriteRefused("REFUSED: a pilot plan writes only to a trial:<name> partition")
    uses_writer = cfg.wave != "critique"
    uses_critic = cfg.wave in ("inline", "critique") and not cfg.no_critic

    if cfg.mock:
        # The mock's drafts come from their slot and attempt, so the waves write what one inline run writes.
        order = {s.id: i for i, s in enumerate(interleave(slots))}
        answers = waves.load_answers(header) if header else {}
        writer = MockWriter(keyed_order=order, answers=answers) if uses_writer else None
        critic = MockCritic(answers=answers) if uses_critic else None
        try:
            gates: Gates = load_gates(closed_form=cfg.closed_form)
        except GatesUnavailable:
            gates = mock_gates()
        pins.update({"writer_model": "mock", "critic_model": "mock" if uses_critic else None, "mock": True, "partition": cfg.partition})
        pins["_mock_answers"] = answers  # carried into the drafts header by run(), never onto a receipt
        return slots, sides, pack, trait_names, gates, writer, critic, pins, None

    try:
        gates = load_gates(closed_form=cfg.closed_form)
        endpoint = require_local_endpoint(cfg.endpoint or "") if uses_writer else None
        listed = proven.load(cfg.writers_file)
        partition = proven.require_partition(cfg.partition)
        _require_critic_policy(cfg, partition)
        machine = proven.load_machine(cfg.machine)
        server = proven.require_server(listed.server, cfg.server_sha256)
        if uses_writer:
            model = require_model_allowed(cfg.model or "", sorted(listed.writers))
            entry = listed.writers[model]
            w_pins = proven.require_pinned(entry, cfg.writer_gguf_sha256)
        else:
            # A critique wave loads no writer, but the drafts' writer must still be on the list with the same pins
            # (Argon §9.4 #2): the family that bars a same-family critic comes from writers.json, never the header.
            drafted = (header or {}).get("writer_pins") or {}
            model = drafted.get("writer_model")
            if model not in listed.writers:
                raise GuardError(f"REFUSED: the drafts name writer {model!r}, which is not on the proven writer list")
            entry = listed.writers[model]
            listed_pins = proven.require_pinned(entry, drafted.get("writer_gguf_sha256"))
            drift = sorted(k for k, v in listed_pins.items() if drafted.get(f"writer_{k}") != v)
            if drift:
                raise GuardError(f"REFUSED: the drafts' writer pins differ from writers.json ({', '.join(drift)}); "
                                 "a critique wave reads only drafts its listed writer wrote")
            w_pins = None
        if proven.needs_grant(machine):
            require_grant(cfg.grant, uses_card=True)
            if cfg.heartbeat is None:
                raise GuardError("REFUSED: a step on the rig needs --heartbeat <watchdog heartbeat file>")
            require_heartbeat(cfg.heartbeat, now())
            pins["grant"] = cfg.grant
        critic = None
        if uses_critic:
            c_endpoint, c_model, c_pins = resolve_critic(listed, cfg.critic_model, cfg.critic_endpoint, cfg.critic_gguf_sha256, writer=entry)
            critic = critic_caller(c_endpoint, c_model)
            pins.update({"critic_model": c_model, **{f"critic_{k}": v for k, v in c_pins.items()}})
    except (GatesUnavailable, GuardError) as e:
        raise WriteRefused(str(e)) from e
    rehearsal = bool(entry.get("rehearsal_only")) if uses_writer else bool((header or {}).get("batch_pins", {}).get("rehearsal"))
    pins.update({"partition": partition, "llama_server_commit": server["commit"], "llama_server_sha256": server["sha256"],
                 "machine": machine, "writers_list_sha256": listed.sha256, "rehearsal": rehearsal})
    writer = None
    if uses_writer:
        pins.update({"writer_model": model, **{f"writer_{k}": v for k, v in w_pins.items()}})

        def writer(messages, seed=0):
            return chat(endpoint, model, messages, temperature=WRITER_TEMPERATURE, seed=seed)

    # llama-server has no unload: the wave script that launched it stops it, so the run leaves nothing to clean up.
    cleanup = None
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
            excluded: frozenset[str] = frozenset()
            if cfg.exemplar_exclude:
                excluded, sha = load_exclusions(cfg.exemplar_exclude)
                pins.update({"exemplar_exclude_sha256": sha, "exemplar_excluded": sorted(excluded)})
            exemplars = load_exemplars(cfg.exemplar_repo, cfg.exemplar_commit, list(cfg.exemplar_files), excluded)
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
    training, training_models = _training(cfg, slots, gates, pins)
    if cfg.wave != "inline":
        return _run_wave(cfg, slots, pack, trait_names, gates, writer, critic, pins, out, excerpts, exemplars, stuck, training, training_models)
    pins.pop("_mock_answers", None)
    prior = exclusion.load(cfg.exclusion_state)
    runner = Runner(gates=gates, pack=pack, trait_names=trait_names, writer=writer, critic=critic, out_dir=out, concurrency=cfg.concurrency, pins=pins,
                    excerpts=excerpts, exemplars=exemplars, stuck=stuck, training=training, training_models=training_models, prior=prior)
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
    state_sha = _leave_state(cfg, prior, runner, slots, halted=halted)
    return {
        "exclusion_state": state_sha,
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


def _run_wave(cfg: WriteConfig, slots, pack, trait_names, gates, writer, critic, pins, out, excerpts, exemplars, stuck, training, training_models) -> dict:
    """One wave of a batch (BUILD-SPEC B1). Its input is the previous wave's drafts file; its output is a new one,
    holding every slot of the batch, beside the wave's accepted rows, receipts and report."""

    answers = pins.pop("_mock_answers", None)
    header_in = drafts_in = None
    if cfg.wave in ("critique", "rewrite"):
        try:
            header_in, drafts_in = waves.read(cfg.drafts)
            waves.require_same_pins(header_in, pins, gates, cfg.wave)
        except waves.WaveError as e:
            raise WriteRefused(str(e)) from e
    run_pins = dict(pins)
    if cfg.wave == "critique":
        # A receipt names the writer that wrote the draft and the host it ran on, beside this wave's critic.
        run_pins.update({**(header_in.get("writer_pins") or {}), "writer_host": header_in.get("writer_host")})
    prior = exclusion.load(cfg.exclusion_state)
    runner = Runner(gates=gates, pack=pack, trait_names=trait_names, writer=writer, critic=critic, out_dir=out, concurrency=cfg.concurrency, pins=run_pins,
                    excerpts=excerpts, exemplars=exemplars, stuck=stuck, training=training, training_models=training_models, prior=prior)
    budget = cfg.stop_after_minutes * 60 if cfg.stop_after_minutes else None
    try:
        header, drafts = waves.run_wave(runner, cfg.wave, slots=slots, header_in=header_in, drafts_in=drafts_in, stop_after_s=budget)
    except waves.WaveError as e:
        raise WriteRefused(str(e)) from e
    header.update({"type": "header", "format": waves.FORMAT, "batch_pins": waves.batch_pins(pins),
                   "gates": {"version": gates.version, "sha256": gates.sha256, "closed_form": gates.closed_form}})
    if cfg.wave in ("write", "rewrite"):
        header["writer_pins"] = waves.writer_pins(pins)
        header["writer_host"] = {k: pins.get(k) for k in ("machine", "llama_server_commit", "llama_server_sha256")}
    if answers is not None:
        header["mock_answers"] = waves.dump_answers(answers)
    header["waves"] = [*(header.get("waves") or []),
                       {"wave": cfg.wave, "input_sha256": pins.get("drafts_sha256"),
                        "pins": {k: v for k, v in pins.items() if k in waves.WAVE_LOCAL or k.startswith(("writer_", "critic_"))}}]
    out_sha = waves.write(out / waves.DRAFTS_FILE, header, drafts)
    batch_check = waves.finish(runner, cfg.wave, drafts, out)
    states = {st: sum(1 for d in drafts if d["state"] == st) for st in (*waves.PENDING, *waves.FINAL)}
    pending = states["awaiting_critique"] + states["needs_rewrite"]
    state_sha = _leave_state(cfg, prior, runner, slots, halted=runner.status.halted) if not pending else None
    return {
        "exclusion_state": state_sha,
        "out_dir": out.name, "wave": cfg.wave, "drafts": waves.DRAFTS_FILE, "drafts_sha256": out_sha, "states": states,
        "accepted": runner.status.accepted, "dropped": runner.status.dropped, "attempted": runner.status.attempted,
        "halted": runner.status.halted, "lesson_halts": runner.status.lesson_halts, "skipped": runner.status.skipped,
        "batch_check": batch_check, "rehearsal": bool(pins.get("rehearsal")), "mock": cfg.mock, "slots": len(slots),
        "outcomes": len(drafts),
    }


def _leave_state(cfg: WriteConfig, prior, runner, slots, *, halted) -> str | None:
    """The exclusion state this batch leaves behind, written beside its rows once the batch is done (inline: at the
    end of the run; waves: in the wave that leaves nothing pending), and only when its batch check is clean. Otherwise
    it leaves none, and the next batch starts from the earlier state."""

    # Only a batch whose batch check came back clean enters the state (R&D on #75): a refused batch, or one with
    # halted cells (whose batch check doesn't run), leaves its rows out, so they can't steer later batches.
    if halted or not getattr(runner, "accepted", None) or getattr(runner, "batch_check", None) != "clean":
        return None
    if runner.pins.get("pilot"):  # a pilot's rows are trial rows: they never steer a real batch
        return None
    batch_id = runner.pins.get("plan_sha256", "")[:16]
    new = exclusion.with_batch(prior, batch_id, runner.accepted, runner.models, slots)
    return exclusion.save(new, runner.out_dir / exclusion.FILE)
