"""Wave mode for `role-set write` (BUILD-SPEC B1): write, critique and rewrite as separate runs.

The writer and the critic no longer have to be loaded at once, or on one host. A write wave runs the writer and
every mechanical check; a candidate that passes waits in the drafts file. A critique wave runs the critic and the
acceptance checks on those candidates. A rewrite wave continues each refused draft's own conversation. The drafts
file is the only hand-off between waves: it carries every slot's conversation, attempts, refusals and state, the
batch's pins, the andon's counts and every wave's record.

One attempt budget per slot runs across all waves (MAX_ATTEMPTS). A slot that spends it is dropped, and the andon
counts the drop in whichever wave it happens. A draft written under other pins is refused: the batch pins (plan,
pack, partition, gates, exemplars, stuck rule, training) must match in every wave, and the writer pins must match
in a rewrite wave. Each wave records its input file's sha256, so the waves chain.

Standards: PIN_PER_STEP (batch and writer pins checked per wave; every wave's pins and input hash recorded),
ANDON_AUTHORITY (the runner's cell, kind and systemic andons, carried across waves), DECOMPOSE_BY_SECRETS (each
wave holds one model; the drafts file is the seam). NAMED_COMPENSATORS: a wave writes only its out directory and
the drafts file; undo is deleting that directory.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from role_set.assemble import Material
from role_set.cards import PIN_KEYS as CARD_KEYS
from role_set.plan import Slot
from role_set.runner import MAX_ATTEMPTS, Outcome, RunHalted, Runner, SlotState, interleave
from role_set.transport import Completion

WAVES = ("write", "critique", "rewrite")
FORMAT = "role-set waves 1"
DRAFTS_FILE = "wave-drafts.jsonl"
PENDING = ("awaiting_critique", "needs_rewrite")
FINAL = ("accepted", "dropped", "skipped")
# Pins that belong to one wave's host, not to the batch: they may differ from wave to wave.
WAVE_LOCAL = ("grant", "machine", "wave", "drafts_sha256", "llama_server_commit", "llama_server_sha256", "writers_list_sha256")


class WaveError(RuntimeError):
    """A drafts file that can't be continued: wrong format, other pins, or nothing for this wave."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_writer_pin(k: str) -> bool:
    # The writer's card (cards.pins: card_*, cards_sha256) is the writer's, like its GGUF: a critique wave loads no
    # writer, so it has no card pins to match. Window 12b's critique was refused when they were counted as batch pins.
    return k.startswith(("writer_", "card_")) or k == "cards_sha256"


def batch_pins(pins: dict) -> dict:
    return {k: v for k, v in pins.items() if k not in WAVE_LOCAL and not _is_writer_pin(k) and not k.startswith("critic_")}


def writer_pins(pins: dict) -> dict:
    return {k: v for k, v in pins.items() if _is_writer_pin(k)}


def header_writer_pins(header: dict) -> dict:
    """The drafts' writer pins, including card pins that a header written before 2026-10-17 kept among its batch pins."""

    return {**writer_pins(header.get("batch_pins") or {}), **(header.get("writer_pins") or {})}


# ---- the drafts file ----------------------------------------------------------------------------------------

def seal_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def read(path: Path) -> tuple[dict, list[dict]]:
    """The header and the draft records. Refused when the file doesn't match the seal its wave wrote beside it
    (the hand-off crosses hosts), or when it isn't a drafts file of this format."""

    try:
        raw = path.read_bytes()
        seal = seal_path(path).read_text(encoding="utf-8").split()[0].strip().lower()
    except (OSError, IndexError) as e:
        raise WaveError(f"REFUSED: cannot read the drafts file and its seal {seal_path(path).name} ({type(e).__name__})") from e
    got = hashlib.sha256(raw).hexdigest()
    if got != seal:
        raise WaveError(f"REFUSED: {path.name} ({got[:12]}) doesn't match its seal ({seal[:12]}); it changed in transit, so fetch it again")
    try:
        lines = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    except ValueError as e:
        raise WaveError(f"REFUSED: cannot read the drafts file ({type(e).__name__})") from e
    if not lines or lines[0].get("type") != "header" or lines[0].get("format") != FORMAT:
        raise WaveError(f"REFUSED: {path.name} is not a drafts file ({FORMAT})")
    drafts = lines[1:]
    if any(d.get("type") != "draft" for d in drafts):
        raise WaveError("REFUSED: the drafts file holds a line that is not a draft")
    return lines[0], drafts


def write(path: Path, header: dict, drafts: list[dict]) -> str:
    """The drafts file and its seal, <name>.sha256 beside it, in sha256sum's format. Returns the hash."""

    data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in [header, *drafts]).encode("utf-8")
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    seal_path(path).write_text(f"{sha}  {path.name}\n", encoding="utf-8", newline="\n")
    return sha


def require_same_pins(header: dict, pins: dict, gates, wave: str) -> None:
    """The batch pins must match in every wave; the writer pins too in a rewrite wave."""

    have, want = batch_pins(header.get("batch_pins") or {}), batch_pins(pins)
    diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
    if diff:
        raise WaveError(f"REFUSED: the drafts were written under other batch pins ({', '.join(diff)}); a wave continues only its own batch")
    g = header.get("gates") or {}
    if (g.get("version"), g.get("sha256")) != (gates.version, gates.sha256):
        raise WaveError(f"REFUSED: the drafts were checked by gates {g.get('version')}; this wave has {gates.version}")
    if wave == "rewrite":
        have_w, want_w = header_writer_pins(header), writer_pins(pins)
        diff = sorted(k for k in set(have_w) | set(want_w) if have_w.get(k) != want_w.get(k))
        if diff:
            raise WaveError(f"REFUSED: a rewrite continues the drafts' own writer; these pins differ: {', '.join(diff)}")


def chain_card_pins(header: dict, prior: Path) -> tuple[dict, str]:
    """The writer's card pins for drafts whose header lost them, taken whole from the write wave's drafts they came from
    (stub rung 12s5a; R&D and Argon, 2026-10-18). A critique wave before the fix dropped an old header's card pins, which
    sat among its batch pins; its input, the write wave's sealed drafts, still holds them. One hop only: these drafts are
    one write wave and one critique, and `prior` is the exact file that critique read. Returns the seven pins and the
    prior's sha256; any miss refuses, and pins are never taken partly from one file and partly from the other."""

    def refuse(why: str):
        raise WaveError(f"REFUSED: the card-pin chain to {prior.name}: {why}")

    have = header_writer_pins(header)
    if CARD_KEYS & (set(have) | set(header.get("batch_pins") or {})):  # (i)
        refuse("these drafts record card pins of their own, and a prior never replaces them")
    hops = header.get("waves") or []
    if [w.get("wave") for w in hops] != ["write", "critique"] or hops[0].get("input_sha256"):
        refuse(f"these drafts are waves {[w.get('wave') for w in hops]}; the chain takes one write wave and one critique")
    p_header, _ = read(prior)  # (ii): the prior's seal, or this raises
    p_hops = p_header.get("waves") or []
    if [w.get("wave") for w in p_hops] != ["write"] or p_hops[0].get("input_sha256"):
        refuse(f"the prior is waves {[w.get('wave') for w in p_hops]}, not one write wave's drafts")
    sha = file_sha256(prior)
    if sha != hops[-1].get("input_sha256"):  # (iii)
        refuse(f"its sha256 {sha[:12]} is not the input {str(hops[-1].get('input_sha256'))[:12]} of the critique that wrote these drafts")
    p_pins = header_writer_pins(p_header)
    missing = sorted(CARD_KEYS - set(p_pins))
    if missing:  # (iv)
        refuse(f"the prior lacks card pins {', '.join(missing)}")
    diff = sorted(k for k in set(p_pins) | set(have) if k not in CARD_KEYS and p_pins.get(k) != have.get(k))
    diff += [k for k in sorted(set(batch_pins(p_header.get("batch_pins") or {})) | set(batch_pins(header.get("batch_pins") or {})))
             if batch_pins(p_header.get("batch_pins") or {}).get(k) != batch_pins(header.get("batch_pins") or {}).get(k)]
    p_gates, gates = p_header.get("gates") or {}, header.get("gates") or {}
    if not p_gates.get("sha256"):
        diff.append("gates (the prior records no gates sha256)")
    diff += [f"gates {k}" for k in ("version", "sha256", "closed_form") if p_gates.get(k) != gates.get(k)]
    if diff:  # (v)
        refuse(f"the prior's pins differ from these drafts' ({', '.join(diff)})")
    return {k: p_pins[k] for k in sorted(CARD_KEYS)}, sha


def _completion_dict(c) -> dict:
    return {"text": c.text, "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens, "model": c.model,
            "seconds": c.seconds, "finish_reason": getattr(c, "finish_reason", None)}


def record(st: SlotState, state: str, *, completion=None, receipt: dict | None = None, row: dict | None = None,
           model: dict | None = None, lookup: bool = False, seq: int | None = None) -> dict:
    return {
        "type": "draft", "slot": st.slot.as_dict(), "state": state, "attempt": st.attempt, "ended": st.ended,
        "messages": st.messages,
        "material": {"text": st.material.text, "source": st.material.source} if st.material else None,
        "facts": st.facts, "spent": st.spent, "refusals": st.refusals,
        "candidate": _completion_dict(completion) if completion is not None else None,
        "row": row, "model": model, "lookup": lookup, "receipt": receipt, "accepted_seq": seq,
    }


def bare(slot: Slot, state: str, *, refusals: list | None = None, skip_reason: str | None = None) -> dict:
    """A slot that never reached a draft: skipped, or dropped before its first writer call."""

    return {"type": "draft", "slot": slot.as_dict(), "state": state, "attempt": 0, "ended": False, "messages": [],
            "material": None, "facts": None, "spent": [], "refusals": list(refusals or []), "candidate": None,
            "row": None, "model": None, "lookup": False, "receipt": None, "accepted_seq": None, "skip_reason": skip_reason}


def state_of(d: dict) -> SlotState:
    m = d.get("material")
    return SlotState(slot=Slot(**d["slot"]), messages=list(d["messages"]), material=Material(m["text"], m["source"]) if m else None,
                     facts=d.get("facts"), attempt=d["attempt"], spent=list(d.get("spent") or []),
                     refusals=list(d.get("refusals") or []), ended=bool(d.get("ended")))


def completion_of(d: dict) -> Completion:
    c = d["candidate"]
    return Completion(text=c["text"], prompt_tokens=c.get("prompt_tokens"), completion_tokens=c.get("completion_tokens"),
                      model=c.get("model") or "", seconds=c.get("seconds") or 0.0, finish_reason=c.get("finish_reason"))


# ---- the andon across waves ---------------------------------------------------------------------------------

def save_andon(runner: Runner) -> dict:
    s = runner.status
    return {
        "halted": s.halted, "lesson_halts": s.lesson_halts, "cell_halts": s.cell_halts, "kinds_off": s.kinds_off,
        "cell_drops": {f"{k[0]}/{k[1]}": v for k, v in runner.cell_drops.items()},
        "cell_drop_keys": {f"{k[0]}/{k[1]}": dict(v) for k, v in runner.cell_drop_keys.items()},
        "halt_order": [[lesson, list(key)] for lesson, key in runner.halt_order],
    }


def load_andon(runner: Runner, andon: dict) -> None:
    s = runner.status
    s.halted = andon.get("halted")
    s.lesson_halts, s.cell_halts, s.kinds_off = dict(andon.get("lesson_halts") or {}), dict(andon.get("cell_halts") or {}), dict(andon.get("kinds_off") or {})
    runner.cell_drops = {tuple(k.split("/", 1)): v for k, v in (andon.get("cell_drops") or {}).items()}
    runner.cell_drop_keys = {tuple(k.split("/", 1)): Counter(v) for k, v in (andon.get("cell_drop_keys") or {}).items()}
    runner.halt_order = [(lesson, tuple(key)) for lesson, key in (andon.get("halt_order") or [])]


def _draft_row(runner: Runner, d: dict) -> dict | None:
    """Assemble the candidate row of a surviving training draft (accepted or awaiting_critique) without mutating runner."""

    if isinstance(d.get("row"), dict):
        return d["row"]
    if not isinstance(d.get("candidate"), dict):
        return None
    from role_set.assemble import AssemblyError, assemble, slot_flaw  # noqa: PLC0415
    from role_set.prompts import parse_json_object  # noqa: PLC0415
    from role_set.runner import normalize_glyphs  # noqa: PLC0415
    from role_set.solvefirst import SolveFirstError  # noqa: PLC0415

    st, c = state_of(d), completion_of(d)
    try:
        fields, _ = normalize_glyphs(parse_json_object(c.text))
        locked = runner.locked_for(st.slot)
        if locked is not None:
            fields["model"], fields["answer"] = locked.model, locked.key
        if st.slot.kind == "C" and "planted_flaw" not in fields:
            fields["planted_flaw"] = slot_flaw(st.slot)
        trait = runner.trait_names.get(st.slot.lesson, st.slot.lesson)
        return assemble(st.slot, trait, fields, st.material)
    except (ValueError, AssemblyError, SolveFirstError):
        return None


def preload(runner: Runner, drafts: list[dict], *, include_awaiting: bool = True) -> None:
    """The batch's accepted rows, in the order they were accepted: the acceptance checks of a later wave run
    against them, as they would inline. Surviving non-rewritten training drafts also seed the SPENT set."""

    done = sorted((d for d in drafts if d["state"] == "accepted"), key=lambda d: d["accepted_seq"])
    for d in done:
        row, slot = d["row"], Slot(**d["slot"])
        runner.accepted.append(row)
        if slot.kind == "B" and d.get("model") is not None:
            runner.models[row["id"]] = d["model"]
        if d.get("lookup"):
            runner.lookups[slot.lesson] = runner.lookups.get(slot.lesson, 0) + 1
        framing = row["turns"][0]["user"].split("\n", 1)[0]
        if framing and slot.kind in ("C", "D", "E"):
            runner.avoid.append(framing)
    if include_awaiting:
        for d in drafts:
            if d["state"] == "awaiting_critique" and d["slot"].get("split") != "apply":
                row = _draft_row(runner, d)
                if row is not None:
                    runner.note_training_row(row)
    for d in drafts:
        runner.avoid.extend(x for x in d.get("spent") or [] if x not in runner.avoid)
    # The earlier waves' counts carry over, so the last wave's report is the batch's (Argon §9.4 #8).
    runner.status.accepted += len(done)
    runner.status.dropped += sum(1 for d in drafts if d["state"] == "dropped")


# ---- the waves ----------------------------------------------------------------------------------------------

def _drop(runner: Runner, st: SlotState) -> dict:
    out = Runner.outcome_of(st)
    with runner.lock:
        runner.finished.append(out)
        runner._log("dropped", slot=st.slot.id, codes=[r["code"] for r in out.refusals])
        runner._andon(st.slot, out)
    return record(st, "dropped")


def _accepted(runner: Runner, st: SlotState, cand) -> dict:
    with runner.lock:
        seq = runner.accept_order[st.slot.id]
        runner.finished.append(Runner.outcome_of(st, cand.row))
    model = cand.fields.get("model") if st.slot.kind == "B" else None
    return record(st, "accepted", completion=cand.completion, receipt=runner.receipts.get(st.slot.id), row=cand.row,
                  model=model, lookup=cand.lookup, seq=seq)


def _write_one(runner: Runner, st: SlotState) -> dict:
    """The writer until a candidate passes the mechanical checks, or the budget is spent."""

    cand = runner.write_step(st)
    if cand is None:
        return _drop(runner, st)
    runner.note_training_row(cand.row)
    # Every writer call is logged; this one waits for the critic.
    runner._record_call(st.slot, st.attempt, cand.completion, "awaiting_critique", cand.glyphs, st.material)
    return record(st, "awaiting_critique", completion=cand.completion, lookup=cand.lookup)


def _critique_one(runner: Runner, d: dict) -> dict:
    """The critic and the acceptance checks on one candidate. The mechanical checks run again first, on the stored
    draft, so a draft that changed in transit, or a lookup cap filled since, is refused, never accepted."""

    st, c = state_of(d), completion_of(d)
    refusal, cand, glyphs = runner.mechanical(st, c)
    if cand is not None:
        refusal = runner.critique_step(st, cand)
        if refusal is None:
            return _accepted(runner, st, cand)
        glyphs = cand.glyphs
    runner.refuse(st, c, refusal, glyphs)
    if st.attempt >= MAX_ATTEMPTS:
        return _drop(runner, st)
    return record(st, "needs_rewrite")


AVOID_HEADER = "## Do not use these framings or phrases; they are spent\n"
PROMPT_TAIL = "\n\nReturn the JSON object now."


def refresh_avoid(messages: list[dict], avoid: list[str]) -> list[dict]:
    """The writer's first prompt with its spent list rebuilt from what the batch has spent since (Argon §9.4 #7): a
    rewrite wave sees the framings accepted in the critique wave and the exclusion state, as an inline retry would."""

    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") != "user":
            continue
        text = m["content"]
        block = AVOID_HEADER + "\n".join(f"- {a}" for a in avoid) if avoid else ""
        at = text.find(AVOID_HEADER)
        if at >= 0:
            end = text.find("\n\n", at + len(AVOID_HEADER))
            end = len(text) if end < 0 else end
            m["content"] = text[:at] + block + text[end:] if block else (text[:at].rstrip("\n") + text[end:])
        elif block:
            m["content"] = (text[: -len(PROMPT_TAIL)] + "\n\n" + block + PROMPT_TAIL if text.endswith(PROMPT_TAIL)
                            else text + "\n\n" + block)
        break
    return out


def _rewrite_one(runner: Runner, d: dict) -> dict:
    st = state_of(d)
    st.messages = refresh_avoid(st.messages, list(runner.avoid[-60:]))
    return _write_one(runner, st)


def _each(runner: Runner, items: list, fn, *, halted_state) -> list[dict]:
    """fn over items, in order or on the runner's pool. A systemic halt stops the wave: the item in hand and the
    ones after it keep the state they came in with (halted_state builds it), and the halt is recorded."""

    results: list = [None] * len(items)

    def one(i: int):
        if runner.status.halted is not None:
            results[i] = halted_state(items[i])
            return
        try:
            results[i] = fn(items[i])
        except RunHalted:
            results[i] = halted_state(items[i])

    if runner.status.concurrency == 1:
        for i in range(len(items)):
            one(i)
    else:
        with ThreadPoolExecutor(max_workers=runner.status.concurrency) as pool:
            list(pool.map(one, range(len(items))))
    return results


def write_wave(runner: Runner, slots: list[Slot], *, stop_after_s: float | None = None) -> list[dict]:
    if stop_after_s is not None:
        runner.deadline = runner.clock() + stop_after_s
    runner.plan_kinds = {s.kind for s in slots}
    lessons: list[str] = []
    for s in slots:
        if s.lesson not in lessons:
            lessons.append(s.lesson)
    for lesson in lessons:
        if not runner.warmup(lesson):
            runner._halt_lesson_warmup(lesson)
    ordered = interleave(slots)

    def fn(slot: Slot) -> dict:
        with runner.lock:
            why = runner._skip(slot)
            if why:
                runner.status.skipped[why] = runner.status.skipped.get(why, 0) + 1
                return bare(slot, "skipped", skip_reason=why)
            runner.status.attempted += 1
        st = runner.open_slot(slot)
        if isinstance(st, Outcome):
            with runner.lock:
                runner.finished.append(st)
                runner._andon(slot, st)
            return bare(slot, "dropped", refusals=st.refusals)
        return _write_one(runner, st)

    train_slots = [s for s in ordered if s.split != "apply"]
    apply_slots = [s for s in ordered if s.split == "apply"]
    halted_fn = lambda slot: bare(slot, "skipped", skip_reason="run halted during the slot")  # noqa: E731
    return _each(runner, train_slots, fn, halted_state=halted_fn) + _each(runner, apply_slots, fn, halted_state=halted_fn)


def run_wave(runner: Runner, wave: str, *, slots: list[Slot], header_in: dict | None, drafts_in: list[dict] | None,
             stop_after_s: float | None = None) -> tuple[dict, list[dict]]:
    """One wave. Returns the new header and every draft of the batch, in plan order."""

    if wave not in WAVES:
        raise WaveError(f"REFUSED: unknown wave {wave!r}")
    if wave == "write":
        if drafts_in:
            raise WaveError("REFUSED: a write wave starts a batch; it takes no drafts file")
        drafts = write_wave(runner, slots, stop_after_s=stop_after_s)
    else:
        assert header_in is not None and drafts_in is not None
        # The drafts hold every slot of this plan, in the plan's order (Argon §9.4 #6): a truncated, reordered or
        # padded file is refused, even under a matching seal.
        want_slots = json.loads(json.dumps([s.as_dict() for s in interleave(slots)]))
        if [d["slot"] for d in drafts_in] != want_slots:
            raise WaveError(f"REFUSED: the drafts hold {len(drafts_in)} slots that don't match this plan's "
                            f"{len(want_slots)} in order; a wave continues only its own batch, whole")
        load_andon(runner, header_in.get("andon") or {})
        runner.plan_kinds = {Slot(**d["slot"]).kind for d in drafts_in}
        preload(runner, drafts_in, include_awaiting=(wave != "critique"))
        want = "awaiting_critique" if wave == "critique" else "needs_rewrite"
        todo = [i for i, d in enumerate(drafts_in) if d["state"] == want]
        if not todo:
            raise WaveError(f"REFUSED: no draft is {want.replace('_', ' ')}; nothing for a {wave} wave")
        fn = _critique_one if wave == "critique" else _rewrite_one
        train_todo = [i for i in todo if drafts_in[i]["slot"].get("split") != "apply"]
        apply_todo = [i for i in todo if drafts_in[i]["slot"].get("split") == "apply"]
        done = (
            _each(runner, [drafts_in[i] for i in train_todo], lambda d: fn(runner, d), halted_state=lambda d: d)
            + _each(runner, [drafts_in[i] for i in apply_todo], lambda d: fn(runner, d), halted_state=lambda d: d)
        )
        drafts = list(drafts_in)
        for i, d in zip(train_todo + apply_todo, done):
            drafts[i] = d
    header = dict(header_in or {})
    header["andon"] = save_andon(runner)
    return header, drafts


def finish(runner: Runner, wave: str, drafts: list[dict], out_dir: Path) -> str | None:
    """The batch check, once no draft is pending: the share-based gates over the finished lessons' rows. The wave's
    report lists every slot's state, so the last wave's report is the batch's."""

    pending = sum(1 for d in drafts if d["state"] in PENDING)
    skipped = sum(1 for d in drafts if d["state"] == "skipped")
    lessons: list[str] = []
    for d in drafts:
        if d["slot"]["lesson"] not in lessons:
            lessons.append(d["slot"]["lesson"])
    whole = [ls for ls in lessons if ls not in runner.status.lesson_halts]
    rows = [r for r in runner.accepted if r.get("lesson") in whole]
    batch_check = None
    # As inline, where a halted cell leaves slots unstarted: a batch the andon stopped anywhere gets no batch check.
    stopped = runner.status.halted or runner.status.cell_halts or runner.status.kinds_off or runner.status.lesson_halts
    if pending == 0 and not stopped and not skipped and rows:
        try:
            runner.gates.check_batch(rows, models=runner.models)
            batch_check = "clean"
        except runner.gates.error_type() as e:
            batch_check = f"REFUSED: {e}"
        runner._log("batch_check", result=batch_check, lessons=whole)
    runner.batch_check = batch_check
    if pending == 0:
        write_batch(drafts, out_dir)
    counts = Counter(d["state"] for d in drafts)
    report = {
        "wave": wave, "pending": pending, "states": dict(sorted(counts.items())), "batch_check": batch_check,
        "status": runner.status.as_dict(runner.clock()),
        "outcomes": [{"slot": d["slot"]["id"], "state": d["state"], "attempts": d["attempt"], "refusals": d["refusals"],
                      "skip_reason": d.get("skip_reason")} for d in drafts],
    }
    (out_dir / "batch-report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return batch_check


def write_batch(drafts: list[dict], out_dir: Path) -> None:
    """The whole batch's data, in the wave that leaves nothing pending (Argon §9.4 #8): every accepted row in the order
    it was accepted, across all waves, with its receipt and, for a kind B row, its `.models.jsonl` line. These replace
    the wave's own files, so one folder holds the batch the store ingests."""

    from role_set.runner import models_line  # noqa: PLC0415

    done = sorted((d for d in drafts if d["state"] == "accepted"), key=lambda d: d["accepted_seq"])
    rows, receipts, models = [], [], []
    for d in done:
        line = json.dumps(d["row"], ensure_ascii=False)
        rows.append(line)
        if d.get("receipt") is not None:
            receipts.append(json.dumps(d["receipt"], ensure_ascii=False))
        if d["slot"]["kind"] == "B" and d.get("model") is not None:
            models.append(json.dumps(models_line(d["row"]["id"], d["slot"].get("structure"), line, d["model"]), ensure_ascii=False))
    for name, lines in (("accepted.jsonl", rows), ("receipts.jsonl", receipts), ("accepted.models.jsonl", models)):
        path = out_dir / name
        if lines:
            path.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
        else:
            path.unlink(missing_ok=True)


# ---- the mock's answers -------------------------------------------------------------------------------------

def dump_answers(answers: dict) -> list:
    """The mock writer's answer lookup, for the mock critic of a later wave (mock runs only; never data)."""

    return [[list(k) if isinstance(k, tuple) else k, list(v) if isinstance(v, tuple) else v] for k, v in answers.items()]


def load_answers(header: dict) -> dict:
    out: dict = {}
    for k, v in header.get("mock_answers") or []:
        out[tuple(k) if isinstance(k, list) else k] = tuple(v) if isinstance(v, list) and len(v) == 2 and isinstance(v[1], list) else v
    return out
