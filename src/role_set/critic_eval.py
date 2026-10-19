"""`role-set critic-eval` and `role-set critic-rescore-offline`: tooling for the critic-signal study.

`critic-eval` scores every item of a sealed set with the critic's verdict check and, for kind C, the flaw check only
(no trait votes), at the critic's card values, with one retry on a VOID verdict reply and call-level journaling.
`critic-rescore-offline` re-scores recorded replies against any key under P0, P2 and (given two seeds) P3, with no
model calls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from role_set import callcache, cards, proven, waves
from role_set.assemble import FLAWS, VERDICTS
from role_set.critic import CriticVerdict, _lines, compare_flaw, compare_verdict, flaw_messages, verdict_messages
from role_set.gates import Gates, GatesUnavailable, load_gates
from role_set.guards import GuardError
from role_set.mock import MockCritic, mock_gates
from role_set.pack import PackError, verify
from role_set.prompts import load_pack, parse_json_object
from role_set.rescore import WINDOW_RE, _window_tokens_from_paths
from role_set.transport import Completion
from role_set.write import critic_caller, resolve_critic

STATEMENT_RE = re.compile(r"^Statement:\s*(.+?)\s*$", re.M)
CHECKER_VERDICT_RE = re.compile(r"^Checker verdict:\s*(\S+)\s*$", re.M)
CONTROL_TYPES = frozenset({"d", "control", "controls"})
GPU_TYPES = frozenset({"d", "b", "e", "f", "control", "controls"})
KEY_ONLY_TYPES = frozenset({"a", "c", "g"})


class CriticEvalError(RuntimeError):
    """A critic-eval or critic-rescore-offline input, seal, pin, family or blinding check failed."""


@dataclass(frozen=True)
class CriticEvalConfig:
    items: Path
    pack: Path | None = None
    writers_file: Path = Path("writers.json")
    cards: Path | None = None
    critic_endpoint: str | None = None
    critic_model: str | None = None
    critic_gguf_sha256: str | None = None
    critic_seed: int = 0
    out: Path = Path("critic-eval")
    mock: bool = False
    call_journal: Path | None = None
    reuse_calls: Path | None = None


@dataclass(frozen=True)
class CriticRescoreOfflineConfig:
    eval_files: tuple[Path, ...]
    items: Path | None = None
    keys: tuple[Path, ...] = ()
    out: Path = Path("critic-rescore-offline")
    mock: bool = False


def _statement_from_user(user: str) -> str:
    m = STATEMENT_RE.search(user)
    return m.group(1).strip() if m else ""


def _trace_from_user(user: str) -> str:
    m = STATEMENT_RE.search(user)
    rest = user[m.end():] if m else user
    rest = re.sub(r"^Checker verdict:.*$", "", rest, flags=re.M)
    return rest.strip()


def _checker_verdict_from_user(user: str) -> str:
    m = CHECKER_VERDICT_RE.search(user)
    return m.group(1).strip() if m else ""


def _flaw_word(reply: str | None) -> str | None:
    if not reply or not isinstance(reply, str):
        return None
    word = reply.strip().split()[0].strip(".,:;'\"").lower() if reply.strip() else ""
    return word if word in FLAWS + ("none",) else None


def _sum_thinking(values: list[int | None]) -> int | None:
    """A check's or run's thinking token sum: None if unmeasured (empty or any call is None)."""
    if not values or any(v is None for v in values):
        return None
    return sum(int(v) for v in values if v is not None)


def extract_item_parts(d: dict) -> dict:
    """Extract id, perturbation_type, kind, material, statement, checker_trace, checker_verdict, and key(s) from a draft."""

    row = d.get("row") or {}
    slot = d.get("slot") or {}
    fields: dict = {}
    if isinstance(d.get("fields"), dict):
        fields = dict(d["fields"])
    elif isinstance((d.get("candidate") or {}).get("text"), str) and d["candidate"]["text"].strip():
        try:
            fields = parse_json_object(d["candidate"]["text"])
        except ValueError:
            fields = {}

    kind = str(d.get("kind") or slot.get("kind") or row.get("kind") or "D")
    item_id = str(d.get("item_id") or d.get("id") or row.get("id") or slot.get("id") or "")
    source_row_id = str(d.get("source_row_id") or d.get("source_id") or row.get("id") or slot.get("id") or item_id)
    perturbation_type = str(d.get("perturbation_type") or d.get("perturbation") or "d")

    mat_field = d.get("material")
    if isinstance(mat_field, dict) and isinstance(mat_field.get("text"), str):
        material = mat_field["text"]
    elif isinstance(mat_field, str):
        material = mat_field
    else:
        material = str(row.get("material") or "")

    turns = row.get("turns") or []
    user0 = str(turns[0].get("user") or "") if turns and isinstance(turns[0], dict) else ""

    statement = str(d.get("statement") or fields.get("statement") or row.get("statement") or _statement_from_user(user0) or "")
    checker_trace = str(
        d.get("checker_trace")
        if d.get("checker_trace") is not None
        else (fields.get("checker_trace") if fields.get("checker_trace") is not None else (row.get("checker_trace") or _trace_from_user(user0)))
    )
    checker_verdict = str(
        d.get("checker_verdict")
        if d.get("checker_verdict") is not None
        else (row.get("checker_verdict") or fields.get("checker_verdict") or _checker_verdict_from_user(user0) or "")
    )

    key_block = d.get("key") if isinstance(d.get("key"), dict) else {}
    quiz = row.get("quiz") if isinstance(row.get("quiz"), dict) else {}

    if "verdict" in key_block:
        key_verdict = str(key_block["verdict"])
    elif kind == "E" and "verdict" in quiz:
        key_verdict = str(quiz["verdict"])
    elif "verdict" in row:
        key_verdict = str(row["verdict"])
    elif fields.get("turn2_verdict"):
        key_verdict = str(fields["turn2_verdict"])
    elif slot.get("verdict"):
        key_verdict = str(slot["verdict"])
    else:
        key_verdict = ""

    if "deciding" in key_block:
        key_deciding = list(key_block["deciding"] or [])
    elif "deciding_lines" in key_block:
        key_deciding = list(key_block["deciding_lines"] or [])
    elif kind == "E" and "deciding" in quiz:
        key_deciding = list(quiz["deciding"] or [])
    elif "deciding" in row:
        key_deciding = list(row["deciding"] or [])
    elif isinstance(fields.get("turn2_deciding"), list):
        key_deciding = list(fields["turn2_deciding"])
    elif isinstance(fields.get("deciding"), list):
        key_deciding = list(fields["deciding"])
    else:
        key_deciding = []

    if "planted_flaw" in key_block:
        key_flaw = key_block["planted_flaw"]
    elif "flaw" in key_block:
        key_flaw = key_block["flaw"]
    elif "planted_flaw" in row:
        key_flaw = row["planted_flaw"]
    elif "planted_flaw" in fields:
        key_flaw = fields["planted_flaw"]
    else:
        key_flaw = None

    if isinstance(d.get("new_material"), list):
        new_lines = [str(x) for x in d["new_material"]]
    elif len(turns) > 1 and isinstance(turns[-1], dict) and isinstance(turns[-1].get("new_material"), list):
        new_lines = [str(x) for x in turns[-1]["new_material"]]
    elif isinstance(fields.get("turn2_new_material"), list):
        new_lines = [str(x) for x in fields["turn2_new_material"]]
    else:
        new_lines = []

    raw_tk = (
        key_block.get("turn_keys")
        if isinstance(key_block.get("turn_keys"), dict)
        else (
            d.get("turn_keys")
            if isinstance(d.get("turn_keys"), dict)
            else (row.get("turn_keys") if isinstance(row.get("turn_keys"), dict) else None)
        )
    )

    turn_keys_out: dict | None = None
    if isinstance(raw_tk, dict) and isinstance(raw_tk.get("1"), dict):
        t1 = raw_tk["1"]
        t1_verdict = str(t1.get("verdict") or "")
        t1_deciding = list(t1.get("deciding") if "deciding" in t1 else (t1.get("deciding_lines") or []))
        turn_keys_out = {
            **copy.deepcopy(raw_tk),
            "1": {**copy.deepcopy(t1), "verdict": t1_verdict, "deciding": t1_deciding},
        }
        t2_seen = (material.rstrip("\n") + "\n" + "\n".join(new_lines)) if new_lines else material
        verdict_turns = [
            {
                "turn": 1,
                "seen_material": material,
                "key_verdict": t1_verdict,
                "key_deciding": t1_deciding,
            },
            {
                "turn": 2,
                "seen_material": t2_seen,
                "key_verdict": key_verdict,
                "key_deciding": key_deciding,
            },
        ]
        seen_material = t2_seen
    else:
        seen_material = material
        if new_lines and key_block.get("turn") != 1 and not all(ln in material for ln in new_lines):
            seen_material = material.rstrip("\n") + "\n" + "\n".join(new_lines)
        verdict_turns = [
            {
                "turn": 1,
                "seen_material": seen_material,
                "key_verdict": key_verdict,
                "key_deciding": key_deciding,
            }
        ]

    return {
        "item_id": item_id,
        "source_row_id": source_row_id,
        "control_item_id": str(d.get("control_item_id") or f"{source_row_id}:d"),
        "perturbation_type": perturbation_type,
        "kind": kind,
        "material": material,
        "new_material": new_lines,
        "seen_material": seen_material,
        "statement": statement,
        "checker_trace": checker_trace,
        "checker_verdict": checker_verdict,
        "key_verdict": key_verdict,
        "key_deciding": key_deciding,
        "key_flaw": str(key_flaw) if key_flaw is not None else None,
        "turn_keys": turn_keys_out,
        "verdict_turns": verdict_turns,
    }


def _writer_for_item(d: dict, header: dict, listed: proven.ProvenList | None, *, mock: bool) -> tuple[dict, dict]:
    h_pins = waves.header_writer_pins(header)
    item_pins = d.get("writer_pins") if isinstance(d.get("writer_pins"), dict) else {}
    receipt = d.get("receipt") if isinstance(d.get("receipt"), dict) else {}
    row = d.get("row") if isinstance(d.get("row"), dict) else {}
    cand = d.get("candidate") if isinstance(d.get("candidate"), dict) else {}

    w_pins = {**h_pins, **{k: v for k, v in receipt.items() if k.startswith(("writer_", "card_"))}, **item_pins}
    for key in ("writer_model", "writer_family", "card_candidate", "writer_repo", "writer_gguf_file"):
        if d.get(key):
            w_pins[key] = d[key]
        elif row.get(key):
            w_pins[key] = row[key]
    if not w_pins.get("writer_model"):
        for fallback in (d.get("writer"), d.get("arm"), cand.get("model")):
            if isinstance(fallback, str) and fallback.strip():
                w_pins["writer_model"] = fallback.strip()
                break

    model = (w_pins.get("writer_model") or "").strip()
    writers = listed.writers if listed is not None else {}
    if model in writers:
        entry = dict(writers[model])
    elif w_pins.get("writer_family"):
        entry = {"served_as": model or "writer", "family": w_pins["writer_family"], "card": w_pins.get("card_candidate")}
    elif mock or model in ("", "mock"):
        entry = {"served_as": model or "mock", "family": ""}
    else:
        raise CriticEvalError(f"REFUSED: item {d.get('item_id') or d.get('id')!r} names writer {model!r}, which is not on the proven writer list")
    return w_pins, entry


def _forbidden_for_item(
    d: dict,
    header: dict,
    w_pins: dict,
    writer_entry: dict,
    listed: proven.ProvenList | None,
    base_tokens: set[str],
) -> set[str]:
    out: set[str] = set(base_tokens)
    for val in (
        d.get("arm"),
        d.get("writer"),
        d.get("writer_model"),
        (d.get("candidate") or {}).get("model") if isinstance(d.get("candidate"), dict) else None,
        w_pins.get("writer_model"),
        w_pins.get("card_candidate"),
        w_pins.get("writer_repo"),
        w_pins.get("writer_gguf_file"),
        writer_entry.get("served_as"),
        writer_entry.get("card"),
        writer_entry.get("repo"),
        writer_entry.get("gguf_file"),
    ):
        if isinstance(val, str) and val.strip() and val.strip() != "mock":
            out.add(val.strip())
    if listed is not None:
        for w in listed.writers.values():
            for k in ("served_as", "card", "repo", "gguf_file"):
                v = w.get(k)
                if isinstance(v, str) and v.strip() and v.strip() != "mock":
                    out.add(v.strip())
    for text_blob in (json.dumps(header, ensure_ascii=False), json.dumps(d.get("receipt") or {}, ensure_ascii=False)):
        for m in WINDOW_RE.findall(text_blob):
            if len(m) >= 7:
                out.add(m)
    return out


def _blind_critic(critic, forbidden: set[str], item_id: str):
    """Refuse-don't-strip: any forbidden writer, served_as or window string in the critic's messages refuses the run."""

    def wrapped(messages: list[dict], **kwargs):
        for m in messages:
            text = str(m.get("content") or "")
            for tok in forbidden:
                if tok and tok in text:
                    raise CriticEvalError(
                        f"REFUSED: unblinded identifier in critic messages at item {item_id}"
                    )
        return critic(messages, **kwargs)

    return wrapped


def run_critic_eval(cfg: CriticEvalConfig, *, critic=None) -> dict:
    """Run the critic's verdict check(s) and (for kind C) flaw check on every item of a sealed set."""

    if cfg.pack is not None:
        try:
            verify(cfg.pack)
            load_pack(cfg.pack)
        except PackError as e:
            raise CriticEvalError(str(e)) from e

    try:
        header, drafts = waves.read(cfg.items)
    except waves.WaveError as e:
        raise CriticEvalError(str(e)) from e
    input_sha256 = waves.file_sha256(cfg.items)

    listed: proven.ProvenList | None = None
    if cfg.writers_file and cfg.writers_file.exists():
        try:
            listed = proven.load(cfg.writers_file)
        except GuardError as e:
            raise CriticEvalError(str(e)) from e
    elif not cfg.mock:
        raise CriticEvalError(f"REFUSED: cannot read writers file {cfg.writers_file}")

    c_model = (cfg.critic_model or ("mock" if cfg.mock else "")).strip()
    c_pins: dict | None = None
    c_card_pins: dict | None = None
    critic_entry: dict = {"served_as": c_model, "family": ""}
    if listed is not None and c_model in listed.critics:
        critic_entry = dict(listed.critics[c_model])
    elif listed is not None and c_model in listed.writers:
        critic_entry = dict(listed.writers[c_model])

    if not cfg.mock:
        if listed is None:
            raise CriticEvalError(f"REFUSED: cannot read writers file {cfg.writers_file}")
        if cfg.cards is None:
            raise CriticEvalError("REFUSED: the critic runs at its card values; pass --cards <card-values.json>")
        try:
            c_endpoint, c_model, c_pins = resolve_critic(listed, cfg.critic_model, cfg.critic_endpoint, cfg.critic_gguf_sha256)
            critic_entry = dict(listed.critics[c_model])
            card_set, card_sha = cards.load(cfg.cards)
            c_card = cards.select(card_set, critic_entry, c_model)
            c_card_pins = cards.pins(c_card, card_sha)
        except GuardError as e:
            raise CriticEvalError(str(e)) from e
        if critic is None:
            critic = critic_caller(c_endpoint, c_model, c_card)
    elif cfg.cards is not None and cfg.cards.exists():
        try:
            card_set, card_sha = cards.load(cfg.cards)
            if c_model in {c.get("candidate") for c in card_set.get("candidates", []) if isinstance(c, dict)}:
                c_card = cards.select(card_set, critic_entry, c_model)
                c_card_pins = cards.pins(c_card, card_sha)
        except GuardError:
            pass

    c_family = (critic_entry.get("family") or "").strip() or None
    if not cfg.mock and not c_family:
        raise CriticEvalError(f"REFUSED: critic {c_model!r} has no family on record")

    closed_form = (header.get("gates") or {}).get("closed_form")
    try:
        gates: Gates = load_gates(closed_form=closed_form)
    except GatesUnavailable as e:
        if not cfg.mock:
            raise CriticEvalError(str(e)) from e
        gates = mock_gates(closed_form=closed_form)

    if cfg.mock and critic is None:
        header_answers = dict(waves.load_answers(header))
        answers = dict(header_answers)
        by_exact_mat: dict[tuple[str, str], tuple[str, list[str]]] = {}
        for d in drafts:
            parts = extract_item_parts(d)
            for vt in parts["verdict_turns"]:
                mat_t = vt["seen_material"].rstrip()
                first_line = mat_t.splitlines()[0].strip() if mat_t.strip() else ""
                if parts["statement"] and vt["key_verdict"]:
                    if vt["turn"] == 1 and (parts["statement"], first_line) not in answers:
                        answers[(parts["statement"], first_line)] = (vt["key_verdict"], list(vt["key_deciding"]))
                    elif vt["turn"] == 2:
                        if ("turn2", parts["statement"], first_line) not in answers:
                            answers[("turn2", parts["statement"], first_line)] = (vt["key_verdict"], list(vt["key_deciding"]))
                        if (parts["statement"], first_line) not in header_answers:
                            by_exact_mat.setdefault((parts["statement"], mat_t), (vt["key_verdict"], list(vt["key_deciding"])))
            if parts["kind"] == "C" and parts["checker_trace"]:
                trace_line = parts["checker_trace"].splitlines()[0].strip()
                if ("flaw", trace_line) not in answers and parts["key_flaw"]:
                    answers[("flaw", trace_line)] = parts["key_flaw"]
        base_mock = MockCritic(answers=answers)
        if by_exact_mat and hasattr(base_mock, "answers"):
            def _mock_with_turn_lookup(messages: list[dict], **kwargs):
                text = messages[-1]["content"] if messages else ""
                if "Material:\n" in text and "\n\nStatement: " in text:
                    mat_part, stmt_part = text.split("Material:\n", 1)[1].split("\n\nStatement: ", 1)
                    stmt = stmt_part.splitlines()[0].strip()
                    if (stmt, mat_part.rstrip()) in by_exact_mat:
                        v_ans, d_ans = by_exact_mat[(stmt, mat_part.rstrip())]
                        body = " | ".join(d_ans) if d_ans else "NONE"
                        return Completion(
                            text=f"VERDICT: {v_ans}\nDECIDING: {body}\nESCALATE: no",
                            prompt_tokens=100,
                            completion_tokens=50,
                            model="mock",
                            seconds=0.01,
                        )
                return base_mock(messages, **kwargs)

            critic = _mock_with_turn_lookup
        else:
            critic = base_mock

    cfg.out.mkdir(parents=True, exist_ok=True)
    journal_path = cfg.call_journal or (cfg.out / callcache.JOURNAL_FILE)
    reuse = callcache.load(cfg.reuse_calls) if cfg.reuse_calls else None
    journal = callcache.CallJournal(journal_path, reuse)
    journaled_critic = journal.wrap("critic", critic)

    base_tokens = _window_tokens_from_paths(cfg.items)
    tag = ((header.get("writer_host") or {}).get("machine") or {}).get("colab_release_tag")
    if isinstance(tag, str) and tag.strip():
        base_tokens.add(tag.strip())
    for w in header.get("waves") or []:
        wtag = ((w.get("pins") or {}).get("machine") or {}).get("colab_release_tag")
        if isinstance(wtag, str) and wtag.strip():
            base_tokens.add(wtag.strip())

    eval_path = cfg.out / "critic-eval.jsonl"
    eval_path.write_text("", encoding="utf-8")

    records: list[dict] = []
    skipped: list[dict] = []
    all_call_thinkings: list[int | None] = []
    thinking_measured_calls = 0
    thinking_unmeasured_calls = 0
    total_output = 0
    total_prompt = 0

    for idx, d in enumerate(drafts):
        parts = extract_item_parts(d)
        item_id = parts["item_id"] or f"item-{idx}"
        w_pins, w_entry = _writer_for_item(d, header, listed, mock=cfg.mock)
        w_model = w_pins.get("writer_model") or w_entry.get("served_as") or ""
        w_family = (w_entry.get("family") or w_pins.get("writer_family") or "").strip() or None
        if not cfg.mock and not w_family:
            raise CriticEvalError(f"REFUSED: item {item_id}'s writer {w_model!r} has no family on record")

        if w_family and c_family:
            try:
                proven.require_other_family(
                    {**w_entry, "served_as": w_model or "writer", "family": w_family},
                    {**critic_entry, "served_as": c_model or "critic", "family": c_family},
                )
            except GuardError as e:
                skipped.append({
                    "item_id": item_id,
                    "source_row_id": parts["source_row_id"],
                    "perturbation_type": parts["perturbation_type"],
                    "writer_model": w_model,
                    "writer_family": w_family,
                    "critic_model": c_model,
                    "critic_family": c_family,
                    "reason": str(e),
                })
                continue

        if parts["kind"] not in ("C", "D", "E"):
            raise CriticEvalError(f"REFUSED: item {item_id} has unsupported kind {parts['kind']!r} for critic-eval")
        if not parts["seen_material"].strip() or not parts["statement"].strip():
            raise CriticEvalError(f"REFUSED: item {item_id} lacks material or statement")
        for vt in parts["verdict_turns"]:
            if vt["key_verdict"] not in VERDICTS:
                raise CriticEvalError(
                    f"REFUSED: item {item_id} turn {vt['turn']} lacks a valid key verdict ({vt['key_verdict']!r})"
                )

        forbidden = _forbidden_for_item(d, header, w_pins, w_entry, listed, base_tokens)
        blinded_critic = _blind_critic(journaled_critic, forbidden, item_id)

        verdict_checks: list[dict] = []
        item_call_thinkings: list[int | None] = []
        v_output_total = 0
        v_prompt_total = 0

        for vt in parts["verdict_turns"]:
            v_msgs = verdict_messages(vt["seen_material"], parts["statement"])
            v: CriticVerdict | None = None
            v_raw = ""
            v_attempts = 0
            v_voids = 0
            turn_thinkings: list[int | None] = []
            v_output = 0
            v_prompt = 0
            for _ask in range(2):
                v_attempts += 1
                c = blinded_critic(v_msgs, seed=cfg.critic_seed)
                v_raw = c.text
                raw_t = getattr(c, "thinking_tokens", None)
                t_val = int(raw_t) if raw_t is not None else None
                turn_thinkings.append(t_val)
                item_call_thinkings.append(t_val)
                all_call_thinkings.append(t_val)
                if t_val is not None:
                    thinking_measured_calls += 1
                else:
                    thinking_unmeasured_calls += 1
                v_output += int(c.completion_tokens or 0)
                v_prompt += int(c.prompt_tokens or 0)
                v = compare_verdict(gates, c.text, vt["key_verdict"], vt["key_deciding"], vt["seen_material"])
                if v.code != "VOID":
                    break
                v_voids += 1

            assert v is not None
            v_output_total += v_output
            v_prompt_total += v_prompt
            verdict_checks.append({
                "turn": vt["turn"],
                "code": v.code,
                "ok": v.ok,
                "detail": v.detail,
                "raw_reply": v_raw,
                "critic_verdict": v.critic_verdict,
                "critic_deciding": list(v.critic_deciding or []),
                "attempts": v_attempts,
                "void_retries": v_voids,
                "thinking_tokens": _sum_thinking(turn_thinkings),
                "output_tokens": v_output,
                "prompt_tokens": v_prompt,
            })

        verdict_check = verdict_checks[-1]

        flaw_check: dict | None = None
        f_thinking: int | None = None
        f_output = 0
        f_prompt = 0
        critic_flaw: str | None = None
        if parts["kind"] == "C":
            if not parts["key_flaw"] or parts["key_flaw"] not in FLAWS + ("none",):
                raise CriticEvalError(f"REFUSED: kind-C item {item_id} lacks a valid planted_flaw key ({parts['key_flaw']!r})")
            f_msgs = flaw_messages(parts["material"], parts["statement"], parts["checker_trace"], parts["checker_verdict"])
            fc = blinded_critic(f_msgs, seed=cfg.critic_seed)
            f_res = compare_flaw(fc.text, parts["key_flaw"])
            raw_ft = getattr(fc, "thinking_tokens", None)
            f_thinking = int(raw_ft) if raw_ft is not None else None
            item_call_thinkings.append(f_thinking)
            all_call_thinkings.append(f_thinking)
            if f_thinking is not None:
                thinking_measured_calls += 1
            else:
                thinking_unmeasured_calls += 1
            f_output = int(fc.completion_tokens or 0)
            f_prompt = int(fc.prompt_tokens or 0)
            critic_flaw = _flaw_word(fc.text)
            flaw_check = {
                "code": f_res.code,
                "ok": f_res.ok,
                "detail": f_res.detail,
                "raw_reply": fc.text,
                "critic_flaw": critic_flaw,
                "thinking_tokens": f_thinking,
                "output_tokens": f_output,
                "prompt_tokens": f_prompt,
            }

        item_thinking = _sum_thinking(item_call_thinkings)
        item_output = v_output_total + f_output
        item_prompt = v_prompt_total + f_prompt
        total_output += item_output
        total_prompt += item_prompt

        rec = {
            "item_id": item_id,
            "id": item_id,
            "source_row_id": parts["source_row_id"],
            "perturbation_type": parts["perturbation_type"],
            "kind": parts["kind"],
            "seed": cfg.critic_seed,
            "verdict_checks": verdict_checks,
            "verdict_check": verdict_check,
            "flaw_check": flaw_check,
            "critic_verdict": verdict_check["critic_verdict"],
            "critic_deciding": list(verdict_check["critic_deciding"] or []),
            "critic_flaw": critic_flaw,
            "thinking_tokens": item_thinking,
            "output_tokens": item_output,
            "prompt_tokens": item_prompt,
        }
        records.append(rec)
        with eval_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    gguf_sha = (c_pins or {}).get("critic_gguf_sha256") or cfg.critic_gguf_sha256 or critic_entry.get("gguf_sha256")
    report = {
        "critic_model": c_model,
        "critic_family": c_family,
        "critic_gguf_sha256": gguf_sha,
        "critic_seed": cfg.critic_seed,
        "critic_pins": c_pins,
        "critic_card": c_card_pins,
        "gates": {"version": gates.version, "sha256": gates.sha256, "closed_form": gates.closed_form},
        "input_sha256": input_sha256,
        "items_total": len(drafts),
        "items_scored": len(records),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "calls": journal.summary(),
        "tokens": {
            "thinking_tokens": _sum_thinking(all_call_thinkings),
            "thinking_measured_calls": thinking_measured_calls,
            "thinking_unmeasured_calls": thinking_unmeasured_calls,
            "output_tokens": total_output,
            "prompt_tokens": total_prompt,
        },
    }
    (cfg.out / "critic-eval-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


# ---- offline re-scorer (P0, P2, P3) -------------------------------------------------------------------------

def _load_key_records(path: Path) -> tuple[dict | None, list[dict]]:
    """Read items or alternate keys from a sealed waves file or a JSONL file (checking its .sha256 seal if present)."""

    if not path.is_file():
        raise CriticEvalError(f"REFUSED: cannot read items/key file {path}")
    seal = waves.seal_path(path)
    if seal.exists():
        raw = path.read_bytes()
        want = seal.read_text(encoding="utf-8").split()[0].strip().lower()
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise CriticEvalError(f"REFUSED: {path.name} ({got[:12]}) doesn't match its seal ({want[:12]})")
    try:
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except ValueError as e:
        raise CriticEvalError(f"REFUSED: cannot parse {path.name} ({type(e).__name__})") from e
    if not lines:
        raise CriticEvalError(f"REFUSED: {path.name} is empty")
    if lines[0].get("type") == "header" and lines[0].get("format") == waves.FORMAT:
        header, drafts = waves.read(path)
        return header, [extract_item_parts(d) for d in drafts]
    return None, [extract_item_parts(d) for d in lines]


def _load_eval_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise CriticEvalError(f"REFUSED: cannot read critic-eval file {path}")
    try:
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    except ValueError as e:
        raise CriticEvalError(f"REFUSED: cannot parse {path.name} ({type(e).__name__})") from e
    if not rows:
        raise CriticEvalError(f"REFUSED: {path.name} holds no critic-eval records")
    return rows


def _load_eval_report(eval_path: Path) -> dict:
    """Read critic-eval-report.json beside each --eval file."""
    report_path = eval_path.parent / "critic-eval-report.json"
    if not report_path.is_file():
        raise CriticEvalError(f"REFUSED: missing critic-eval-report.json beside {eval_path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise CriticEvalError(f"REFUSED: cannot parse {report_path.name} ({type(e).__name__})") from e
    if not isinstance(data, dict):
        raise CriticEvalError(f"REFUSED: {report_path.name} is not a JSON object")
    return data


def _report_gguf_sha(rep: dict) -> str | None:
    pins = rep.get("critic_pins") if isinstance(rep.get("critic_pins"), dict) else {}
    sha = rep.get("critic_gguf_sha256") or pins.get("critic_gguf_sha256") or pins.get("gguf_sha256")
    return str(sha).strip() if isinstance(sha, str) and sha.strip() else None


def _verdict_replies_from_eval(eval_rec: dict, num_turns: int) -> list[str]:
    """Extract each turn's raw_reply from eval_rec. Never reconstructs a reply from parsed fields."""
    iid = eval_rec.get("item_id") or eval_rec.get("id")
    if isinstance(eval_rec.get("verdict_checks"), list) and eval_rec["verdict_checks"]:
        checks = eval_rec["verdict_checks"]
    elif isinstance(eval_rec.get("verdict_check"), dict):
        checks = [eval_rec["verdict_check"]]
    else:
        raise CriticEvalError(f"REFUSED: eval record {iid!r} lacks verdict_checks / verdict_check")

    if len(checks) != num_turns:
        raise CriticEvalError(
            f"REFUSED: eval record {iid!r} has {len(checks)} verdict check(s), but key has {num_turns} turn(s)"
        )

    replies: list[str] = []
    for idx, vc in enumerate(checks, start=1):
        if not isinstance(vc, dict):
            raise CriticEvalError(f"REFUSED: eval record {iid!r} turn {idx} check is not a dict")
        raw = vc.get("raw_reply")
        if not isinstance(raw, str) or not raw.strip():
            raise CriticEvalError(
                f"REFUSED: eval record {iid!r} turn {idx} is missing raw_reply; offline re-scoring never reconstructs replies"
            )
        replies.append(raw)
    return replies


def _verdict_reply_from_eval(eval_rec: dict) -> str:
    """Single-turn raw_reply extractor; refuses when raw_reply is missing."""
    return _verdict_replies_from_eval(eval_rec, 1)[0]


def _flaw_reply_from_eval(eval_rec: dict) -> str:
    """Flaw raw_reply extractor; refuses when raw_reply is missing."""
    iid = eval_rec.get("item_id") or eval_rec.get("id")
    fc = eval_rec.get("flaw_check") if isinstance(eval_rec.get("flaw_check"), dict) else {}
    raw = fc.get("raw_reply")
    if not isinstance(raw, str) or not raw.strip():
        raise CriticEvalError(
            f"REFUSED: eval record {iid!r} is missing flaw_check.raw_reply; offline re-scoring never reconstructs replies"
        )
    return raw


def score_single_item(gates: Gates, eval_rec: dict, key_item: dict, rule_set: str) -> dict:
    """Score one recorded critic-eval record against `key_item` under P0 or P2 across all turns."""

    if rule_set not in ("P0", "P2"):
        raise CriticEvalError(f"REFUSED: unknown rule set {rule_set!r}")

    verdict_turns = key_item.get("verdict_turns") or [
        {
            "turn": 1,
            "seen_material": key_item.get("seen_material") or key_item.get("material") or "",
            "key_verdict": key_item["key_verdict"],
            "key_deciding": key_item["key_deciding"],
        }
    ]
    multi_turn = len(verdict_turns) > 1
    v_replies = _verdict_replies_from_eval(eval_rec, len(verdict_turns))

    def _rule_name(turn_num: int, code: str) -> str:
        return f"turn{turn_num}:{code}" if multi_turn else code

    last_v: CriticVerdict | None = None
    for vt, v_reply in zip(verdict_turns, v_replies, strict=True):
        t_num = int(vt.get("turn") or 1)
        key_verdict = vt["key_verdict"]
        key_deciding = vt["key_deciding"]
        material = vt["seen_material"]

        v = compare_verdict(gates, v_reply, key_verdict, key_deciding, material)
        last_v = v
        cd = list(v.critic_deciding if v.critic_deciding is not None else _lines(gates.parse_answer(v_reply).get("deciding")))

        if not v.ok:
            rname = _rule_name(t_num, v.code)
            return {
                "outcome": "refuse",
                "accepted": False,
                "refused": True,
                "turn": t_num,
                "deciding_rule": rname,
                "rule": v.code,
                "detail": v.detail,
            }

        if rule_set == "P2":
            if key_verdict == "cannot_tell" and len(cd) > 0:
                rname = _rule_name(t_num, "CANNOT_TELL_DECIDING")
                return {
                    "outcome": "refuse",
                    "accepted": False,
                    "refused": True,
                    "turn": t_num,
                    "deciding_rule": rname,
                    "rule": "CANNOT_TELL_DECIDING",
                    "detail": f"turn {t_num}: key is cannot_tell but critic quoted {len(cd)} deciding line(s)",
                }
            if key_verdict in ("supported", "unsupported") and len(cd) > len(key_deciding) + 1:
                rname = _rule_name(t_num, "QUOTE_TOO_LONG")
                return {
                    "outcome": "refuse",
                    "accepted": False,
                    "refused": True,
                    "turn": t_num,
                    "deciding_rule": rname,
                    "rule": "QUOTE_TOO_LONG",
                    "detail": f"turn {t_num}: critic quoted {len(cd)} lines, more than key's {len(key_deciding)} + 1",
                }

    if rule_set == "P2" and key_item["kind"] == "C" and key_item.get("key_flaw"):
        f_reply = _flaw_reply_from_eval(eval_rec)
        f_res = compare_flaw(f_reply, key_item["key_flaw"])
        if not f_res.ok:
            return {
                "outcome": "refuse",
                "accepted": False,
                "refused": True,
                "turn": len(verdict_turns),
                "deciding_rule": f_res.code,
                "rule": f_res.code,
                "detail": f_res.detail,
            }

    assert last_v is not None
    return {
        "outcome": "accept",
        "accepted": True,
        "refused": False,
        "turn": len(verdict_turns),
        "deciding_rule": "AGREE",
        "rule": "AGREE",
        "detail": last_v.detail,
    }


def _match_eval_record(key_item: dict, by_id: dict[str, dict]) -> dict:
    """Exact offline matcher:
    - A GPU item (d, b, e, f) matches only its own item_id.
    - A key-only item (a, c, g) matches only its control_item_id, which must be <src>:d.
    - Anything else refuses."""

    iid = key_item["item_id"]
    ptype = key_item["perturbation_type"]
    src = key_item["source_row_id"]

    if ptype in GPU_TYPES:
        if not iid or iid not in by_id:
            raise CriticEvalError(f"REFUSED: no recorded critic reply found for GPU item {iid!r}")
        return by_id[iid]

    if ptype in KEY_ONLY_TYPES:
        expected_ctrl = f"{src}:d"
        ctrl_id = key_item.get("control_item_id") or expected_ctrl
        if ctrl_id != expected_ctrl:
            raise CriticEvalError(
                f"REFUSED: key-only item {iid!r} has control_item_id {ctrl_id!r}, expected {expected_ctrl!r}"
            )
        if ctrl_id not in by_id:
            raise CriticEvalError(
                f"REFUSED: no recorded control critic reply {ctrl_id!r} found for key-only item {iid!r}"
            )
        rec = by_id[ctrl_id]
        rec_ptype = str(rec.get("perturbation_type") or rec.get("perturbation") or "d")
        if rec_ptype not in CONTROL_TYPES:
            raise CriticEvalError(
                f"REFUSED: control_item_id {ctrl_id!r} for key-only item {iid!r} has perturbation_type {rec_ptype!r}, not 'd'"
            )
        return rec

    raise CriticEvalError(f"REFUSED: unknown perturbation_type {ptype!r} for item {iid!r}")


def _index_eval(rows: list[dict]) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for r in rows:
        iid = str(r.get("item_id") or r.get("id") or "")
        if iid:
            by_id[iid] = r
    return by_id


def render_offline_table(summary: dict[str, dict[str, dict]]) -> str:
    lines = [
        "| rule_set | perturbation_type | role | n | accepted | refused | accept_rate | refuse_rate | primary_metric |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rs, types in summary.items():
        for ptype in sorted(types):
            s = types[ptype]
            is_ctrl = ptype in CONTROL_TYPES
            role = "control" if is_ctrl else "perturbed"
            primary = f"refuse_rate={s['refuse_rate']:.3f}" if is_ctrl else f"accept_rate={s['accept_rate']:.3f}"
            lines.append(
                f"| {rs} | {ptype} | {role} | {s['n']} | {s['accepted']} | {s['refused']} "
                f"| {s['accept_rate']:.3f} | {s['refuse_rate']:.3f} | {primary} |"
            )
    return "\n".join(lines)


def _merge_key_into_base(base: dict, ki: dict) -> dict:
    """When an alternate key record omits material, inherit material/new_material from its control item and rebuild turns."""
    mat = ki["material"] or base["material"]
    new_lines = ki["new_material"] if ki.get("new_material") else list(base.get("new_material") or [])
    tk = ki.get("turn_keys") if ki.get("turn_keys") is not None else copy.deepcopy(base.get("turn_keys"))
    if isinstance(tk, dict) and isinstance(tk.get("1"), dict):
        t1 = tk["1"]
        t1_verdict = str(t1.get("verdict") or "")
        t1_deciding = list(t1.get("deciding") if "deciding" in t1 else (t1.get("deciding_lines") or []))
        t2_seen = (mat.rstrip("\n") + "\n" + "\n".join(new_lines)) if new_lines else mat
        v_turns = [
            {"turn": 1, "seen_material": mat, "key_verdict": t1_verdict, "key_deciding": t1_deciding},
            {"turn": 2, "seen_material": t2_seen, "key_verdict": ki["key_verdict"], "key_deciding": ki["key_deciding"]},
        ]
        seen_mat = t2_seen
    else:
        seen_mat = (mat.rstrip("\n") + "\n" + "\n".join(new_lines)) if new_lines else (ki["seen_material"] or base["seen_material"] or mat)
        v_turns = [
            {"turn": 1, "seen_material": seen_mat, "key_verdict": ki["key_verdict"], "key_deciding": ki["key_deciding"]},
        ]
    return {
        **ki,
        "kind": ki["kind"] or base["kind"],
        "material": mat,
        "new_material": new_lines,
        "seen_material": seen_mat,
        "turn_keys": tk,
        "verdict_turns": v_turns,
    }


def run_critic_rescore_offline(cfg: CriticRescoreOfflineConfig) -> dict:
    """Re-score recorded critic-eval replies offline under P0, P2, and (when two runs are given) P3."""

    if not cfg.eval_files:
        raise CriticEvalError("REFUSED: critic-rescore-offline needs at least one --eval <critic-eval.jsonl>")
    if len(cfg.eval_files) > 2:
        raise CriticEvalError("REFUSED: critic-rescore-offline takes one --eval file (P0, P2) or two (P0, P2, P3)")
    if cfg.items is None:
        raise CriticEvalError("REFUSED: critic-rescore-offline needs --items <items.jsonl>")

    items_path = Path(cfg.items)
    items_header, base_items_list = _load_key_records(items_path)
    items_sha256 = waves.file_sha256(items_path)

    eval_paths = [Path(p) for p in cfg.eval_files]
    eval_runs = [_load_eval_records(p) for p in eval_paths]
    eval_reports = [_load_eval_report(p) for p in eval_paths]

    closed_form = (
        (items_header.get("gates") or {}).get("closed_form")
        if isinstance(items_header, dict) and isinstance(items_header.get("gates"), dict)
        else (eval_reports[0].get("gates") or {}).get("closed_form")
    )
    try:
        gates: Gates = load_gates(closed_form=closed_form)
    except GatesUnavailable as e:
        if not cfg.mock:
            raise CriticEvalError(str(e)) from e
        gates = mock_gates(closed_form=closed_form)

    for p, rep in zip(eval_paths, eval_reports, strict=True):
        rep_gates = rep.get("gates") if isinstance(rep.get("gates"), dict) else {}
        if (
            rep_gates.get("version") != gates.version
            or rep_gates.get("sha256") != gates.sha256
            or rep_gates.get("closed_form") != gates.closed_form
        ):
            raise CriticEvalError(
                f"REFUSED: {p.parent / 'critic-eval-report.json'} gates "
                f"{(rep_gates.get('version'), rep_gates.get('sha256'), rep_gates.get('closed_form'))} "
                f"do not match offline gates {(gates.version, gates.sha256, gates.closed_form)}"
            )
        if rep.get("input_sha256") != items_sha256:
            raise CriticEvalError(
                f"REFUSED: {p.parent / 'critic-eval-report.json'} input_sha256 ({rep.get('input_sha256')}) "
                f"does not match --items {items_path.name} ({items_sha256})"
            )

    if len(eval_runs) == 2:
        rep0, rep1 = eval_reports[0], eval_reports[1]
        model0, model1 = rep0.get("critic_model"), rep1.get("critic_model")
        if not model0 or model0 != model1:
            raise CriticEvalError(
                f"REFUSED: P3 requires the same critic_model in both reports ({model0!r} vs {model1!r})"
            )
        gguf0, gguf1 = _report_gguf_sha(rep0), _report_gguf_sha(rep1)
        if gguf0 != gguf1 or (not cfg.mock and (not gguf0 or not gguf1)):
            raise CriticEvalError(
                f"REFUSED: P3 requires the same critic gguf_sha256 in both reports ({gguf0!r} vs {gguf1!r})"
            )
        if rep0.get("input_sha256") != rep1.get("input_sha256"):
            raise CriticEvalError("REFUSED: P3 requires the same input_sha256 in both reports")

        seed0, seed1 = rep0.get("critic_seed"), rep1.get("critic_seed")
        if (
            seed0 is None
            or isinstance(seed0, bool)
            or not isinstance(seed0, int)
            or seed1 is None
            or isinstance(seed1, bool)
            or not isinstance(seed1, int)
        ):
            raise CriticEvalError("REFUSED: P3 requires a non-null integer critic_seed in both reports")
        for run_idx, (run_rows, expected_seed) in enumerate(((eval_runs[0], seed0), (eval_runs[1], seed1))):
            for r in run_rows:
                r_seed = r.get("seed")
                if r_seed is None or isinstance(r_seed, bool) or r_seed != expected_seed:
                    raise CriticEvalError(
                        f"REFUSED: P3 run {run_idx} record {r.get('item_id')!r} has missing or mismatched seed {r_seed!r}"
                    )
        if seed0 == seed1:
            raise CriticEvalError(
                f"REFUSED: P3 requires two critic-eval runs at different seeds; both runs have seed {seed0}"
            )

    key_items: list[dict] = []
    by_item_id: dict[str, int] = {}
    for ki in base_items_list:
        by_item_id[ki["item_id"]] = len(key_items)
        key_items.append(ki)
    for kp in cfg.keys:
        _, k_records = _load_key_records(Path(kp))
        for ki in k_records:
            if ki["item_id"] in by_item_id:
                idx = by_item_id[ki["item_id"]]
                base = dict(key_items[idx])
                if not ki["material"] and base.get("material"):
                    ki = _merge_key_into_base(base, ki)
                key_items[idx] = ki
            else:
                if not ki["material"]:
                    ctrl_id = ki.get("control_item_id") or f"{ki['source_row_id']}:d"
                    if ctrl_id in by_item_id:
                        ki = _merge_key_into_base(key_items[by_item_id[ctrl_id]], ki)
                by_item_id[ki["item_id"]] = len(key_items)
                key_items.append(ki)

    idx0 = _index_eval(eval_runs[0])
    idx1 = _index_eval(eval_runs[1]) if len(eval_runs) == 2 else {}

    rule_sets = ("P0", "P2", "P3") if len(eval_runs) == 2 else ("P0", "P2")
    rows_out: list[dict] = []
    by_item: dict[str, dict] = {}

    for ki in key_items:
        rec0 = _match_eval_record(ki, idx0)
        p0 = score_single_item(gates, rec0, ki, "P0")
        p2 = score_single_item(gates, rec0, ki, "P2")
        item_scores: dict[str, dict] = {"P0": p0, "P2": p2}

        if len(eval_runs) == 2:
            rec1 = _match_eval_record(ki, idx1)
            p2_run1 = score_single_item(gates, rec1, ki, "P2")
            if p2["refused"] or p2_run1["refused"]:
                if p2["refused"] and p2_run1["refused"]:
                    d_rule = p2["deciding_rule"] if p2["deciding_rule"] == p2_run1["deciding_rule"] else f"{p2['deciding_rule']}+{p2_run1['deciding_rule']}"
                    detail = f"run0 refused ({p2['deciding_rule']}); run1 refused ({p2_run1['deciding_rule']})"
                    turn_num = p2.get("turn", 1)
                elif p2["refused"]:
                    d_rule = p2["deciding_rule"]
                    detail = f"run0 refused ({p2['deciding_rule']}): {p2['detail']}"
                    turn_num = p2.get("turn", 1)
                else:
                    d_rule = p2_run1["deciding_rule"]
                    detail = f"run1 refused ({p2_run1['deciding_rule']}): {p2_run1['detail']}"
                    turn_num = p2_run1.get("turn", 1)
                p3 = {"outcome": "refuse", "accepted": False, "refused": True, "turn": turn_num, "deciding_rule": d_rule, "rule": d_rule, "detail": detail}
            else:
                p3 = {"outcome": "accept", "accepted": True, "refused": False, "turn": p2.get("turn", 1), "deciding_rule": "AGREE", "rule": "AGREE", "detail": "both seeds agreed under P2"}
            item_scores["P3"] = p3

        by_item[ki["item_id"]] = {
            "item_id": ki["item_id"],
            "source_row_id": ki["source_row_id"],
            "perturbation_type": ki["perturbation_type"],
            "kind": ki["kind"],
            **item_scores,
        }
        for rs in rule_sets:
            sc = item_scores[rs]
            rows_out.append({
                "item_id": ki["item_id"],
                "source_row_id": ki["source_row_id"],
                "perturbation_type": ki["perturbation_type"],
                "kind": ki["kind"],
                "rule_set": rs,
                **sc,
            })

    summary: dict[str, dict[str, dict]] = {}
    for rs in rule_sets:
        by_type: dict[str, list[dict]] = {}
        for r in rows_out:
            if r["rule_set"] == rs:
                by_type.setdefault(r["perturbation_type"], []).append(r)
        summary[rs] = {}
        for ptype, plist in sorted(by_type.items()):
            n = len(plist)
            acc = sum(1 for x in plist if x["accepted"])
            ref = sum(1 for x in plist if x["refused"])
            is_ctrl = ptype in CONTROL_TYPES
            summary[rs][ptype] = {
                "n": n,
                "role": "control" if is_ctrl else "perturbed",
                "accepted": acc,
                "refused": ref,
                "accept_rate": round(acc / n, 4) if n else 0.0,
                "refuse_rate": round(ref / n, 4) if n else 0.0,
            }

    table = render_offline_table(summary)
    if str(cfg.out).endswith(".jsonl"):
        jsonl_path = Path(cfg.out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = jsonl_path.with_name(jsonl_path.stem + "-report.json")
    else:
        cfg.out.mkdir(parents=True, exist_ok=True)
        jsonl_path = cfg.out / "critic-rescore-offline.jsonl"
        report_path = cfg.out / "critic-rescore-offline-report.json"

    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out), encoding="utf-8")
    report = {
        "rule_sets": list(rule_sets),
        "summary": summary,
        "by_item": by_item,
        "rows": rows_out,
        "table": table,
    }
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
