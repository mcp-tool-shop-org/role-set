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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from role_set import callcache, cards, proven, waves
from role_set.assemble import FLAWS, VERDICTS
from role_set.critic import (
    FLAW_PROMPT_SETTINGS,
    VERDICT_PROMPT_SETTINGS,
    CriticVerdict,
    _lines,
    compare_flaw,
    compare_verdict,
    flaw_messages,
    verdict_messages,
)
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
CRITIC_EVAL_CHECKS = ("all", "verdict")


class CriticEvalError(RuntimeError, GuardError):
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
    flaw_prompt: str = "old"
    verdict_prompt: str = field(default_factory=lambda: os.environ.get("PIN_CRITIC_VERDICT_PROMPT") or "old")
    checks: str = field(default_factory=lambda: os.environ.get("PIN_CRITIC_CHECKS") or "all")
    flaw_votes: int = 1
    verdict_from: Path | None = None
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

    raw_ck = (
        d.get("content_kind")
        or slot.get("content_kind")
        or row.get("content_kind")
        or (mat_field.get("content_kind") if isinstance(mat_field, dict) else None)
        or fields.get("content_kind")
    )
    content_kind = str(raw_ck).strip() if isinstance(raw_ck, str) and raw_ck.strip() else None

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
    elif isinstance(d.get("key"), str) and d["key"].strip():
        key_verdict = d["key"].strip()
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

    has_key_deciding = True
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
        has_key_deciding = False

    if "planted_flaw" in key_block:
        key_flaw = key_block["planted_flaw"]
    elif "flaw" in key_block:
        key_flaw = key_block["flaw"]
    elif "planted_flaw" in row:
        key_flaw = row["planted_flaw"]
    elif "planted_flaw" in fields:
        key_flaw = fields["planted_flaw"]
    elif "planted_flaw" in slot:
        key_flaw = slot["planted_flaw"]
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
        t1_has_dec = "deciding" in t1 or "deciding_lines" in t1
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
                "has_key_deciding": t1_has_dec,
            },
            {
                "turn": 2,
                "seen_material": t2_seen,
                "key_verdict": key_verdict,
                "key_deciding": key_deciding,
                "has_key_deciding": has_key_deciding,
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
                "has_key_deciding": has_key_deciding,
            }
        ]

    return {
        "item_id": item_id,
        "source_row_id": source_row_id,
        "control_item_id": str(d.get("control_item_id") or f"{source_row_id}:d"),
        "perturbation_type": perturbation_type,
        "kind": kind,
        "content_kind": content_kind,
        "material": material,
        "new_material": new_lines,
        "seen_material": seen_material,
        "statement": statement,
        "checker_trace": checker_trace,
        "checker_verdict": checker_verdict,
        "key_verdict": key_verdict,
        "key_deciding": key_deciding,
        "has_key_deciding": has_key_deciding,
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
    elif mock or model in ("", "mock") or not header:
        if model not in ("", "mock"):
            raise CriticEvalError(f"REFUSED: item {d.get('item_id') or d.get('id')!r} names writer {model!r}, which is not on the proven writer list")
        entry = {"served_as": model or ("mock" if mock else ""), "family": ""}
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

    wrapped.takes_check = getattr(critic, "takes_check", False)
    wrapped.card = getattr(critic, "card", None)
    return wrapped


def _load_eval_items(path: Path) -> tuple[dict, list[dict]]:
    """Read items for `critic-eval` from a sealed waves file (`waves.read`) or a bench JSONL file (`id`, `material`,
    `statement`, `key` per line, checking its `.sha256` seal when present)."""

    if not path.is_file():
        raise CriticEvalError(f"REFUSED: cannot read items file {path}")
    seal = waves.seal_path(path)
    raw = path.read_bytes()
    if seal.exists():
        try:
            want = seal.read_text(encoding="utf-8").split()[0].strip().lower()
        except (OSError, IndexError) as e:
            raise CriticEvalError(f"REFUSED: cannot read the drafts file and its seal {seal.name} ({type(e).__name__})") from e
        got = hashlib.sha256(raw).hexdigest()
        if got != want:
            raise CriticEvalError(
                f"REFUSED: {path.name} ({got[:12]}) doesn't match its seal ({want[:12]}); it changed in transit, so fetch it again"
            )
    try:
        lines = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    except ValueError as e:
        raise CriticEvalError(f"REFUSED: cannot parse {path.name} ({type(e).__name__})") from e
    if not lines:
        raise CriticEvalError(f"REFUSED: {path.name} is empty")
    if not all(isinstance(x, dict) for x in lines):
        raise CriticEvalError(f"REFUSED: {path.name} holds a line that is not a JSON object")
    if lines[0].get("type") == "header" or any(x.get("type") == "draft" for x in lines):
        try:
            return waves.read(path)
        except waves.WaveError as e:
            raise CriticEvalError(str(e)) from e
    return {}, lines


def run_critic_eval(cfg: CriticEvalConfig, *, critic=None) -> dict:
    """Run the critic's verdict check(s) and (for kind C) flaw check on every item of a sealed set."""

    if cfg.flaw_prompt not in FLAW_PROMPT_SETTINGS:
        raise CriticEvalError(
            f"REFUSED: unknown flaw_prompt {cfg.flaw_prompt!r}; expected one of {FLAW_PROMPT_SETTINGS}"
        )
    if cfg.verdict_prompt not in VERDICT_PROMPT_SETTINGS:
        raise CriticEvalError(
            f"REFUSED: unknown verdict_prompt {cfg.verdict_prompt!r}; expected one of {VERDICT_PROMPT_SETTINGS}"
        )
    if cfg.checks not in CRITIC_EVAL_CHECKS:
        raise CriticEvalError(
            f"REFUSED: unknown checks {cfg.checks!r}; expected one of {CRITIC_EVAL_CHECKS}"
        )
    if cfg.flaw_votes not in (1, 3):
        raise CriticEvalError(
            f"REFUSED: unknown flaw_votes {cfg.flaw_votes!r}; expected 1 or 3"
        )

    if cfg.pack is not None:
        try:
            verify(cfg.pack)
            load_pack(cfg.pack)
        except PackError as e:
            raise CriticEvalError(str(e)) from e

    header, drafts = _load_eval_items(cfg.items)
    input_sha256 = waves.file_sha256(cfg.items)

    verdict_from_sha256: str | None = None
    verdict_from_by_id: dict[str, dict] = {}
    if cfg.verdict_from is not None:
        vf_path = Path(cfg.verdict_from)
        vf_rows = _load_eval_records(vf_path)
        verdict_from_sha256 = waves.file_sha256(vf_path)
        vf_report_path = vf_path.parent / "critic-eval-report.json"
        has_verified_input_sha = False
        if vf_report_path.is_file():
            vf_rep = _load_eval_report(vf_path)
            if vf_rep.get("input_sha256") != input_sha256:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from {vf_path.name} report input_sha256 ({vf_rep.get('input_sha256')}) "
                    f"does not match --items {cfg.items.name} ({input_sha256})"
                )
            has_verified_input_sha = True
            vf_rep_seed = vf_rep.get("critic_seed")
            if vf_rep_seed is None or isinstance(vf_rep_seed, bool) or vf_rep_seed != cfg.critic_seed:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from {vf_path.name} report critic_seed ({vf_rep_seed!r}) "
                    f"does not match --critic-seed ({cfg.critic_seed})"
                )
            vf_rep_vp = vf_rep.get("verdict_prompt") if vf_rep.get("verdict_prompt") is not None else "old"
            if vf_rep_vp != cfg.verdict_prompt:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from {vf_path.name} report verdict_prompt ({vf_rep_vp!r}) "
                    f"does not match --verdict-prompt ({cfg.verdict_prompt!r})"
                )
            vf_rep_checks = vf_rep.get("checks") if vf_rep.get("checks") is not None else "all"
            if vf_rep_checks != cfg.checks:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from {vf_path.name} report checks ({vf_rep_checks!r}) "
                    f"does not match --checks ({cfg.checks!r})"
                )
        for r in vf_rows:
            if "input_sha256" in r:
                if r.get("input_sha256") != input_sha256:
                    raise CriticEvalError(
                        f"REFUSED: --verdict-from record {r.get('item_id')!r} input_sha256 ({r.get('input_sha256')}) "
                        f"does not match --items {cfg.items.name} ({input_sha256})"
                    )
                has_verified_input_sha = True
            r_seed = r.get("seed")
            if r_seed is None or isinstance(r_seed, bool) or r_seed != cfg.critic_seed:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from record {r.get('item_id')!r} has seed {r_seed!r}, "
                    f"expected {cfg.critic_seed}"
                )
            r_vp = r.get("verdict_prompt") if r.get("verdict_prompt") is not None else "old"
            if r_vp != cfg.verdict_prompt:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from record {r.get('item_id')!r} has verdict_prompt {r_vp!r}, "
                    f"expected {cfg.verdict_prompt!r}"
                )
            r_checks = r.get("checks") if r.get("checks") is not None else "all"
            if r_checks != cfg.checks:
                raise CriticEvalError(
                    f"REFUSED: --verdict-from record {r.get('item_id')!r} has checks {r_checks!r}, "
                    f"expected {cfg.checks!r}"
                )
        if not has_verified_input_sha:
            raise CriticEvalError(
                f"REFUSED: --verdict-from {vf_path.name} has no input_sha256 "
                f"(missing critic-eval-report.json beside it and no input_sha256 on records)"
            )
        verdict_from_by_id = _index_eval(vf_rows)

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
    c_card: dict | None = None
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
                    dec = list(vt["key_deciding"]) or (
                        [first_line] if vt["key_verdict"] != "cannot_tell" and not vt.get("has_key_deciding", True) and first_line else []
                    )
                    if vt["turn"] == 1 and (parts["statement"], first_line) not in answers:
                        answers[(parts["statement"], first_line)] = (vt["key_verdict"], dec)
                    elif vt["turn"] == 2:
                        if ("turn2", parts["statement"], first_line) not in answers:
                            answers[("turn2", parts["statement"], first_line)] = (vt["key_verdict"], dec)
                        if (parts["statement"], first_line) not in header_answers:
                            by_exact_mat.setdefault((parts["statement"], mat_t), (vt["key_verdict"], dec))
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

    if cfg.verdict_from is not None:
        for idx, d in enumerate(drafts):
            parts = extract_item_parts(d)
            item_id = parts["item_id"] or f"item-{idx}"
            w_pins, w_entry = _writer_for_item(d, header, listed, mock=cfg.mock)
            w_model = w_pins.get("writer_model") or w_entry.get("served_as") or ""
            w_family = (w_entry.get("family") or w_pins.get("writer_family") or "").strip() or None
            if w_family and c_family:
                try:
                    proven.require_other_family(
                        {**w_entry, "served_as": w_model or "writer", "family": w_family},
                        {**critic_entry, "served_as": c_model or "critic", "family": c_family},
                    )
                except GuardError:
                    continue
            if item_id not in verdict_from_by_id:
                raise CriticEvalError(
                    f"REFUSED: item {item_id!r} is missing from --verdict-from {Path(cfg.verdict_from).name}"
                )
            _verdict_replies_from_eval(verdict_from_by_id[item_id], len(parts["verdict_turns"]))

    cfg.out.mkdir(parents=True, exist_ok=True)
    journal_path = cfg.call_journal or (cfg.out / callcache.JOURNAL_FILE)
    try:
        reuse = (
            callcache.load(
                cfg.reuse_calls,
                expected_flaw_prompt=cfg.flaw_prompt,
                expected_verdict_prompt=cfg.verdict_prompt,
                expected_checks=cfg.checks,
            )
            if cfg.reuse_calls
            else None
        )
        journal = callcache.CallJournal(
            journal_path,
            reuse,
            flaw_prompt=cfg.flaw_prompt,
            verdict_prompt=cfg.verdict_prompt,
            checks=cfg.checks,
        )
    except GuardError as e:
        raise CriticEvalError(str(e)) from e
    journaled_critic = journal.wrap("critic", critic, card=c_card)

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
    verdict_reused_count = 0
    online_parts_by_id: dict[str, dict] = {}
    skipped_by_type: dict[str, list[str]] = {}
    has_trackb_ext = cfg.flaw_votes == 3 or cfg.verdict_from is not None

    for idx, d in enumerate(drafts):
        parts = extract_item_parts(d)
        item_id = parts["item_id"] or f"item-{idx}"
        w_pins, w_entry = _writer_for_item(d, header, listed, mock=cfg.mock)
        w_model = w_pins.get("writer_model") or w_entry.get("served_as") or ""
        w_family = (w_entry.get("family") or w_pins.get("writer_family") or "").strip() or None
        if not cfg.mock and not w_family and (header or w_model):
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
                skipped_by_type.setdefault(parts["perturbation_type"], []).append(item_id)
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

        if cfg.verdict_from is not None:
            vf_rec = verdict_from_by_id[item_id]
            vf_replies = _verdict_replies_from_eval(vf_rec, len(parts["verdict_turns"]))
            vf_orig_checks = (
                vf_rec["verdict_checks"]
                if isinstance(vf_rec.get("verdict_checks"), list) and vf_rec["verdict_checks"]
                else [vf_rec["verdict_check"]]
            )
            for vt, v_raw, orig_vc in zip(parts["verdict_turns"], vf_replies, vf_orig_checks, strict=True):
                v = compare_verdict(
                    gates,
                    v_raw,
                    vt["key_verdict"],
                    vt["key_deciding"],
                    vt["seen_material"],
                    check_evidence=vt.get("has_key_deciding", True),
                    content_kind=parts.get("content_kind"),
                )
                vc_entry = {
                    "turn": vt["turn"],
                    "code": v.code,
                    "ok": v.ok,
                    "detail": v.detail,
                    "raw_reply": v_raw,
                    "critic_verdict": v.critic_verdict,
                    "critic_deciding": list(v.critic_deciding or []),
                    "attempts": int(orig_vc.get("attempts", 1)),
                    "void_retries": int(orig_vc.get("void_retries", 0)),
                    "thinking_tokens": orig_vc.get("thinking_tokens"),
                    "output_tokens": int(orig_vc.get("output_tokens") or 0),
                    "prompt_tokens": int(orig_vc.get("prompt_tokens") or 0),
                }
                if v.repairs:
                    vc_entry["repairs"] = list(v.repairs)
                verdict_checks.append(vc_entry)
            verdict_reused_count += 1
        else:
            for vt in parts["verdict_turns"]:
                v_msgs = verdict_messages(
                    vt["seen_material"],
                    parts["statement"],
                    verdict_prompt=cfg.verdict_prompt,
                )
                v: CriticVerdict | None = None
                v_raw = ""
                v_attempts = 0
                v_voids = 0
                turn_thinkings: list[int | None] = []
                v_output = 0
                v_prompt = 0
                for _ask in range(2):
                    v_attempts += 1
                    ask_seed = callcache.retry_seed(cfg.critic_seed, _ask) if _ask else cfg.critic_seed
                    c = blinded_critic(v_msgs, seed=ask_seed, slot=item_id, attempt=_ask)
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
                    v = compare_verdict(
                        gates,
                        c.text,
                        vt["key_verdict"],
                        vt["key_deciding"],
                        vt["seen_material"],
                        check_evidence=vt.get("has_key_deciding", True),
                        content_kind=parts.get("content_kind"),
                    )
                    if v.code != "VOID":
                        break
                    v_voids += 1

                assert v is not None
                v_output_total += v_output
                v_prompt_total += v_prompt
                vc_entry = {
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
                }
                if v.repairs:
                    vc_entry["repairs"] = list(v.repairs)
                verdict_checks.append(vc_entry)

        verdict_check = verdict_checks[-1]

        flaw_check: dict | None = None
        f_thinking: int | None = None
        f_output = 0
        f_prompt = 0
        critic_flaw: str | None = None
        if parts["kind"] == "C" and cfg.checks != "verdict":
            if not parts["key_flaw"] or parts["key_flaw"] not in FLAWS + ("none",):
                raise CriticEvalError(f"REFUSED: kind-C item {item_id} lacks a valid planted_flaw key ({parts['key_flaw']!r})")
            f_msgs = flaw_messages(
                parts["material"],
                parts["statement"],
                parts["checker_trace"],
                parts["checker_verdict"],
                flaw_prompt=cfg.flaw_prompt,
            )
            if cfg.flaw_votes == 1:
                fc = blinded_critic(f_msgs, seed=cfg.critic_seed, slot=item_id, attempt=0)
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
                    "flaw_prompt": cfg.flaw_prompt,
                    "thinking_tokens": f_thinking,
                    "output_tokens": f_output,
                    "prompt_tokens": f_prompt,
                }
            else:
                votes: list[dict] = []
                vote_thinkings: list[int | None] = []
                for vote_idx in range(cfg.flaw_votes):
                    vote_seed = cfg.critic_seed + vote_idx
                    fc_k = blinded_critic(f_msgs, seed=vote_seed, slot=item_id, attempt=vote_idx)
                    f_res_k = compare_flaw(fc_k.text, parts["key_flaw"])
                    raw_ft_k = getattr(fc_k, "thinking_tokens", None)
                    f_thinking_k = int(raw_ft_k) if raw_ft_k is not None else None
                    vote_thinkings.append(f_thinking_k)
                    item_call_thinkings.append(f_thinking_k)
                    all_call_thinkings.append(f_thinking_k)
                    if f_thinking_k is not None:
                        thinking_measured_calls += 1
                    else:
                        thinking_unmeasured_calls += 1
                    f_out_k = int(fc_k.completion_tokens or 0)
                    f_prm_k = int(fc_k.prompt_tokens or 0)
                    f_output += f_out_k
                    f_prompt += f_prm_k
                    c_flaw_k = _flaw_word(fc_k.text)
                    votes.append({
                        "vote": vote_idx,
                        "seed": vote_seed,
                        "code": f_res_k.code,
                        "ok": f_res_k.ok,
                        "detail": f_res_k.detail,
                        "raw_reply": fc_k.text,
                        "critic_flaw": c_flaw_k,
                        "thinking_tokens": f_thinking_k,
                        "output_tokens": f_out_k,
                        "prompt_tokens": f_prm_k,
                    })
                f_thinking = _sum_thinking(vote_thinkings)
                vote_words = [v["critic_flaw"] for v in votes]
                none_votes = sum(1 for w in vote_words if w == "none")
                flaw_label_votes = sum(1 for w in vote_words if w is not None and w != "none")
                void_votes = sum(1 for w in vote_words if w is None)
                non_none_votes = sum(1 for w in vote_words if w != "none")
                flaw_refuses = none_votes < 2
                key_has_flaw = parts["key_flaw"] != "none"
                pc_ok = (flaw_label_votes >= 2) if key_has_flaw else (none_votes >= 2)
                disagree_votes = none_votes if key_has_flaw else flaw_label_votes
                pc_code = "AGREE" if pc_ok else ("FLAW_MISMATCH" if disagree_votes >= 2 else "VOID")
                pc_detail = (
                    f"votes {[w or 'VOID' for w in vote_words]} "
                    f"(flaw={flaw_label_votes}/3, none={none_votes}/3, void={void_votes}/3) vs key {parts['key_flaw']}"
                )
                match_votes = sum(1 for w in vote_words if w == parts["key_flaw"])
                p2_ok = match_votes >= 2
                p2_code = "AGREE" if p2_ok else ("FLAW_MISMATCH" if (flaw_label_votes + none_votes - match_votes) >= 2 else "VOID")
                v0_word = votes[0]["critic_flaw"]
                v0_pc_ok = v0_word is not None and ((v0_word == "none") == (parts["key_flaw"] == "none"))
                v0_pc_code = "VOID" if v0_word is None else ("AGREE" if v0_pc_ok else "FLAW_MISMATCH")
                # Pick majority label if >=2 votes agree on a label; else "none" if not flaw_refuses, else first non-none label
                maj_word = next((w for w in vote_words if w is not None and vote_words.count(w) >= 2), None)
                if maj_word is not None:
                    critic_flaw = maj_word
                elif not flaw_refuses:
                    critic_flaw = "none"
                else:
                    critic_flaw = next((w for w in vote_words if w is not None and w != "none"), None)
                flaw_check = {
                    "code": pc_code,
                    "ok": pc_ok,
                    "detail": pc_detail,
                    "p2_code": p2_code,
                    "p2_ok": p2_ok,
                    "raw_reply": votes[0]["raw_reply"],
                    "critic_flaw": critic_flaw,
                    "flaw_prompt": cfg.flaw_prompt,
                    "flaw_votes": cfg.flaw_votes,
                    "non_none_votes": non_none_votes,
                    "flaw_refuses": flaw_refuses,
                    "votes": votes,
                    "one_vote": {
                        "code": v0_pc_code,
                        "ok": v0_pc_ok,
                        "p2_code": votes[0]["code"],
                        "p2_ok": votes[0]["ok"],
                        "raw_reply": votes[0]["raw_reply"],
                        "critic_flaw": v0_word,
                    },
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
            "verdict_prompt": cfg.verdict_prompt,
            "flaw_prompt": cfg.flaw_prompt,
            "checks": cfg.checks,
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
        if has_trackb_ext:
            rec["content_kind"] = parts.get("content_kind")
            rec["input_sha256"] = input_sha256
            rec["flaw_votes"] = cfg.flaw_votes
            if verdict_from_sha256 is not None:
                rec["verdict_from_sha256"] = verdict_from_sha256
                rec["verdict_turns_reused"] = len(verdict_checks)
            online_parts_by_id[item_id] = {**parts, "item_id": item_id}
            rec["scores"] = {
                "P0": score_single_item(gates, rec, online_parts_by_id[item_id], "P0"),
                "Pc": score_single_item(gates, rec, online_parts_by_id[item_id], "Pc"),
                "Pc_1vote": score_single_item(gates, rec, online_parts_by_id[item_id], "Pc_1vote"),
                "P2": score_single_item(gates, rec, online_parts_by_id[item_id], "P2"),
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
        "verdict_prompt": cfg.verdict_prompt,
        "flaw_prompt": cfg.flaw_prompt,
        "checks": cfg.checks,
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
    if has_trackb_ext:
        report["flaw_votes"] = cfg.flaw_votes
        if verdict_from_sha256 is not None:
            report["verdict_from_sha256"] = verdict_from_sha256
            report["verdict_reused_count"] = verdict_reused_count
        online_rule_sets = ("P0", "Pc", "Pc_1vote", "P2")
        online_rows: list[dict] = []
        online_by_item: dict[str, dict] = {}
        for rec in records:
            iid = rec["item_id"]
            sc_map = rec["scores"]
            online_by_item[iid] = {
                "item_id": iid,
                "source_row_id": rec["source_row_id"],
                "perturbation_type": rec["perturbation_type"],
                "kind": rec["kind"],
                **sc_map,
            }
            for rs_name in online_rule_sets:
                online_rows.append({
                    "item_id": iid,
                    "source_row_id": rec["source_row_id"],
                    "perturbation_type": rec["perturbation_type"],
                    "kind": rec["kind"],
                    "rule_set": rs_name,
                    **sc_map[rs_name],
                })
        online_summary = _summarize_scores(online_rows, online_rule_sets, skipped_by_type)
        report["summary"] = online_summary
        report["by_item"] = online_by_item
        report["one_vote"] = {"Pc": online_summary["Pc_1vote"]}
    (cfg.out / "critic-eval-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


# ---- offline re-scorer (P0, Pc, P2, P3) ---------------------------------------------------------------------

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


def _skipped_ids_from_report(rep: dict) -> list[str]:
    """Extract the skipped item_ids from a critic-eval-report.json object."""
    raw = rep.get("skipped")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise CriticEvalError("REFUSED: critic-eval-report.json 'skipped' must be a list")
    ids: list[str] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            iid = entry.strip()
        elif isinstance(entry, dict):
            iid = str(entry.get("item_id") or entry.get("id") or "").strip()
            if not iid:
                raise CriticEvalError("REFUSED: critic-eval-report.json 'skipped' entry lacks item_id")
        else:
            raise CriticEvalError("REFUSED: invalid entry in critic-eval-report.json 'skipped'")
        if iid not in ids:
            ids.append(iid)
    return ids


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


def _flaw_votes_from_eval(eval_rec: dict) -> list[str]:
    """Extract flaw vote raw replies from eval_rec (1 or 3 votes). Never reconstructs a reply from parsed fields."""
    iid = eval_rec.get("item_id") or eval_rec.get("id")
    fc = eval_rec.get("flaw_check") if isinstance(eval_rec.get("flaw_check"), dict) else {}
    votes = fc.get("votes")
    if votes is not None:
        if not isinstance(votes, list) or len(votes) not in (1, 3):
            raise CriticEvalError(
                f"REFUSED: eval record {iid!r} has invalid flaw_check.votes (expected 1 or 3 entries)"
            )
        replies: list[str] = []
        for idx, v in enumerate(votes):
            if not isinstance(v, dict):
                raise CriticEvalError(f"REFUSED: eval record {iid!r} flaw vote {idx} is not a dict")
            raw = v.get("raw_reply")
            if not isinstance(raw, str) or not raw.strip():
                raise CriticEvalError(
                    f"REFUSED: eval record {iid!r} flaw vote {idx} is missing raw_reply; offline re-scoring never reconstructs replies"
                )
            replies.append(raw)
        return replies
    return [_flaw_reply_from_eval(eval_rec)]


def score_single_item(gates: Gates, eval_rec: dict, key_item: dict, rule_set: str) -> dict:
    """Score one recorded critic-eval record against `key_item` under P0, Pc, Pc_1vote, or P2 across all turns."""

    if rule_set not in ("P0", "Pc", "Pc_1vote", "P2", "P2_1vote"):
        raise CriticEvalError(f"REFUSED: unknown rule set {rule_set!r}")

    verdict_turns = key_item.get("verdict_turns") or [
        {
            "turn": 1,
            "seen_material": key_item.get("seen_material") or key_item.get("material") or "",
            "key_verdict": key_item["key_verdict"],
            "key_deciding": key_item["key_deciding"],
            "has_key_deciding": key_item.get("has_key_deciding", True),
        }
    ]
    multi_turn = len(verdict_turns) > 1
    v_replies = _verdict_replies_from_eval(eval_rec, len(verdict_turns))

    def _rule_name(turn_num: int, code: str) -> str:
        return f"turn{turn_num}:{code}" if multi_turn else code

    last_v: CriticVerdict | None = None
    content_kind = key_item.get("content_kind") or eval_rec.get("content_kind")
    for vt, v_reply in zip(verdict_turns, v_replies, strict=True):
        t_num = int(vt.get("turn") or 1)
        key_verdict = vt["key_verdict"]
        key_deciding = vt["key_deciding"]
        has_key_deciding = vt.get("has_key_deciding", True)
        material = vt["seen_material"]

        v = compare_verdict(
            gates,
            v_reply,
            key_verdict,
            key_deciding,
            material,
            check_evidence=has_key_deciding,
            content_kind=content_kind,
        )
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

        if rule_set in ("P2", "P2_1vote"):
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
            if has_key_deciding and key_verdict in ("supported", "unsupported") and len(cd) > len(key_deciding) + 1:
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

    if (
        rule_set in ("Pc", "Pc_1vote", "P2", "P2_1vote")
        and key_item["kind"] == "C"
        and eval_rec.get("checks", "all") != "verdict"
    ):
        key_flaw = key_item.get("key_flaw")
        if not key_flaw or key_flaw not in FLAWS + ("none",):
            raise CriticEvalError(
                f"REFUSED: kind-C item {key_item.get('item_id')!r} lacks a valid key_flaw ({key_flaw!r}) for {rule_set}"
            )
        vote_replies = _flaw_votes_from_eval(eval_rec)
        if rule_set in ("Pc_1vote", "P2_1vote") or len(vote_replies) == 1:
            f_reply = vote_replies[0]
            f_res = compare_flaw(f_reply, key_flaw)
            if rule_set in ("Pc", "Pc_1vote"):
                if f_res.code == "VOID":
                    return {
                        "outcome": "refuse",
                        "accepted": False,
                        "refused": True,
                        "turn": len(verdict_turns),
                        "deciding_rule": "VOID",
                        "rule": "VOID",
                        "detail": f_res.detail,
                    }
                critic_flaw = _flaw_word(f_reply)
                if (critic_flaw == "none") != (key_flaw == "none"):
                    return {
                        "outcome": "refuse",
                        "accepted": False,
                        "refused": True,
                        "turn": len(verdict_turns),
                        "deciding_rule": "FLAW_MISMATCH",
                        "rule": "FLAW_MISMATCH",
                        "detail": f"critic flaw {critic_flaw} != key flaw {key_flaw} (coarse)",
                    }
            elif not f_res.ok:
                return {
                    "outcome": "refuse",
                    "accepted": False,
                    "refused": True,
                    "turn": len(verdict_turns),
                    "deciding_rule": f_res.code,
                    "rule": f_res.code,
                    "detail": f_res.detail,
                }
        else:
            words = [_flaw_word(r) for r in vote_replies]
            none_votes = sum(1 for w in words if w == "none")
            flaw_label_votes = sum(1 for w in words if w is not None and w != "none")
            void_votes = sum(1 for w in words if w is None)
            if rule_set == "Pc":
                key_has_flaw = key_flaw != "none"
                pc_ok = (flaw_label_votes >= 2) if key_has_flaw else (none_votes >= 2)
                if not pc_ok:
                    disagree_votes = none_votes if key_has_flaw else flaw_label_votes
                    code = "FLAW_MISMATCH" if disagree_votes >= 2 else "VOID"
                    return {
                        "outcome": "refuse",
                        "accepted": False,
                        "refused": True,
                        "turn": len(verdict_turns),
                        "deciding_rule": code,
                        "rule": code,
                        "detail": (
                            f"votes {[w or 'VOID' for w in words]} "
                            f"(flaw={flaw_label_votes}/3, none={none_votes}/3, void={void_votes}/3) != key flaw {key_flaw} (coarse)"
                        ),
                    }
            else:
                match_votes = sum(1 for w in words if w == key_flaw)
                if match_votes < 2:
                    code = "FLAW_MISMATCH" if (flaw_label_votes + none_votes - match_votes) >= 2 else "VOID"
                    return {
                        "outcome": "refuse",
                        "accepted": False,
                        "refused": True,
                        "turn": len(verdict_turns),
                        "deciding_rule": code,
                        "rule": code,
                        "detail": f"votes {[w or 'VOID' for w in words]} ({match_votes}/3 match {key_flaw})",
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


def _summarize_scores(
    rows_out: list[dict],
    rule_sets: tuple[str, ...],
    skipped_by_type: dict[str, list[str]],
) -> dict[str, dict[str, dict]]:
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
            sk_type = skipped_by_type.get(ptype, [])
            summary[rs][ptype] = {
                "n": n,
                "role": "control" if is_ctrl else "perturbed",
                "accepted": acc,
                "refused": ref,
                "accept_rate": round(acc / n, 4) if n else 0.0,
                "refuse_rate": round(ref / n, 4) if n else 0.0,
                "skipped_count": len(sk_type),
                "skipped": list(sk_type),
            }
    return summary


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


def render_offline_table(summary: dict[str, dict[str, dict]], *, skipped: list[str] | None = None) -> str:
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
    if skipped is not None:
        ids_str = ", ".join(skipped) if skipped else "none"
        lines.append(f"\nSkipped ({len(skipped)}): {ids_str}")
    return "\n".join(lines)


def _merge_key_into_base(base: dict, ki: dict) -> dict:
    """When an alternate key record omits material, inherit material/new_material from its control item and rebuild turns."""
    mat = ki["material"] or base["material"]
    new_lines = ki["new_material"] if ki.get("new_material") else list(base.get("new_material") or [])
    tk = ki.get("turn_keys") if ki.get("turn_keys") is not None else copy.deepcopy(base.get("turn_keys"))
    if isinstance(tk, dict) and isinstance(tk.get("1"), dict):
        t1 = tk["1"]
        t1_verdict = str(t1.get("verdict") or "")
        t1_has_dec = "deciding" in t1 or "deciding_lines" in t1
        t1_deciding = list(t1.get("deciding") if "deciding" in t1 else (t1.get("deciding_lines") or []))
        t2_seen = (mat.rstrip("\n") + "\n" + "\n".join(new_lines)) if new_lines else mat
        v_turns = [
            {
                "turn": 1,
                "seen_material": mat,
                "key_verdict": t1_verdict,
                "key_deciding": t1_deciding,
                "has_key_deciding": t1_has_dec,
            },
            {
                "turn": 2,
                "seen_material": t2_seen,
                "key_verdict": ki["key_verdict"],
                "key_deciding": ki["key_deciding"],
                "has_key_deciding": ki.get("has_key_deciding", True),
            },
        ]
        seen_mat = t2_seen
    else:
        seen_mat = (mat.rstrip("\n") + "\n" + "\n".join(new_lines)) if new_lines else (ki["seen_material"] or base["seen_material"] or mat)
        v_turns = [
            {
                "turn": 1,
                "seen_material": seen_mat,
                "key_verdict": ki["key_verdict"],
                "key_deciding": ki["key_deciding"],
                "has_key_deciding": ki.get("has_key_deciding", True),
            },
        ]
    return {
        **ki,
        "kind": ki["kind"] or base["kind"],
        "content_kind": ki.get("content_kind") or base.get("content_kind"),
        "material": mat,
        "new_material": new_lines,
        "seen_material": seen_mat,
        "turn_keys": tk,
        "verdict_turns": v_turns,
    }


def run_critic_rescore_offline(cfg: CriticRescoreOfflineConfig) -> dict:
    """Re-score recorded critic-eval replies offline under P0, Pc, P2, and (when two runs are given) P3."""

    if not cfg.eval_files:
        raise CriticEvalError("REFUSED: critic-rescore-offline needs at least one --eval <critic-eval.jsonl>")
    if len(cfg.eval_files) > 2:
        raise CriticEvalError("REFUSED: critic-rescore-offline takes one --eval file (P0, Pc, P2) or two (P0, Pc, P2, P3)")
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

    skipped_0 = _skipped_ids_from_report(eval_reports[0])
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
        fp0 = str(rep0.get("flaw_prompt") or "old")
        fp1 = str(rep1.get("flaw_prompt") or "old")
        if fp0 != fp1:
            raise CriticEvalError(
                f"REFUSED: P3 requires the same flaw_prompt in both reports ({fp0!r} vs {fp1!r})"
            )
        vp0 = str(rep0.get("verdict_prompt") or "old")
        vp1 = str(rep1.get("verdict_prompt") or "old")
        if vp0 != vp1:
            raise CriticEvalError(
                f"REFUSED: P3 requires the same verdict_prompt in both reports ({vp0!r} vs {vp1!r})"
            )
        ck0 = str(rep0.get("checks") or "all")
        ck1 = str(rep1.get("checks") or "all")
        if ck0 != ck1:
            raise CriticEvalError(
                f"REFUSED: P3 requires the same checks setting in both reports ({ck0!r} vs {ck1!r})"
            )

        skipped_1 = _skipped_ids_from_report(rep1)
        if set(skipped_0) != set(skipped_1):
            raise CriticEvalError(
                f"REFUSED: P3 requires the same skipped item ids in both reports ({skipped_0} vs {skipped_1})"
            )

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

    has_3votes = any(
        int(rep.get("flaw_votes") or 1) == 3
        for rep in eval_reports
    ) or any(
        isinstance((r.get("flaw_check") or {}).get("votes"), list) and len(r["flaw_check"]["votes"]) == 3
        for run_rows in eval_runs
        for r in run_rows
    )
    if has_3votes:
        rule_sets = ("P0", "Pc", "Pc_1vote", "P2", "P3") if len(eval_runs) == 2 else ("P0", "Pc", "Pc_1vote", "P2")
    else:
        rule_sets = ("P0", "Pc", "P2", "P3") if len(eval_runs) == 2 else ("P0", "Pc", "P2")

    rows_out: list[dict] = []
    by_item: dict[str, dict] = {}
    eval_skipped_set = set(skipped_0)
    left_out: list[str] = []
    skipped_by_type: dict[str, list[str]] = {}

    for ki in key_items:
        iid = ki["item_id"]
        ptype = ki["perturbation_type"]
        src = ki["source_row_id"]
        if ptype in KEY_ONLY_TYPES:
            expected_ctrl = f"{src}:d"
            ctrl_id = ki.get("control_item_id") or expected_ctrl
            if ctrl_id != expected_ctrl:
                raise CriticEvalError(
                    f"REFUSED: key-only item {iid!r} has control_item_id {ctrl_id!r}, expected {expected_ctrl!r}"
                )
            if iid in eval_skipped_set or ctrl_id in eval_skipped_set:
                left_out.append(iid)
                skipped_by_type.setdefault(ptype, []).append(iid)
                continue
        elif ptype in GPU_TYPES:
            if iid in eval_skipped_set:
                left_out.append(iid)
                skipped_by_type.setdefault(ptype, []).append(iid)
                continue

        rec0 = _match_eval_record(ki, idx0)
        p0 = score_single_item(gates, rec0, ki, "P0")
        pc = score_single_item(gates, rec0, ki, "Pc")
        p2 = score_single_item(gates, rec0, ki, "P2")
        item_scores: dict[str, dict] = {"P0": p0, "Pc": pc, "P2": p2}
        if has_3votes:
            item_scores["Pc_1vote"] = score_single_item(gates, rec0, ki, "Pc_1vote")

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

    summary = _summarize_scores(rows_out, rule_sets, skipped_by_type)

    table = render_offline_table(summary, skipped=left_out)
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
        "skipped_count": len(left_out),
        "skipped": left_out,
        "summary": summary,
        "by_item": by_item,
        "rows": rows_out,
        "table": table,
    }
    if has_3votes:
        report["one_vote"] = {"Pc": summary["Pc_1vote"]}
    report_path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
