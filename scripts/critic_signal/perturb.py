"""`scripts/critic_signal/perturb.py`: deterministic perturbation generator for the critic-signal study.

Takes a sealed file of gold-confirmed rows plus `--seed`, and writes:
- `<out>/items.jsonl` (+ `.sha256` seal): GPU-scored items of types (d) control, (b) material flip,
  (e) cannot_tell trap, and (f) clean trace (kind C only).
- `<out>/keys-a.jsonl`, `<out>/keys-c.jsonl`, `<out>/keys-g.jsonl` (+ `.sha256` seals): offline key-only
  perturbations over the controls for (a) verdict flip, (c) flaw relabel, and (g) decoy deciding.
- `<out>/manifest.json`: records the seed, input/output hashes, every item's source row, perturbation type,
  exact changes, and any skipped rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from role_set import waves  # noqa: E402
from role_set.assemble import FLAWS, VERDICTS, _ws  # noqa: E402
from role_set.critic import FLAW_DEFINITIONS  # noqa: E402
from role_set.critic_eval import extract_item_parts  # noqa: E402
from role_set.gates import GatesUnavailable, load_gates  # noqa: E402
from role_set.mock import mock_gates  # noqa: E402

NUMBER_RE = re.compile(r"(?<![\w.])(?<!\d\.)(\d+:\d+|\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?!\w|\.\d)")
VERSION_PREFIX_RE = re.compile(r"\b(?:version|release|v)\s*$", re.I)
THE_SUBJECT_RE = re.compile(r"^(\s*The\s+[A-Za-z][A-Za-z0-9_-]*(?:\s+[A-Za-z][A-Za-z0-9_-]*)?)\b(.*)$")
FORBIDDEN_QUALIFIERS = ("if approved", "allegedly", "reportedly")
F_TRACE_FRAME_WORDS = frozenset({
    "the", "material", "says", "that", "matches", "contradicts", "statement", "so", "it", "is", "supported", "unsupported",
})


class PerturbError(RuntimeError):
    """Input is unsealed, malformed, or lacks a valid key."""


def _find_numbers(text: str) -> list[re.Match[str]]:
    """Find non-version numbers in `text` (never matching inside version strings like 1.2.4 or after 'version ')."""
    return [m for m in NUMBER_RE.finditer(text) if not VERSION_PREFIX_RE.search(text[:m.start()])]


def _verify_seal_and_load(path: Path) -> tuple[dict, list[dict], str]:
    if not path.is_file():
        raise PerturbError(f"REFUSED: cannot read input file {path}")
    seal = waves.seal_path(path)
    if not seal.is_file():
        raise PerturbError(f"REFUSED: input file {path.name} is not sealed (missing {seal.name})")
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    try:
        want = seal.read_text(encoding="utf-8").split()[0].strip().lower()
    except (OSError, IndexError) as e:
        raise PerturbError(f"REFUSED: cannot read seal {seal.name} ({type(e).__name__})") from e
    if got != want:
        raise PerturbError(f"REFUSED: {path.name} ({got[:12]}) does not match its seal ({want[:12]})")

    try:
        lines = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
    except ValueError as e:
        raise PerturbError(f"REFUSED: cannot parse {path.name} ({type(e).__name__})") from e
    if not lines:
        raise PerturbError(f"REFUSED: {path.name} is empty")

    if lines[0].get("type") == "header" and lines[0].get("format") == waves.FORMAT:
        header, drafts = waves.read(path)
        return header, drafts, got

    try:
        g = load_gates()
    except GatesUnavailable:
        g = mock_gates()
    header = {
        "type": "header",
        "format": waves.FORMAT,
        "gates": {"version": g.version, "sha256": g.sha256, "closed_form": g.closed_form},
        "batch_pins": {},
        "writer_pins": {},
        "waves": [],
    }
    drafts = []
    for idx, item in enumerate(lines):
        if item.get("type") == "draft":
            drafts.append(item)
        else:
            rid = str(item.get("id") or f"row-{idx}")
            kind = str(item.get("kind") or "D")
            drafts.append({
                "type": "draft",
                "slot": {"id": rid, "lesson": str(item.get("lesson") or "T02"), "kind": kind, "side": str(item.get("side") or "x"), "tier": int(item.get("tier") or 1)},
                "state": "accepted",
                "attempt": 1,
                "ended": False,
                "messages": [],
                "material": {"text": str(item.get("material") or ""), "source": (item.get("sources") or [{"invented": "source"}])[0]},
                "facts": None,
                "spent": [],
                "refusals": [],
                "candidate": None,
                "row": item,
                "model": None,
                "lookup": False,
                "receipt": None,
                "accepted_seq": idx,
            })
    return header, drafts, got


def _validate_source_item(d: dict, idx: int) -> dict:
    parts = extract_item_parts(d)
    src_id = parts["source_row_id"] or parts["item_id"] or f"item-{idx}"
    if not parts["material"].strip():
        raise PerturbError(f"REFUSED: input row {src_id!r} lacks material")
    if not parts["statement"].strip():
        raise PerturbError(f"REFUSED: input row {src_id!r} lacks statement")
    for vt in parts["verdict_turns"]:
        t_num = vt["turn"]
        kv = vt["key_verdict"]
        kd = vt["key_deciding"]
        if kv not in VERDICTS:
            raise PerturbError(f"REFUSED: input row {src_id!r} turn {t_num} lacks a valid key verdict ({kv!r})")
        if kv in ("supported", "unsupported") and not kd:
            raise PerturbError(f"REFUSED: input row {src_id!r} turn {t_num} has verdict {kv!r} but no deciding lines in its key")
        if kv == "cannot_tell" and kd:
            raise PerturbError(f"REFUSED: input row {src_id!r} turn {t_num} has verdict 'cannot_tell' but non-empty deciding lines in its key")
    if parts["kind"] == "C":
        if not parts["key_flaw"] or parts["key_flaw"] not in FLAWS + ("none",):
            raise PerturbError(f"REFUSED: kind-C input row {src_id!r} lacks a valid planted_flaw key ({parts['key_flaw']!r})")
        if not parts["checker_trace"].strip() or parts["checker_verdict"] not in VERDICTS:
            raise PerturbError(f"REFUSED: kind-C input row {src_id!r} lacks checker_trace or checker_verdict")
    return {**parts, "source_row_id": src_id}


def _build_draft_record(
    base_d: dict,
    parts: dict,
    ptype: str,
    *,
    material: str | None = None,
    new_material: list[str] | None = None,
    checker_trace: str | None = None,
    checker_verdict: str | None = None,
    key_verdict: str | None = None,
    key_deciding: list[str] | None = None,
    key_flaw: str | None = None,
    turn_keys: dict | None = None,
) -> dict:
    out = copy.deepcopy(base_d)
    src_id = parts["source_row_id"]
    item_id = f"{src_id}:{ptype}"

    mat = parts["material"] if material is None else material
    new_mat = list(parts["new_material"] if new_material is None else new_material)
    c_trace = parts["checker_trace"] if checker_trace is None else checker_trace
    c_verdict = parts["checker_verdict"] if checker_verdict is None else checker_verdict
    k_verdict = parts["key_verdict"] if key_verdict is None else key_verdict
    k_deciding = list(parts["key_deciding"] if key_deciding is None else key_deciding)
    k_flaw = parts["key_flaw"] if key_flaw is None else key_flaw
    t_keys = copy.deepcopy(parts["turn_keys"] if turn_keys is None else turn_keys)

    out["type"] = "draft"
    out["item_id"] = item_id
    out["id"] = item_id
    out["source_row_id"] = src_id
    out["control_item_id"] = f"{src_id}:d"
    out["perturbation_type"] = ptype
    out["kind"] = parts["kind"]
    out["statement"] = parts["statement"]
    out["checker_trace"] = c_trace if parts["kind"] == "C" else None
    out["checker_verdict"] = c_verdict if parts["kind"] == "C" else None
    if new_material is not None or parts.get("new_material") or t_keys is not None:
        out["new_material"] = new_mat

    key_block: dict = {"verdict": k_verdict, "deciding": k_deciding}
    if parts["kind"] == "C":
        key_block["planted_flaw"] = k_flaw
    if t_keys is not None:
        key_block["turn_keys"] = t_keys
        out["turn_keys"] = t_keys
    out["key"] = key_block

    if isinstance(out.get("material"), dict):
        out["material"]["text"] = mat
    else:
        out["material"] = {"text": mat, "source": {"invented": "material"}}

    row = out.get("row") if isinstance(out.get("row"), dict) else {}
    if row:
        row["material"] = mat
        if parts["kind"] == "E" and isinstance(row.get("quiz"), dict):
            row["quiz"]["verdict"] = k_verdict
            row["quiz"]["deciding"] = k_deciding
        else:
            row["verdict"] = k_verdict
            row["deciding"] = k_deciding
        if t_keys is not None:
            row["turn_keys"] = t_keys
        turns = row.get("turns")
        if isinstance(turns, list) and len(turns) > 1 and isinstance(turns[-1], dict):
            if new_material is not None or "new_material" in turns[-1]:
                turns[-1]["new_material"] = list(new_mat)
        if parts["kind"] == "C":
            row["planted_flaw"] = k_flaw
            row["checker_verdict"] = c_verdict
        out["row"] = row

    if isinstance(out.get("fields"), dict):
        if new_material is not None and "turn2_new_material" in out["fields"]:
            out["fields"]["turn2_new_material"] = list(new_mat)
        if parts["kind"] == "C":
            out["fields"]["checker_trace"] = c_trace
            out["fields"]["checker_verdict"] = c_verdict

    if isinstance(out.get("candidate"), dict) and isinstance(out["candidate"].get("text"), str):
        try:
            cand_fields = json.loads(out["candidate"]["text"])
            if isinstance(cand_fields, dict):
                if new_material is not None and "turn2_new_material" in cand_fields:
                    cand_fields["turn2_new_material"] = list(new_mat)
                if parts["kind"] == "C":
                    cand_fields["checker_trace"] = c_trace
                    cand_fields["checker_verdict"] = c_verdict
                out["candidate"]["text"] = json.dumps(cand_fields, ensure_ascii=False)
        except ValueError:
            pass

    return out


def _flip_number_in_token(tok: str, rng: random.Random, *, forbidden: set[str] | None = None) -> str:
    avoid = set(forbidden or ()) | {tok}
    for extra in (0, 1, 2, 3, 4, 5, 7, 10, 15, 20):
        if ":" in tok:
            h_str, m_str = tok.split(":", 1)
            h = int(h_str)
            delta = rng.choice((1, 2, 3)) + extra
            new_h = (h + delta) % 24
            if new_h == h:
                new_h = (h + 1) % 24
            cand = f"{new_h:02d}:{m_str}" if len(h_str) == 2 else f"{new_h}:{m_str}"
        elif "." in tok:
            head, tail = tok.split(".", 1)
            val = int(head.replace(",", ""))
            delta = rng.choice((1, 2, 5, 10)) + extra
            cand = f"{val + delta}.{tail}"
        elif "," in tok:
            val = int(tok.replace(",", ""))
            delta = rng.choice((100, 500, 1000)) + extra * 100
            cand = f"{val + delta:,}"
        else:
            val = int(tok)
            delta = rng.choice((1, 2, 3, 5)) + extra
            cand = str(val + delta)
        if cand not in avoid:
            return cand
    return f"{tok}9"


def _flip_line_for_verdict(line: str, statement: str, verdict: str, rng: random.Random) -> tuple[str | None, str | None, str | None, str]:
    """Apply the (b) verdict-flipping number rule to a single deciding line:
    - supported: change only a number that appears in the statement and appears exactly once on the deciding line,
      to a value that isn't in the statement.
    - unsupported: only when the statement and the deciding line differ in exactly one number, set the line's
      number to the statement's.
    Returns (new_line, old_num, new_num, skip_reason)."""

    line_matches = _find_numbers(line)
    if not line_matches:
        return None, None, None, "no number on deciding line"
    stmt_matches = _find_numbers(statement)
    stmt_nums = [m.group(1) for m in stmt_matches]
    line_nums = [m.group(1) for m in line_matches]

    if verdict == "supported":
        eligible = [m for m in line_matches if m.group(1) in stmt_nums and line_nums.count(m.group(1)) == 1]
        if not eligible:
            return None, None, None, "supported: no number appears in statement and exactly once on deciding line"
        chosen = rng.choice(eligible)
        old_num = chosen.group(1)
        new_num = _flip_number_in_token(old_num, rng, forbidden=set(stmt_nums) | set(line_nums))
        new_line = line[:chosen.start(1)] + new_num + line[chosen.end(1):]
        return new_line, old_num, new_num, ""

    if verdict == "unsupported":
        line_diff = [m for m in line_matches if m.group(1) not in stmt_nums]
        stmt_diff = [n for n in stmt_nums if n not in line_nums]
        if len(line_diff) != 1 or len(stmt_diff) != 1 or line_nums.count(line_diff[0].group(1)) != 1:
            return None, None, None, "unsupported: statement and deciding line do not differ in exactly one number"
        chosen = line_diff[0]
        old_num = chosen.group(1)
        new_num = stmt_diff[0]
        new_line = line[:chosen.start(1)] + new_num + line[chosen.end(1):]
        return new_line, old_num, new_num, ""

    return None, None, None, "cannot_tell verdict has no deciding line"


def _candidate_turns(parts: dict) -> list[tuple[int, str, list[str]]]:
    """Return turns to inspect on `parts`, checking the final turn first and turn 1 second on a two-turn row."""
    if parts.get("turn_keys") and isinstance(parts["turn_keys"].get("1"), dict):
        t1 = parts["turn_keys"]["1"]
        return [
            (2, parts["key_verdict"], list(parts["key_deciding"])),
            (1, str(t1.get("verdict") or ""), list(t1.get("deciding") or [])),
        ]
    return [(1, parts["key_verdict"], list(parts["key_deciding"]))]


def _perturb_b_material_flip(parts: dict, rng: random.Random) -> tuple[str | None, list[str] | None, dict | None, str]:
    """(b) material flip: in the deciding line (in base `material` or `turns[-1].new_material`), change a number so the
    true verdict flips."""

    mat_lines = parts["material"].splitlines()
    trailing = "\n" if parts["material"].endswith("\n") else ""
    new_mat_lines = list(parts.get("new_material") or [])
    statement = parts["statement"]
    last_reason = "no deciding line with a flippable number"

    for turn_num, t_verdict, t_deciding in _candidate_turns(parts):
        if t_verdict not in ("supported", "unsupported") or not t_deciding:
            last_reason = "cannot_tell verdict has no deciding line"
            continue
        dec_norms = {_ws(x) for x in t_deciding}

        # Check turns[-1].new_material first when on turn 2
        if turn_num == 2 and new_mat_lines:
            for idx, line in enumerate(new_mat_lines):
                if _ws(line) in dec_norms or any(_ws(d) in _ws(line) for d in t_deciding):
                    new_line, old_num, new_num, why = _flip_line_for_verdict(line, statement, t_verdict, rng)
                    if new_line is not None:
                        edited_new = list(new_mat_lines)
                        edited_new[idx] = new_line
                        return None, edited_new, {
                            "turn": turn_num,
                            "location": "new_material",
                            "line_index": idx,
                            "old_line": line,
                            "new_line": new_line,
                            "old_number": old_num,
                            "new_number": new_num,
                        }, ""
                    last_reason = why

        # Check base material
        for idx, line in enumerate(mat_lines):
            if _ws(line) in dec_norms or any(_ws(d) in _ws(line) for d in t_deciding):
                new_line, old_num, new_num, why = _flip_line_for_verdict(line, statement, t_verdict, rng)
                if new_line is not None:
                    edited_mat = list(mat_lines)
                    edited_mat[idx] = new_line
                    return "\n".join(edited_mat) + trailing, None, {
                        "turn": turn_num,
                        "location": "material",
                        "line_index": idx,
                        "old_line": line,
                        "new_line": new_line,
                        "old_number": old_num,
                        "new_number": new_num,
                    }, ""
                last_reason = why

    return None, None, None, last_reason


def _perturb_e_cannot_tell_trap(parts: dict, rng: random.Random) -> tuple[str | None, list[str] | None, dict | None, str]:
    """(e) cannot_tell trap: delete the deciding line from the material (or `turns[-1].new_material`), or swap that
    line's subject for another entity already named in the passage. Never add templated qualifiers or truncate lines."""

    mat_lines = parts["material"].splitlines()
    trailing = "\n" if parts["material"].endswith("\n") else ""
    new_mat_lines = list(parts.get("new_material") or [])
    last_reason = "cannot_tell source verdict or no deciding line to remove/swap"

    for turn_num, t_verdict, t_deciding in _candidate_turns(parts):
        if t_verdict not in ("supported", "unsupported") or not t_deciding:
            continue
        dec_set = {_ws(d) for d in t_deciding}

        # Check turns[-1].new_material first when on turn 2
        if turn_num == 2 and new_mat_lines:
            dec_in_new = [
                i for i, ln in enumerate(new_mat_lines)
                if _ws(ln) in dec_set or any(_ws(d) in _ws(ln) for d in t_deciding)
            ]
            if dec_in_new:
                non_dec_lines = [
                    ln for ln in mat_lines if ln.strip() and _ws(ln) not in dec_set
                ] + [
                    ln for i, ln in enumerate(new_mat_lines) if i not in dec_in_new and ln.strip()
                ]
                other_subjects = [
                    m.group(1).strip() for ln in non_dec_lines if (m := THE_SUBJECT_RE.match(ln))
                ]
                first_idx = dec_in_new[0]
                first_line = new_mat_lines[first_idx]
                m_dec = THE_SUBJECT_RE.match(first_line)
                if m_dec and other_subjects:
                    dec_subj = m_dec.group(1).strip()
                    candidates = [s for s in other_subjects if s.lower() != dec_subj.lower()]
                    if candidates and (not non_dec_lines or rng.random() < 0.5):
                        replacement_subj = rng.choice(candidates)
                        new_line = replacement_subj + m_dec.group(2)
                        edited_new = list(new_mat_lines)
                        edited_new[first_idx] = new_line
                        return None, edited_new, {
                            "turn": turn_num,
                            "location": "new_material",
                            "method": "swap_subject",
                            "line_index": first_idx,
                            "old_line": first_line,
                            "new_line": new_line,
                            "old_subject": dec_subj,
                            "new_subject": replacement_subj,
                        }, ""
                if non_dec_lines:
                    removed = [new_mat_lines[i] for i in dec_in_new]
                    kept_new = [ln for i, ln in enumerate(new_mat_lines) if i not in dec_in_new]
                    return None, kept_new, {
                        "turn": turn_num,
                        "location": "new_material",
                        "method": "delete_deciding_line",
                        "deleted_indices": dec_in_new,
                        "deleted_lines": removed,
                    }, ""

        dec_indices = [
            i for i, ln in enumerate(mat_lines)
            if _ws(ln) in dec_set or any(_ws(d) in _ws(ln) for d in t_deciding)
        ]
        if not dec_indices:
            last_reason = "deciding line not found in material"
            continue

        non_dec_lines = [ln for i, ln in enumerate(mat_lines) if i not in dec_indices and ln.strip()]
        other_subjects = [
            m.group(1).strip() for ln in non_dec_lines if (m := THE_SUBJECT_RE.match(ln))
        ]

        first_dec_idx = dec_indices[0]
        first_dec_line = mat_lines[first_dec_idx]
        m_dec = THE_SUBJECT_RE.match(first_dec_line)
        if m_dec and other_subjects:
            dec_subj = m_dec.group(1).strip()
            candidates = [s for s in other_subjects if s.lower() != dec_subj.lower()]
            if candidates and (not non_dec_lines or rng.random() < 0.5):
                replacement_subj = rng.choice(candidates)
                new_line = replacement_subj + m_dec.group(2)
                new_lines = list(mat_lines)
                new_lines[first_dec_idx] = new_line
                return "\n".join(new_lines) + trailing, None, {
                    "turn": turn_num,
                    "location": "material",
                    "method": "swap_subject",
                    "line_index": first_dec_idx,
                    "old_line": first_dec_line,
                    "new_line": new_line,
                    "old_subject": dec_subj,
                    "new_subject": replacement_subj,
                }, ""

        if non_dec_lines:
            removed = [mat_lines[i] for i in dec_indices]
            kept = [ln for i, ln in enumerate(mat_lines) if i not in dec_indices]
            return "\n".join(kept) + trailing, None, {
                "turn": turn_num,
                "location": "material",
                "method": "delete_deciding_line",
                "deleted_indices": dec_indices,
                "deleted_lines": removed,
            }, ""

        last_reason = "single-line material with no other subject to swap"

    return None, None, None, last_reason


def _perturb_f_clean_trace(parts: dict) -> tuple[str | None, str | None, dict | None, str]:
    """(f) clean trace, kind C only: replace the checker's trace with a sound one built from the item's own key:
    the key's deciding quote and its true verdict, with checker_verdict set to match, using the fixed trace frame."""

    if parts["kind"] != "C" or not parts["key_flaw"] or parts["key_flaw"] == "none":
        return None, None, None, "kind C row has planted_flaw == 'none'"
    joined = " ".join(x.strip() for x in parts["key_deciding"] if x.strip())
    if not joined or parts["key_verdict"] not in ("supported", "unsupported"):
        return None, None, None, "kind C row has cannot_tell verdict or no deciding lines for clean trace"
    if parts["key_verdict"] == "supported":
        new_trace = f'The material says: "{joined}". That matches the statement, so it is supported.'
    else:
        new_trace = f'The material says: "{joined}". That contradicts the statement, so it is unsupported.'
    new_cv = parts["key_verdict"]
    return new_trace, new_cv, {
        "old_checker_trace": parts["checker_trace"],
        "new_checker_trace": new_trace,
        "old_checker_verdict": parts["checker_verdict"],
        "new_checker_verdict": new_cv,
    }, ""


def run_perturb(input_path: Path, out_dir: Path, *, seed: int = 0) -> dict:
    """Generate sealed GPU items (`d`, `b`, `e`, `f`), sealed alternate key files (`a`, `c`, `g`), and `manifest.json`."""

    header, drafts, input_sha256 = _verify_seal_and_load(Path(input_path))
    validated = [(d, _validate_source_item(d, i)) for i, d in enumerate(drafts)]

    gpu_items: list[dict] = []
    keys_a: list[dict] = []
    keys_c: list[dict] = []
    keys_g: list[dict] = []
    manifest_items: list[dict] = []
    skipped: list[dict] = []

    flaw_choices = tuple(FLAW_DEFINITIONS.keys())

    for d, parts in validated:
        src_id = parts["source_row_id"]

        # (d) control: unchanged copy
        ctrl = _build_draft_record(d, parts, "d")
        gpu_items.append(ctrl)
        manifest_items.append({
            "item_id": ctrl["item_id"],
            "source_row_id": src_id,
            "perturbation_type": "d",
            "target": "gpu",
            "changes": {},
        })

        # (b) material flip
        rng_b = random.Random(f"{seed}:{src_id}:b")
        mat_b, new_mat_b, diff_b, why_b = _perturb_b_material_flip(parts, rng_b)
        if diff_b is not None and (mat_b is not None or new_mat_b is not None):
            item_b = _build_draft_record(d, parts, "b", material=mat_b, new_material=new_mat_b)
            gpu_items.append(item_b)
            manifest_items.append({
                "item_id": item_b["item_id"],
                "source_row_id": src_id,
                "perturbation_type": "b",
                "target": "gpu",
                "turn": diff_b["turn"],
                "changes": {"material": diff_b},
            })
        else:
            skipped.append({
                "source_row_id": src_id,
                "perturbation_type": "b",
                "reason": why_b,
            })

        # (e) cannot_tell trap
        rng_e = random.Random(f"{seed}:{src_id}:e")
        mat_e, new_mat_e, diff_e, why_e = _perturb_e_cannot_tell_trap(parts, rng_e)
        if diff_e is not None and (mat_e is not None or new_mat_e is not None):
            item_e = _build_draft_record(d, parts, "e", material=mat_e, new_material=new_mat_e)
            gpu_items.append(item_e)
            manifest_items.append({
                "item_id": item_e["item_id"],
                "source_row_id": src_id,
                "perturbation_type": "e",
                "target": "gpu",
                "turn": diff_e["turn"],
                "changes": {"material": diff_e},
            })
        else:
            skipped.append({
                "source_row_id": src_id,
                "perturbation_type": "e",
                "reason": why_e,
            })

        # (f) clean trace (kind C only)
        if parts["kind"] == "C":
            trace_f, cv_f, diff_f, why_f = _perturb_f_clean_trace(parts)
            if trace_f is not None and cv_f is not None and diff_f is not None:
                item_f = _build_draft_record(d, parts, "f", checker_trace=trace_f, checker_verdict=cv_f)
                gpu_items.append(item_f)
                manifest_items.append({
                    "item_id": item_f["item_id"],
                    "source_row_id": src_id,
                    "perturbation_type": "f",
                    "target": "gpu",
                    "changes": {"trace": diff_f},
                })
            else:
                skipped.append({
                    "source_row_id": src_id,
                    "perturbation_type": "f",
                    "reason": why_f,
                })

        # (a) verdict flip (key-only, over control; flips the final turn's key, keeping turn 1 untouched)
        rng_a = random.Random(f"{seed}:{src_id}:a")
        final_turn = 2 if parts.get("turn_keys") else 1
        if parts["key_verdict"] == "supported":
            flipped_v = "unsupported"
        elif parts["key_verdict"] == "unsupported":
            flipped_v = "supported"
        else:
            flipped_v = rng_a.choice(("supported", "unsupported"))
        item_a = _build_draft_record(d, parts, "a", key_verdict=flipped_v)
        keys_a.append(item_a)
        manifest_items.append({
            "item_id": item_a["item_id"],
            "source_row_id": src_id,
            "perturbation_type": "a",
            "target": "offline_key",
            "turn": final_turn,
            "changes": {"key": {"turn": final_turn, "old_verdict": parts["key_verdict"], "new_verdict": flipped_v}},
        })

        # (c) flaw relabel (key-only, kind C only, over control)
        if parts["kind"] == "C":
            rng_c = random.Random(f"{seed}:{src_id}:c")
            other_flaws = [f for f in flaw_choices if f != parts["key_flaw"]]
            new_flaw = rng_c.choice(other_flaws)
            item_c = _build_draft_record(d, parts, "c", key_flaw=new_flaw)
            keys_c.append(item_c)
            manifest_items.append({
                "item_id": item_c["item_id"],
                "source_row_id": src_id,
                "perturbation_type": "c",
                "target": "offline_key",
                "changes": {"key": {"old_planted_flaw": parts["key_flaw"], "new_planted_flaw": new_flaw}},
            })

        # (g) decoy deciding (key-only, over control)
        rng_g = random.Random(f"{seed}:{src_id}:g")
        g_done = False
        g_reason = "cannot_tell verdict has no deciding line to swap"
        all_dec_norms = {_ws(x) for vt in parts["verdict_turns"] for x in vt["key_deciding"]}
        for turn_num, t_verdict, t_deciding in _candidate_turns(parts):
            if t_verdict not in ("supported", "unsupported") or not t_deciding:
                continue
            dec_norms = {_ws(x) for x in t_deciding}
            in_new = bool(
                turn_num == 2
                and parts.get("new_material")
                and any(_ws(ln) in dec_norms or any(_ws(d_ln) in _ws(ln) for d_ln in t_deciding) for ln in parts["new_material"])
            )
            pool_lines = (
                parts["material"].splitlines() + list(parts.get("new_material") or [])
                if turn_num == 2
                else parts["material"].splitlines()
            )
            non_dec = [
                ln.strip() for ln in pool_lines
                if ln.strip()
                and _ws(ln) not in all_dec_norms
                and not any(_ws(d_ln) in _ws(ln) for d_ln in t_deciding)
            ]
            if not non_dec:
                g_reason = "no non-deciding line in material"
                continue
            content_candidates = [ln for ln in non_dec if len(ln.split()) >= 5] or non_dec
            decoy = rng_g.choice(content_candidates)
            if turn_num == 1 and parts.get("turn_keys"):
                tk_g = copy.deepcopy(parts["turn_keys"])
                tk_g["1"]["deciding"] = [decoy]
                item_g = _build_draft_record(d, parts, "g", turn_keys=tk_g)
            else:
                item_g = _build_draft_record(d, parts, "g", key_deciding=[decoy])
            keys_g.append(item_g)
            manifest_items.append({
                "item_id": item_g["item_id"],
                "source_row_id": src_id,
                "perturbation_type": "g",
                "target": "offline_key",
                "turn": turn_num,
                "changes": {
                    "key": {
                        "turn": turn_num,
                        "location": "new_material" if in_new else "material",
                        "old_deciding": t_deciding,
                        "new_deciding": [decoy],
                    }
                },
            })
            g_done = True
            break

        if not g_done:
            skipped.append({
                "source_row_id": src_id,
                "perturbation_type": "g",
                "reason": g_reason,
            })

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_header = copy.deepcopy(header)
    out_header["perturb_seed"] = seed
    out_header["source_sha256"] = input_sha256

    items_path = out_dir / "items.jsonl"
    items_sha = waves.write(items_path, out_header, gpu_items)

    keys_shas: dict[str, str] = {}
    for kname, klist in (("a", keys_a), ("c", keys_c), ("g", keys_g)):
        kpath = out_dir / f"keys-{kname}.jsonl"
        keys_shas[kname] = waves.write(kpath, out_header, klist)

    manifest = {
        "seed": seed,
        "input_file": Path(input_path).name,
        "input_sha256": input_sha256,
        "items_file": items_path.name,
        "items_sha256": items_sha,
        "keys_files": {k: f"keys-{k}.jsonl" for k in keys_shas},
        "keys_sha256": keys_shas,
        "counts": {
            "d": sum(1 for x in gpu_items if x["perturbation_type"] == "d"),
            "b": sum(1 for x in gpu_items if x["perturbation_type"] == "b"),
            "e": sum(1 for x in gpu_items if x["perturbation_type"] == "e"),
            "f": sum(1 for x in gpu_items if x["perturbation_type"] == "f"),
            "a": len(keys_a),
            "c": len(keys_c),
            "g": len(keys_g),
            "gpu_total": len(gpu_items),
            "skipped": len(skipped),
        },
        "items": manifest_items,
        "skipped": skipped,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "--items", dest="input_path", required=True, help="sealed file of confirmed-correct rows")
    p.add_argument("--seed", type=int, default=0, help="random seed for deterministic perturbations")
    p.add_argument("--out", required=True, help="output directory for sealed items.jsonl, keys-*.jsonl and manifest.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run_perturb(Path(args.input_path), Path(args.out), seed=args.seed)
    except PerturbError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
