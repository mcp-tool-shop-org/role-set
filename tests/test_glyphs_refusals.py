import json
from pathlib import Path

import pytest

from role_set.assemble import assemble
from role_set.gates import load_gates
from role_set.plan import plan_lesson
from role_set.runner import normalize_glyphs, truth_checks

CURLY = "’"


def _a_fields(passage: str) -> dict:
    return {
        "passage": passage,
        "question": "What do you leave alone when a statement looks wrong?",
        "reply": "I leave the statement as written and judge it against the page.",
        "key_points": ["judge the statement as written", "leave the wording alone"],
    }


def test_normalize_glyphs_makes_prose_plain_and_leaves_verbatim_lines():
    fields, n = normalize_glyphs({
        "passage": f"You don{CURLY}t fix it. “Quoted” words.",
        "key_points": [f"don{CURLY}t fix"],
        "deciding": [f"It{CURLY}s 4."],
        "turn2_new_material": [f"Note{CURLY}s line."],
    })
    assert fields["passage"] == 'You don\'t fix it. "Quoted" words.' and fields["key_points"] == ["don't fix"]
    assert fields["deciding"] == [f"It{CURLY}s 4."] and fields["turn2_new_material"] == [f"Note{CURLY}s line."]
    assert n == 4


def test_a_curly_trait_name_passes_the_real_gate_once_normalised():
    try:
        gates = load_gates()
    except Exception:  # pragma: no cover - CI installs the pinned package
        pytest.skip("aspire-si-gates not installed")
    slot = next(s for s in plan_lesson("T18", ("doesn't rewrite the statement", "still quotes the deciding line"), seed=7) if s.kind == "A")
    # Only "don't fix" names T18 here, so the glyph decides: the third rehearsal's GATE refusals had this shape.
    passage = f"As a verifier, you don{CURLY}t fix a claim you check. You judge the words that are there, and nothing else."
    raw = assemble(slot, "doesn't fix things", _a_fields(passage))
    with pytest.raises(gates.error_type(), match="doesn't name its trait"):
        gates.check_item(raw)
    fixed, n = normalize_glyphs(_a_fields(passage))
    assert n == 1
    gates.check_item(assemble(slot, "doesn't fix things", fixed))


def test_truth_checks_compare_quotes_without_the_glyphs():
    row = {"material": f"The shop{CURLY}s hours are 9 to 5.", "turns": [
        {"user": "Statement: The shop opens at 9.", "thinking": "The line \"The shop's hours are 9 to 5.\" settles it.", "reply": "Supported."}]}
    assert truth_checks(row, None) is None


def test_refused_drafts_are_kept_and_receipts_record_the_glyphs(tmp_path: Path):
    from test_runner import SIDES, _gates, _runner

    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")
    r, _, _ = _runner(tmp_path, gates=_gates(refuse_ids={slot.id}))
    out = r.run_slot(slot)
    assert not out.accepted
    kept = [json.loads(line) for line in (tmp_path / "out" / "refusals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 3 and {k["code"] for k in kept} == {"GATE"} and all(k["draft"] and k["slot"] == slot.id for k in kept)

    r2, writer, _ = _runner(tmp_path / "b")
    real = r2.writer

    def curly(messages, seed=0):
        import dataclasses

        c = real(messages, seed)
        return dataclasses.replace(c, text=c.text.replace("I reach", f"I{CURLY}m sure I reach"))

    r2.writer = curly
    assert r2.run_slot(slot).accepted
    receipt = json.loads((tmp_path / "b" / "out" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert receipt["glyphs_normalized"] == 1
