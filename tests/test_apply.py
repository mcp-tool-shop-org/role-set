"""Apply items (run sheet B6; the schema in aspire-si#127, R&D's rulings on aspire-si#128): planned, assembled and kept
out of training. The writer prompts and the run wait on the maintainer's call on who writes them."""

import json
from pathlib import Path

import pytest

from role_set.assemble import AssemblyError, Material, assemble
from role_set.plan import plan_apply, plan_apply_batch
from role_set.runner import spent_span, trained_spans

NOTICE = Material("Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n", {"invented": "notice, written for this item"})
BASE = {"framing": "A notice from the front desk.", "statement": "The desk is open on Sundays.",
        "deciding": ["The desk is open on Sundays from 10:00 to 14:00."], "thinking": "T", "reply": "R"}


def _d_slot(lesson="T05", verdict="supported"):
    s = next(x for x in plan_apply(lesson, ("a", "b"), seed=0) if x.kind == "D")
    s.verdict, s.escalate, s.multi_line = verdict, "no", False
    return s


def test_an_apply_plan_is_five_c_and_five_d_per_lesson():
    slots = plan_apply("T05", ("goes to the source", "holds the page"), seed=3)
    assert [s.id for s in slots] == [f"T05-{k}-AP-{n:02d}" for k in "CD" for n in range(1, 6)]
    assert {s.split for s in slots} == {"apply"} and all(s.two_turn is None and s.pair is None for s in slots)
    verdicts = [s.verdict for s in slots]
    assert sorted(verdicts) == sorted(["supported"] * 4 + ["unsupported"] * 3 + ["cannot_tell"] * 3)
    assert sum(s.escalate != "no" for s in slots) == 1
    assert plan_apply("T05", ("goes to the source", "holds the page"), seed=3) == slots  # the same seed, the same plan
    batch = plan_apply_batch({"T05": ("x", "y"), "T10": ("x", "y")}, seed=0)
    assert len(batch) == 20 and len({s.id for s in batch}) == 20


def test_the_cli_writes_an_apply_plan(tmp_path: Path):
    from role_set.cli import main

    out = tmp_path / "apply.json"
    assert main(["plan", "--lessons", "T02,T10", "--seed", "1", "--out", str(out), "--apply"]) == 0
    slots = json.loads(out.read_text(encoding="utf-8"))["slots"]
    assert len(slots) == 20 and all(s["split"] == "apply" and "-AP-" in s["id"] for s in slots)


def test_an_apply_row_carries_its_split_and_is_one_turn():
    row = assemble(_d_slot(), "t", BASE, NOTICE)
    assert row["split"] == "apply" and len(row["turns"]) == 1 and "r10_twin" not in row
    s = _d_slot()
    s.two_turn = "stands"
    with pytest.raises(AssemblyError, match="one turn"):
        assemble(s, "t", {**BASE, "turn2_user": "A manager insists.", "turn2_thinking": "No new line.", "turn2_reply": "It stands."}, NOTICE)


def test_a_t10_apply_item_needs_a_keyed_companion_claim():
    s = _d_slot("T10")
    with pytest.raises(AssemblyError, match="r10_twin"):
        assemble(s, "t", BASE, NOTICE)
    twin = {"statement": "The desk is open every Sunday of the year from 9:00.", "verdict": "unsupported"}
    row = assemble(s, "t", {**BASE, "r10_twin": twin}, NOTICE)
    assert row["r10_twin"] == twin and row["split"] == "apply"
    with pytest.raises(AssemblyError, match="must differ"):
        assemble(s, "t", {**BASE, "r10_twin": {**twin, "verdict": "supported"}}, NOTICE)
    with pytest.raises(AssemblyError, match="repeats the item's own"):
        assemble(s, "t", {**BASE, "r10_twin": {"statement": "The desk is open on Sundays.", "verdict": "unsupported"}}, NOTICE)
    with pytest.raises(AssemblyError, match="r10_twin"):
        assemble(s, "t", {**BASE, "r10_twin": {"statement": "x", "verdict": "maybe"}}, NOTICE)


def test_a_t09_apply_item_says_whether_its_order_is_free():
    s = _d_slot("T09")
    s.content_kind = "notice"
    with pytest.raises(AssemblyError, match="order_free"):
        assemble(s, "t", BASE, NOTICE)
    assert assemble(s, "t", {**BASE, "order_free": True}, NOTICE)["order_free"] is True
    assert assemble(s, "t", {**BASE, "order_free": False}, NOTICE)["order_free"] is False
    with pytest.raises(AssemblyError, match="order_free"):
        assemble(s, "t", {**BASE, "order_free": "yes"}, NOTICE)
    s.content_kind = "changelog"
    with pytest.raises(AssemblyError, match="must be false"):
        assemble(s, "t", {**BASE, "order_free": True}, NOTICE)
    assert "order_free" not in assemble(_d_slot("T05"), "t", BASE, NOTICE)


@pytest.mark.parametrize("lesson", ["T01", "T04", "T14"])
def test_a_t01_or_t04_apply_item_says_whether_dropping_its_deciding_lines_leaves_silence(lesson):
    s = _d_slot(lesson)
    with pytest.raises(AssemblyError, match="silent_without_deciding"):
        assemble(s, "t", BASE, NOTICE)
    assert assemble(s, "t", {**BASE, "silent_without_deciding": True}, NOTICE)["silent_without_deciding"] is True
    assert assemble(s, "t", {**BASE, "silent_without_deciding": False}, NOTICE)["silent_without_deciding"] is False
    silent = _d_slot(lesson, verdict="cannot_tell")
    with pytest.raises(AssemblyError, match="no deciding|has none"):
        assemble(silent, "t", {**BASE, "deciding": [], "silent_without_deciding": True}, NOTICE)
    assert "silent_without_deciding" not in assemble(_d_slot("T05"), "t", BASE, NOTICE)


def test_a_trained_items_wording_is_spent_for_apply_items():
    trained = {"material": "Desk hours", "turns": [{"user": "Material:\nDesk hours\nRead the notice and tell me whether the desk opens on Sundays at all.", "thinking": "", "reply": ""}]}
    spent = trained_spans([trained])
    fresh = {"material": "Desk hours", "turns": [{"user": "Material:\nDesk hours\nIs the counter staffed at the weekend?", "thinking": "", "reply": ""}]}
    reused = {"material": "Desk hours", "turns": [{"user": "Material:\nDesk hours\nRead the notice and tell me whether the desk opens on Sundays at all, please.", "thinking": "", "reply": ""}]}
    assert spent_span(fresh, spent) is None
    got = spent_span(reused, spent)
    assert got in spent and len(got.split()) == 8
    # Material lines are quoted, not written, so a shared material line is never spent wording.
    shared_material = {"material": "Read the notice and tell me whether the desk opens on Sundays at all.",
                       "turns": [{"user": "Read the notice and tell me whether the desk opens on Sundays at all.", "thinking": "", "reply": ""}]}
    assert spent_span(shared_material, spent) is None


def test_the_runner_checks_an_apply_item_against_the_whole_training_set(tmp_path: Path):
    from test_runner import _gates, _runner

    seen = []
    gates = _gates(against=lambda row, accepted: seen.append(len(accepted)) or [])
    training = [{"id": f"T05-C-{n:02d}", "kind": "C", "lesson": "T05", "turns": [{"user": "u", "thinking": "", "reply": ""}]}
                for n in range(1, 4)]
    r, _, _ = _runner(tmp_path, gates=gates)
    r.training = training
    slot = next(s for s in plan_apply("T02", ("goes to the source", "holds the page"), seed=0) if s.kind == "D")
    r.run_slot(slot)
    assert seen and seen[0] == len(training)  # nothing accepted yet in this run, so only the training rows


def test_an_apply_plan_without_the_training_set_is_refused(tmp_path: Path, capsys):
    from role_set.cli import main
    from role_set.pack import PackManifest, pack_hash
    from test_write import TRAIT_FILES

    pack = tmp_path / "pack"
    for name, text in TRAIT_FILES.items():
        (pack / name).parent.mkdir(parents=True, exist_ok=True)
        (pack / name).write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(TRAIT_FILES)
    (pack / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    plan = tmp_path / "apply.json"
    assert main(["plan", "--lessons", "T02", "--seed", "1", "--out", str(plan), "--apply"]) == 0
    rc = main(["write", "--plan", str(plan), "--pack", str(pack), "--out", str(tmp_path / "o"), "--mock"])
    assert rc == 2 and "--training" in capsys.readouterr().err
