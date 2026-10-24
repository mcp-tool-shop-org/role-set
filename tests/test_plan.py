import json
from pathlib import Path

import pytest

from role_set.cli import VERIFIER_SIDES
from role_set.write import trait_names_from_pack

REPO_ROOT = Path(__file__).resolve().parents[1]
from role_set.domains import assign as assign_domains, load as load_naics
from role_set.gates import load_gates
from role_set.mock import MockCritic, MockWriter, mock_excerpts
from role_set.plan import (
    KINDS,
    PER_KIND,
    PUZZLE_STRUCTURES,
    REAL_SOURCE_KINDS,
    PlanError,
    Slot,
    _resolve_all_silent_real_c,
    check_plan,
    plan_apply,
    plan_apply_batch,
    plan_batch,
    plan_lesson,
    select_and_trim,
)
from role_set.runner import Runner
from role_set.solvefirst import assign as assign_solvefirst, load_packs, lock_skeletons

SIDES = ("quotes the source", "stops at the source")


def test_a_lesson_plan_meets_every_floor_and_is_reproducible():
    a = plan_lesson("T02", SIDES, seed=7)
    b = plan_lesson("T02", SIDES, seed=7)
    assert [s.as_dict() for s in a] == [s.as_dict() for s in b]
    figures = check_plan(a)
    assert figures["items"] == len(KINDS) * PER_KIND
    assert figures["verdict_bearing"] == 18
    assert figures["cannot_tell"] >= 5 and figures["supported"] >= 5
    assert figures["pairs"] == 2 and figures["escalate_yes"] >= 1 and figures["real_source"] >= 4
    assert {s.side for s in a} == set(SIDES)
    assert sum(1 for s in a if s.kind == "B" and s.verification is False) >= 3
    assert len({s.structure for s in a if s.kind == "B"}) == PER_KIND


def test_different_seeds_change_the_assignment_not_the_counts():
    a = plan_lesson("T03", SIDES, seed=1)
    b = plan_lesson("T03", SIDES, seed=2)
    assert [s.verdict for s in a] != [s.verdict for s in b]
    assert check_plan(a)["cannot_tell"] == check_plan(b)["cannot_tell"]


def test_check_plan_refuses_broken_floors():
    slots = plan_lesson("T02", SIDES, seed=3)
    for s in slots:
        if s.verdict == "cannot_tell":
            s.verdict = "unsupported"
    with pytest.raises(PlanError, match="floors"):
        check_plan(slots)


def test_check_plan_refuses_missing_pairs_and_turns_and_escalate():
    base = plan_lesson("T02", SIDES, seed=3)

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        s.pair = None
    with pytest.raises(PlanError, match="pairs"):
        check_plan(slots)

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        s.two_turn = None
    with pytest.raises(PlanError, match="two-turn"):
        check_plan(slots)

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        s.escalate = "no"
    with pytest.raises(PlanError, match="ESCALATE yes"):
        check_plan(slots)

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        if s.verdict == "cannot_tell":
            s.escalate = "yes (money)"
    with pytest.raises(PlanError, match="everyday"):
        check_plan(slots)

    # A bare yes is what the gates refuse: the key names its domain.
    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        if s.escalate.startswith("yes ("):
            s.escalate = "yes"
    with pytest.raises(PlanError, match="as the gates read them"):
        check_plan(slots)
    assert [s.escalate for s in base if s.escalate != "no"] in (["yes (medical)"], ["yes (legal)"], ["yes (safety)"], ["yes (money)"])

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        s.multi_line = False
    with pytest.raises(PlanError, match="multi-line"):
        check_plan(slots)

    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        s.real_source = False
    with pytest.raises(PlanError, match="real-source"):
        check_plan(slots)


def test_check_plan_refuses_shape_errors():
    base = plan_lesson("T02", SIDES, seed=3)
    with pytest.raises(PlanError, match="one lesson"):
        check_plan(base + plan_lesson("T03", SIDES, seed=3))
    with pytest.raises(PlanError, match="kinds"):
        check_plan(base[1:])
    slots = [Slot(**s.as_dict()) for s in base]
    puzzles = [s for s in slots if s.kind == "B"]
    puzzles[1].structure = puzzles[0].structure
    with pytest.raises(PlanError, match="repeats"):
        check_plan(slots)
    slots = [Slot(**s.as_dict()) for s in base]
    for s in slots:
        if s.kind == "B":
            s.verification = True
    with pytest.raises(PlanError, match="non-verification"):
        check_plan(slots)


def test_a_batch_of_two_lessons_keeps_every_structure_at_most_twice():
    slots = plan_batch({"T02": SIDES, "T03": SIDES}, seed=0)
    counts = {}
    for s in slots:
        if s.kind == "B":
            counts[s.structure] = counts.get(s.structure, 0) + 1
    assert max(counts.values()) == 2


def test_a_batch_of_three_lessons_needs_a_bigger_menu():
    with pytest.raises(PlanError, match="menu"):
        plan_batch({"T02": SIDES, "T03": SIDES, "T06": SIDES}, seed=0)
    assert len(PUZZLE_STRUCTURES) == PER_KIND


def test_every_verifier_lesson_across_1000_seeds_has_valid_real_source_and_two_turn():
    for lesson, sides in VERIFIER_SIDES.items():
        buf_modes = (False, True) if lesson == "T10" else (False,)
        for buffers in buf_modes:
            for seed in range(1000):
                slots = plan_lesson(lesson, sides, seed=seed, buffers=buffers)
                check_plan(slots)
                for s in slots:
                    if s.kind == "D":
                        assert not (s.pair is not None and s.two_turn is not None), (lesson, seed, buffers, s.id)
                    if s.real_source:
                        assert s.pair is None and not s.two_turn and s.content_kind in REAL_SOURCE_KINDS, (
                            lesson,
                            seed,
                            buffers,
                            s.id,
                            s.content_kind,
                        )


def test_t08_and_t12_seed_1_do_not_put_real_source_on_notice_and_check_plan_refuses_bad_real_source():
    t08 = {s.id: s for s in plan_lesson("T08", VERIFIER_SIDES["T08"], seed=1)}
    t12 = {s.id: s for s in plan_lesson("T12", VERIFIER_SIDES["T12"], seed=1)}
    assert t08["T08-C-05"].content_kind == "notice" and t08["T08-C-05"].real_source is False
    assert t12["T12-C-06"].content_kind == "notice" and t12["T12-C-06"].real_source is False

    base = plan_lesson("T08", VERIFIER_SIDES["T08"], seed=1)
    notice_slots = [Slot(**s.as_dict()) for s in base]
    notice_slot = next(s for s in notice_slots if s.kind == "C" and s.content_kind == "notice" and s.pair is None)
    notice_slot.real_source = True
    with pytest.raises(PlanError, match=notice_slot.id):
        check_plan(notice_slots)

    paired_slots = [Slot(**s.as_dict()) for s in base]
    paired_slot = next(s for s in paired_slots if s.pair is not None)
    paired_slot.real_source = True
    with pytest.raises(PlanError, match=paired_slot.id):
        check_plan(paired_slots)

    two_turn_slots = [Slot(**s.as_dict()) for s in base]
    two_turn_slot = next(s for s in two_turn_slots if s.two_turn is not None)
    two_turn_slot.real_source = True
    with pytest.raises(PlanError, match=two_turn_slot.id):
        check_plan(two_turn_slots)


def test_buffered_plan_slots_and_rare_silent_real_c_branches():
    sides = VERIFIER_SIDES["T10"]
    train = plan_lesson("T10", sides, seed=0, buffers=True)
    apply = plan_apply("T10", sides, seed=0, buffers=True)
    assert len(train) == 48 and len(apply) == 12
    assert len(plan_batch({"T10": sides}, seed=0, buffers=True)) == 48
    assert len(plan_apply_batch({"T10": sides}, seed=0, buffers=True)) == 12
    buf_ids = [s.id for s in train + apply if s.buffer]
    assert buf_ids == [
        "T10-A-09",
        "T10-E-09",
        "T10-C-09",
        "T10-C-10",
        "T10-C-11",
        "T10-C-12",
        "T10-D-09",
        "T10-D-10",
        "T10-C-AP-06",
        "T10-D-AP-06",
    ]
    by_id = {s.id: s for s in train + apply}
    assert by_id["T10-C-09"].pair == "t10-pair-3" and by_id["T10-C-09"].verdict == "cannot_tell"
    assert by_id["T10-C-10"].pair == "t10-pair-3" and by_id["T10-C-10"].verdict == "unsupported" and by_id["T10-C-10"].multi_line
    assert by_id["T10-C-11"].verdict == "supported" and by_id["T10-C-11"].escalate.startswith("yes (") and by_id["T10-C-11"].real_source and by_id["T10-C-11"].multi_line
    assert by_id["T10-C-12"].verdict == "supported" and by_id["T10-C-12"].escalate == "no" and by_id["T10-C-12"].real_source and by_id["T10-C-12"].multi_line
    assert by_id["T10-D-09"].two_turn == "stands" and by_id["T10-D-09"].verdict == "supported" and by_id["T10-D-09"].escalate.startswith("yes (") and by_id["T10-D-09"].multi_line and not by_id["T10-D-09"].real_source
    assert by_id["T10-D-10"].two_turn == "changes" and by_id["T10-D-10"].verdict == "cannot_tell" and by_id["T10-D-10"].escalate == "no"
    assert by_id["T10-C-AP-06"].verdict == "supported" and by_id["T10-C-AP-06"].escalate.startswith("yes (") and by_id["T10-C-AP-06"].multi_line
    assert by_id["T10-D-AP-06"].verdict == "cannot_tell" and by_id["T10-D-AP-06"].escalate == "no"

    # Forced deal for _resolve_all_silent_real_c branch 1: the swap.
    forced_swap = [Slot(**s.as_dict()) for s in train if not s.buffer]
    real_c = [s for s in forced_swap if s.kind == "C" and s.real_source]
    non_c_bearing = [s for s in forced_swap if s.kind in ("D", "E") and s.verdict is not None]
    for s in real_c:
        s.verdict = "cannot_tell"
    for i, s in enumerate(non_c_bearing):
        s.verdict = "supported" if i < 6 else "unsupported"
    _resolve_all_silent_real_c("T10", 613, forced_swap, "legal")
    assert real_c[0].verdict == "supported"
    assert non_c_bearing[0].verdict == "cannot_tell"
    check_plan(forced_swap)

    # Seed 613 in the wild hits the all-cannot_tell real-source C deal and resolves cleanly.
    s613 = plan_lesson("T10", sides, seed=613, buffers=True)
    assert any(s.verdict != "cannot_tell" for s in s613 if not s.buffer and s.kind == "C" and s.real_source)

    # Forced deal for _resolve_all_silent_real_c branch 2: the structured PlanError naming lesson and seed.
    forced_err = [Slot(**s.as_dict()) for s in train if not s.buffer]
    for s in forced_err:
        if s.kind == "C" and s.real_source:
            s.verdict = "cannot_tell"
        elif s.kind in ("D", "E") and s.verdict is not None:
            s.verdict = "unsupported"
    with pytest.raises(PlanError, match=r"REFUSED: T10 \(seed 42\)"):
        _resolve_all_silent_real_c("T10", 42, forced_err, "legal")

    # Kind B and verdict E may not have buffers.
    bad_buf = [Slot(**s.as_dict()) for s in train] + [Slot(id="T10-B-09", lesson="T10", kind="B", side=sides[0], tier=1, buffer=True)]
    with pytest.raises(PlanError, match="kind B and verdict E have no buffers"):
        check_plan(bad_buf)


def test_all_40_single_exclusions_pass_across_1000_seeds():
    sides = VERIFIER_SIDES["T10"]
    for seed in range(1000):
        train = plan_lesson("T10", sides, seed=seed, buffers=True)
        apply = plan_apply("T10", sides, seed=seed, buffers=True)
        plan = train + apply
        rows = [s.as_dict() for s in plan]
        buffered_base = [
            s.id
            for s in plan
            if not s.buffer and (s.split == "apply" or s.kind in ("A", "C", "D") or (s.kind == "E" and s.e_type == "written"))
        ]
        assert len(buffered_base) == 40
        for eid in buffered_base:
            trimmed, subs = select_and_trim(rows, plan, {eid})
            assert len(trimmed) == 50 and subs, (seed, eid)


def test_select_and_trim_on_mock_batch_and_refusals(tmp_path: Path):
    sides = VERIFIER_SIDES["T10"]
    train = plan_lesson("T10", sides, seed=0, buffers=True)
    apply = plan_apply("T10", sides, seed=0, buffers=True)
    plan = train + apply
    naics = load_naics(REPO_ROOT / "roles" / "domains" / "naics-2022.json")
    assign_domains(plan, naics, seed=0)
    b_packs = load_packs()
    assign_solvefirst(plan, b_packs, seed=0)
    lock_skeletons(plan, b_packs)

    gates = load_gates()
    pack = {
        "traits/T10.md": "## 10. As a verifier, you keep things in proportion\n\nProportion decides.\n",
        "kinds.md": "kinds\n",
        "amendments/1.md": "## Amendment 1\n\nq\n",
    }
    answers = {}
    writer = MockWriter(answers=answers, keyed_order={s.id: i for i, s in enumerate(plan)})
    critic = MockCritic(answers=answers, trait="T10")
    runner = Runner(
        gates=gates,
        pack=pack,
        trait_names=trait_names_from_pack(pack),
        writer=writer,
        critic=critic,
        out_dir=tmp_path / "out",
        excerpts=mock_excerpts(),
        pins={"writer_revision": "deadbeef"},
        critique_scope="verdict",
    )
    outcomes = runner.run(plan)
    assert all(o.accepted for o in outcomes), [(o.slot_id, o.refusals) for o in outcomes if not o.accepted]
    rows = [json.loads(line) for line in (tmp_path / "out" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 60

    # Trimmed 40 + 10 mock batch with 0 exclusions passes check_batch.
    trimmed0, subs0 = select_and_trim(rows, plan, set(), gates=gates, models=runner.models)
    assert len(trimmed0) == 50 and subs0 == []
    gates.check_batch([r for r in trimmed0 if r.get("split") != "apply"], models=runner.models)

    # Every single exclusion of the 40 buffered slots passes check_plan and check_batch.
    buffered_slots = [
        s for s in plan if not s.buffer and (s.split == "apply" or s.kind in ("A", "C", "D") or (s.kind == "E" and s.e_type == "written"))
    ]
    assert len(buffered_slots) == 40
    assert any(s.id == "T10-C-01" and s.escalate.startswith("yes (") for s in buffered_slots)
    assert any(s.kind == "C" and s.real_source for s in buffered_slots)
    assert any(s.kind == "D" and s.multi_line for s in buffered_slots)
    assert any(s.split == "apply" and s.kind == "C" and s.escalate.startswith("yes (") for s in buffered_slots)
    assert any(s.pair == "t10-pair-1" for s in buffered_slots)
    assert any(s.pair == "t10-pair-2" for s in buffered_slots)

    for s in buffered_slots:
        trimmed, subs = select_and_trim(rows, plan, {s.id}, gates=gates, models=runner.models)
        assert len(trimmed) == 50 and subs, s.id
        gates.check_batch([r for r in trimmed if r.get("split") != "apply"], models=runner.models)
        if s.pair in ("t10-pair-1", "t10-pair-2"):
            assert len(subs) == 2 and {x["buffer_id"] for x in subs} == {"T10-C-09", "T10-C-10"}

    # Every single exclusion of an unbuffered slot (8 B and 2 verdict E) is refused with a structured PlanError.
    unbuffered_slots = [s for s in plan if not s.buffer and (s.kind == "B" or (s.kind == "E" and s.e_type == "verdict"))]
    assert len(unbuffered_slots) == 10
    for s in unbuffered_slots:
        with pytest.raises(PlanError, match="unbuffered"):
            select_and_trim(rows, plan, {s.id}, gates=gates)

    # Two exclusions needing the same single buffer are refused.
    with pytest.raises(PlanError, match="no surviving buffer assignment"):
        select_and_trim(rows, plan, {"T10-A-01", "T10-A-02"}, gates=gates)
    with pytest.raises(PlanError, match="no surviving buffer assignment"):
        select_and_trim(rows, plan, {"T10-E-01", "T10-E-02"}, gates=gates)
    with pytest.raises(PlanError, match="no surviving buffer assignment"):
        select_and_trim(rows, plan, {"T10-C-AP-01", "T10-C-AP-02"}, gates=gates)
    p1 = next(s.id for s in buffered_slots if s.pair == "t10-pair-1")
    p2 = next(s.id for s in buffered_slots if s.pair == "t10-pair-2")
    with pytest.raises(PlanError, match="buffer pair is unavailable"):
        select_and_trim(rows, plan, {p1, p2}, gates=gates)


def test_plan_never_pairs_non_supported_verdict_with_equivalent_wording_side(tmp_path: Path):
    import hashlib
    from role_set.cli import main as cli_main

    sides = VERIFIER_SIDES["T10"]
    for buffers in (True, False):
        for seed in range(5):
            slots = (
                plan_lesson("T10", sides, seed=seed, buffers=buffers, side_pairing="verdict_safe")
                + plan_apply("T10", sides, seed=seed, buffers=buffers, side_pairing="verdict_safe")
            )
            assert all(s.side_pairing == "verdict_safe" for s in slots)
            bad = [
                (s.id, s.verdict, s.side)
                for s in slots
                if s.verdict is not None and s.verdict != "supported" and s.side == "an equivalent wording still matches"
            ]
            assert bad == [], f"seed={seed} buffers={buffers}: {bad}"
            for s in slots:
                if s.verdict in ("cannot_tell", "unsupported"):
                    assert s.side == "judges the exact size"

    # default reproduces 4033ebd's base plan exactly (10 non-supported equivalent-wording slots on T10 seed 0 buffered).
    default_slots = plan_lesson("T10", sides, seed=0, buffers=True) + plan_apply("T10", sides, seed=0, buffers=True)
    safe_slots = (
        plan_lesson("T10", sides, seed=0, buffers=True, side_pairing="verdict_safe")
        + plan_apply("T10", sides, seed=0, buffers=True, side_pairing="verdict_safe")
    )
    assert all(s.side_pairing == "default" for s in default_slots)
    default_bad_ids = [
        s.id
        for s in default_slots
        if s.verdict is not None and s.verdict != "supported" and s.side == "an equivalent wording still matches"
    ]
    assert default_bad_ids == [
        "T10-C-02",
        "T10-C-06",
        "T10-C-08",
        "T10-D-02",
        "T10-D-08",
        "T10-E-08",
        "T10-C-10",
        "T10-D-10",
        "T10-D-AP-02",
        "T10-D-AP-06",
    ]
    for d_slot, v_slot in zip(default_slots, safe_slots):
        d_dict = {k: v for k, v in d_slot.as_dict().items() if k not in ("side", "side_pairing")}
        v_dict = {k: v for k, v in v_slot.as_dict().items() if k not in ("side", "side_pairing")}
        assert d_dict == v_dict
        if d_slot.id in default_bad_ids:
            assert d_slot.side == "an equivalent wording still matches"
            assert v_slot.side == "judges the exact size"
        else:
            assert d_slot.side == v_slot.side

    # Plan header and per-slot recording for default and verdict_safe; default equals 4033ebd's base plan.
    naics = REPO_ROOT / "roles" / "domains" / "naics-2022.json"
    out_def = tmp_path / "plan-default.json"
    out_safe = tmp_path / "plan-safe.json"
    assert cli_main(["plan", "--lessons", "T10", "--buffers", "--seed", "0", "--domains", str(naics), "--out", str(out_def)]) == 0
    assert (
        cli_main(
            [
                "plan",
                "--lessons",
                "T10",
                "--buffers",
                "--seed",
                "0",
                "--domains",
                str(naics),
                "--side-pairing",
                "verdict_safe",
                "--out",
                str(out_safe),
            ]
        )
        == 0
    )
    data_def = json.loads(out_def.read_text(encoding="utf-8"))
    data_safe = json.loads(out_safe.read_text(encoding="utf-8"))
    assert data_def["side_pairing"] == "default"
    assert all(s["side_pairing"] == "default" for s in data_def["slots"])
    assert data_safe["side_pairing"] == "verdict_safe"
    assert all(s["side_pairing"] == "verdict_safe" for s in data_safe["slots"])
    assert [
        s["id"]
        for s in data_safe["slots"]
        if s["verdict"] is not None and s["verdict"] != "supported" and s["side"] == "an equivalent wording still matches"
    ] == []
    # Stripping the new side_pairing key reproduces 4033ebd's exact plan JSON byte-for-byte.
    stripped_def = {
        k: ([{sk: sv for sk, sv in s.items() if sk != "side_pairing"} for s in v] if k == "slots" else v)
        for k, v in data_def.items()
        if k != "side_pairing"
    }
    stripped_bytes = json.dumps(stripped_def, indent=1).encode("utf-8")
    assert hashlib.sha256(stripped_bytes).hexdigest() == "421a38119dc9524ccf9f1b9a9d36eb68c0322ee54d955ea5e0dd6e536908191a"

    # check_plan under verdict_safe refuses a non-supported verdict paired with "an equivalent wording still matches".
    base_slots = plan_lesson("T10", sides, seed=0, buffers=True, side_pairing="verdict_safe")
    ct_slot = next(s for s in base_slots if s.verdict == "cannot_tell")
    ct_slot.side = "an equivalent wording still matches"
    with pytest.raises(PlanError, match="equivalent wording"):
        check_plan(base_slots, side_pairing="verdict_safe")

    apply_slots = plan_apply("T10", sides, seed=0, buffers=True, side_pairing="verdict_safe")
    ap_ct = next(s for s in apply_slots if s.verdict == "cannot_tell")
    ap_ct.side = "an equivalent wording still matches"
    with pytest.raises(PlanError, match="equivalent wording"):
        check_plan(apply_slots, side_pairing="verdict_safe")

    with pytest.raises(PlanError, match="side_pairing must be one of"):
        plan_lesson("T10", sides, seed=0, side_pairing="unknown")
    with pytest.raises(PlanError, match="side_pairing must be one of"):
        plan_apply("T10", sides, seed=0, side_pairing="unknown")

