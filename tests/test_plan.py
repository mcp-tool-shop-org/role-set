import pytest

from role_set.plan import (
    KINDS,
    PER_KIND,
    PUZZLE_STRUCTURES,
    PlanError,
    Slot,
    check_plan,
    plan_batch,
    plan_lesson,
)

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
            s.escalate = "yes"
    with pytest.raises(PlanError, match="everyday"):
        check_plan(slots)

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
