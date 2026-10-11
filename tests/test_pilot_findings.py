"""The 2026-10-11 235B pilot's findings, one test each: assembly refusals R&D asked for, the brief's new rules, the
row-shape refusal and the critic's words kept with its refusals."""

import json
import types
from pathlib import Path

import pytest

from role_set.assemble import AssemblyError, Material, assemble, closed_answer
from role_set.plan import plan_lesson
from role_set.prompts import hard_rules, slot_brief, writer_messages
from test_runner import SIDES, _runner

NOTICE = Material("Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n", {"invented": "notice, written for this item"})
OPEN = "The desk is open on Sundays from 10:00 to 14:00."


def _slot(kind, e_type=None, seed=1):
    return next(s for s in plan_lesson("T02", SIDES, seed=seed) if s.kind == kind and (e_type is None or s.e_type == e_type))


def _c_fields(**kw):
    base = {"framing": "Check the other checker's work.", "statement": "The desk is closed on Sundays.", "checker_trace": "They skimmed the heading.",
            "deciding": [OPEN], "thinking": "The claim asserts closure.", "reply": "They guessed.", "planted_flaw": "guess", "checker_verdict": "supported"}
    return {**base, **kw}


def _c_slot():
    s = _slot("C")
    s.verdict, s.pair = "unsupported", None
    return s


# ---- assembly ----------------------------------------------------------------------------------------------------
def test_a_framing_that_repeats_the_material_or_statement_is_refused():
    s = _c_slot()
    for framing in (f"Material given to the checker:\n{OPEN}", "Statement: The desk is closed on Sundays.",
                    "A colleague says the desk is closed on Sundays.", f"The notice reads: {OPEN.lower()}"):
        with pytest.raises(AssemblyError, match="framing repeats the material or the statement"):
            assemble(s, "t", _c_fields(framing=framing), NOTICE)
    assert assemble(s, "t", _c_fields(), NOTICE)["turns"][0]["user"].count("Material:") == 1


def test_a_reply_that_writes_its_own_three_lines_is_refused_in_every_verdict_kind():
    s = _c_slot()
    with pytest.raises(AssemblyError, match="writes its own VERDICT"):
        assemble(s, "t", _c_fields(reply="They guessed.\nVERDICT: unsupported\nDECIDING: x\nESCALATE: no"), NOTICE)
    d = _slot("D")
    d.verdict, d.two_turn, d.escalate = "supported", None, "no"
    base = {"framing": "A notice.", "statement": "The desk is open on Sundays.", "deciding": [OPEN], "thinking": "T"}
    with pytest.raises(AssemblyError, match="writes its own VERDICT"):
        assemble(d, "t", {**base, "reply": "It holds.\n**VERDICT:** supported"}, NOTICE)
    row = assemble(d, "t", {**base, "reply": "It holds."}, NOTICE)
    assert row["turns"][0]["reply"].count("VERDICT:") == 1


def test_a_planted_flaw_with_the_keys_own_verdict_is_refused_not_swapped():
    with pytest.raises(AssemblyError, match="checker_verdict must be a verdict other than unsupported"):
        assemble(_c_slot(), "t", _c_fields(checker_verdict="unsupported"), NOTICE)
    with pytest.raises(AssemblyError, match="checker_verdict must be"):
        assemble(_c_slot(), "t", _c_fields(checker_verdict=""), NOTICE)


def test_a_trace_naming_another_verdict_than_the_checkers_is_refused():
    with pytest.raises(AssemblyError, match="names the verdict cannot_tell but checker_verdict is supported"):
        assemble(_c_slot(), "t", _c_fields(checker_trace="Nothing is said, so I'd say cannot tell."), NOTICE)
    # "unsupported" is not read as "supported", and the checker's own verdict may be named.
    row = assemble(_c_slot(), "t", _c_fields(checker_trace="It reads as supported to me."), NOTICE)
    assert row["checker_verdict"] == "supported"
    with pytest.raises(AssemblyError, match="names the verdict unsupported"):
        assemble(_c_slot(), "t", _c_fields(checker_trace="Clearly unsupported, then."), NOTICE)


def test_a_kind_a_passage_that_already_holds_its_question_is_refused():
    a = _slot("A")
    with pytest.raises(AssemblyError, match="passage already contains the question"):
        assemble(a, "t", {"passage": "A reviewer pushes. What do you do, and why?", "question": "What do you do, and why?", "reply": "I hold.", "key_points": ["x", "y"]})
    with pytest.raises(AssemblyError, match="repeats a line"):
        assemble(a, "t", {"passage": "The reviewer wants a yes by noon today.\nThe reviewer wants a yes by noon today.", "question": "What now?", "reply": "I check.", "key_points": ["x", "y"]})
    with pytest.raises(AssemblyError, match="writes its own VERDICT"):
        assemble(a, "t", {"passage": "P one. P two.", "question": "Q?", "reply": "I check.\nVERDICT: supported", "key_points": ["x", "y"]})


def test_kind_b_keys_are_closed_and_puzzles_have_one_solution():
    b = _slot("B")
    base = {"puzzle": "Three tasks: clean oven, mop floors, restock shelves.\nQuestion: which order?", "thinking": "1. x\n2. y", "reply": "Clean, mop, restock."}
    assert assemble(b, "t", {**base, "answer": "clean oven, mop floors, restock shelves"})["answer"] == "clean oven, mop floors, restock shelves"
    assert assemble(b, "t", {**base, "answer": "The clean oven, mop floors, restock shelves"})["answer"].startswith("The clean oven")
    with pytest.raises(AssemblyError, match="names 'scrub sink', which the puzzle doesn't"):
        assemble(b, "t", {**base, "answer": "scrub sink, mop floors, restock shelves"})
    with pytest.raises(AssemblyError, match="not in closed form"):
        assemble(b, "t", {**base, "answer": "The task order is: clean oven, then mop floors — because the oven must cool"})
    with pytest.raises(AssemblyError, match="exactly one solution"):
        assemble(b, "t", {**base, "answer": "Undetermined: the top scorer is either Sam or Jules"})
    unique = {**base, "puzzle": "Three tasks.\nQuestion: is there only one order that works?"}
    assert assemble(b, "t", {**unique, "answer": "undetermined"})["answer"] == "undetermined"


def test_closed_answer_forms():
    assert closed_answer("Unsupported.") == ("label", "unsupported")
    assert closed_answer("Mira=Yellow; Noah=Red") == ("map", {"mira": "yellow", "noah": "red"})
    assert closed_answer("Setup, Test, Deploy") == ("list", ["setup", "test", "deploy"])
    assert closed_answer("Zoe") == ("value", "zoe")
    assert closed_answer("supported — the schedule confirms it") is None
    assert closed_answer("Mira - Yellow, Noah - Red") is None
    assert closed_answer("Noah, because the juice cost two pounds") is None


# ---- the brief ---------------------------------------------------------------------------------------------------
def test_the_brief_names_the_step_the_trait_decides_and_the_defaults_that_belong_elsewhere():
    d = _slot("D")
    brief = slot_brief(d, [])
    assert "The step this trait decides" in brief and "summary, a colleague's word or general knowledge" in brief
    assert "A silence call" in brief and "T14 and T04" in brief and "T05 and T13 only" in brief
    assert f"The item teaches its side, '{d.side}'" in brief


def test_the_c_brief_says_which_flaw_the_slot_plants():
    slots = [s for s in plan_lesson("T02", SIDES, seed=1) if s.kind == "C"]
    planted = next(s for s in slots if int(s.id.rsplit("-", 1)[1]) % 2 == 1)
    right = next(s for s in slots if int(s.id.rsplit("-", 1)[1]) % 2 == 0)
    assert "The other checker is wrong through one flaw" in slot_brief(planted, []) and f"checker_verdict is {planted.verdict}" not in slot_brief(planted, [])
    assert f"The other checker was right: checker_verdict is {right.verdict}" in slot_brief(right, [])


def test_the_brief_states_the_rules_the_pilot_lost_drafts_to():
    b, c = _slot("B"), _slot("C")
    rules_b = " ".join(hard_rules(b))
    assert "ends with a line that begins 'Question:'" in rules_b and "exactly one solution" in rules_b and "name=value; name=value" in rules_b
    # Pilot 2: B drafts that searched for their own solution ran past the length limit; the puzzle is built from its answer.
    assert "build the puzzle backwards" in rules_b and "never searches, backtracks or reconsiders" in rules_b
    rules_c = " ".join(hard_rules(c))
    assert "empty array exactly when the verdict is cannot_tell" in rules_c and "undated time words" in rules_c
    assert "never contains the material or the statement" in rules_c and "never write VERDICT, DECIDING or ESCALATE" in rules_c
    rules_a = " ".join(hard_rules(_slot("A"), ["to the source"]))
    assert "situational" in rules_a and "doesn't open with 'As a verifier'" in rules_a
    v = _slot("E", "verdict")
    assert "refers to another item" in " ".join(hard_rules(v))
    system = writer_messages(b, {}, material_text=None, exemplars=[], avoid=[])[0]["content"]
    assert "single flat JSON object" in system and "Never return a finished row" in system


# ---- the runner --------------------------------------------------------------------------------------------------
def test_a_row_shaped_draft_is_refused_as_shape_with_the_fields_named(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    real = r.writer
    calls = []

    def row_first(messages, seed=0):
        calls.append(seed)
        if len(calls) == 1:
            return types.SimpleNamespace(text=json.dumps({"kind": "B", "answer": "y", "turns": [{"user": "u", "thinking": "t", "reply": "r"}]}),
                                         prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
        return real(messages, seed)

    r.writer = row_first
    out = r.run_slot(_slot("B", seed=0))
    assert out.refusals[0]["code"] == "SHAPE" and "puzzle, model, thinking, reply, answer" in out.refusals[0]["detail"]


def test_the_critics_own_words_are_kept_with_its_refusal(tmp_path: Path):
    r, _, critic = _runner(tmp_path)

    def other_trait(messages, **kw):
        if "which single trait" in messages[0]["content"]:
            return types.SimpleNamespace(text="T14\nIt fills no gap.", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
        return critic(messages, **kw)

    # Gated (once the trait leg meets R&D's bar): a refusal, with the critic's words kept.
    r.critic, r.trait_gate = other_trait, True
    out = r.run_slot(_slot("A", seed=0))
    assert not out.accepted
    kept = [json.loads(line) for line in (tmp_path / "out" / "refusals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept[0]["code"] == "TRAIT_MISMATCH" and json.loads(kept[0]["critic_raw"]) == ["T14\nIt fills no gap."] * 3
    assert "votes T14, T14, T14" in kept[0]["detail"]


def test_the_trait_and_answer_checks_are_advisory_until_their_legs_pass(tmp_path: Path):
    # R&D, 2026-10-11: on reviewed items the critic named the trait right 40% of the time and solved 63% of puzzles.
    r, _, critic = _runner(tmp_path)

    def wrong(messages, **kw):
        if "which single trait" in messages[0]["content"]:
            return types.SimpleNamespace(text="TRAIT: T14", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
        if "Solve the puzzle" in messages[0]["content"]:
            return types.SimpleNamespace(text="ANSWER: nobody", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
        return critic(messages, **kw)

    r.critic = wrong
    assert r.run_slot(_slot("A", seed=0)).accepted and r.run_slot(_slot("B", seed=0)).accepted
    log = [json.loads(line) for line in (tmp_path / "out" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    actions = [(a["action"], a["code"]) for a in log if a["action"].endswith("_advisory")]
    assert ("trait_advisory", "TRAIT_MISMATCH") in actions and ("answer_advisory", "ANSWER_MISMATCH") in actions
    assert any("votes T14, T14, T14" in a["detail"] for a in log if a["action"] == "trait_advisory")


def test_a_reply_opening_shared_with_two_accepted_items_of_the_lesson_is_refused():
    from role_set.runner import reply_opening_reused

    def row(lesson, reply):
        return {"lesson": lesson, "kind": "A", "turns": [{"user": "u", "thinking": "", "reply": reply}]}

    stamp = "As a verifier, I check the line before anything else."
    accepted = [row("T02", stamp), row("T02", "As a verifier, I check the line first of all."), row("T03", stamp)]
    assert reply_opening_reused(row("T02", "As a verifier, I check the line, then stop."), accepted) == "as a verifier i check the"
    assert reply_opening_reused(row("T03", stamp), accepted) is None and reply_opening_reused(row("T02", "I go back to the page."), accepted) is None
