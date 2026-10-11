"""The critic side of the 235B pilot's fixes (R&D's rulings, 2026-10-11): the trait asked as the deciding step by a
majority of three, the B answer closed and compared closed, and the trait leg of critic-control with its bar."""

import json
import types
from pathlib import Path

from role_set.critic import (TRAIT_SYSTEM, answer_messages, compare_answer, compare_trait_votes, majority, puzzle_for_critic,
                             same_answer, trait_messages, trait_named)
from role_set.critic_control import TRAIT_BAR, trait_leg
from role_set.plan import plan_lesson
from role_set.trait_steps import TRAIT_STEPS
from test_runner import SIDES, _runner


def _said(text):
    return types.SimpleNamespace(text=text, prompt_tokens=1, completion_tokens=1, model="m", seconds=0)


# ---- the trait ---------------------------------------------------------------------------------------------------
def test_the_trait_prompt_asks_for_the_deciding_step_with_the_contrasted_error_apart():
    msgs = trait_messages("thinking and reply", {"T14": "doesn't fill gaps", "T02": "goes to the source"}, TRAIT_STEPS)
    assert msgs[0]["content"] == TRAIT_SYSTEM and "not the error the item warns against" in TRAIT_SYSTEM
    assert "only for T05 and T13" in TRAIT_SYSTEM and "CONTRASTED:" in TRAIT_SYSTEM
    assert "- T02: goes to the source. Decides: a summary" in msgs[1]["content"] and "- T14: doesn't fill gaps. Decides:" in msgs[1]["content"]


def test_the_trait_check_reads_the_whole_item_and_tells_close_traits_apart():
    from role_set.critic import PAIR_DISTINCTIONS, teaching_text

    row = {"turns": [{"user": "The manager insists it shipped.", "thinking": "No new line.", "reply": "It stands."}], "key_points": ["k"]}
    text = teaching_text(row)
    assert "User: The manager insists it shipped." in text and "No new line." in text and '["k"]' in text
    user = trait_messages(text, {"T05": "independent", "T13": "doesn't agree just to please"})[1]["content"]
    assert "## Telling close traits apart" in user and all(p in user for p in PAIR_DISTINCTIONS)
    assert {p.split(":")[0] for p in PAIR_DISTINCTIONS} == {"T05 or T13", "T04 or T14", "T03 or T16", "T02 or T14", "T01 or T11", "T05 or T02", "T16 or T10"}


def test_answers_compare_without_case_or_a_leading_article_but_never_by_word_span():
    from role_set.assemble import bare

    assert bare("The Yard  Door") == "yard door" and compare_answer("ANSWER: yard door", "the yard door").ok
    assert compare_answer("ANSWER: C", "ticket C").code == "ANSWER_MISMATCH"
    assert "exactly as the puzzle names it" in answer_messages("Facts.\nQuestion: who?", "Zoe")[0]["content"]


def test_trait_named_reads_the_trait_line_or_a_bare_id():
    assert trait_named("TRAIT: T05\nCONTRASTED: T13") == "T05" and trait_named("**TRAIT:** t09\nCONTRASTED: none") == "T09"
    assert trait_named("T12") == "T12" and trait_named("I think it's precise.") == ""


def test_a_majority_of_three_decides_and_a_split_is_a_mismatch():
    assert majority(["T05", "T05", "T14"]) == "T05" and majority(["T01", "T02", "T03"]) is None
    ok = compare_trait_votes(["TRAIT: T02", "TRAIT: T02", "TRAIT: T14"], "T02")
    assert ok.ok and "votes T02, T02, T14" in ok.detail
    split = compare_trait_votes(["TRAIT: T01", "TRAIT: T02", "TRAIT: T03"], "T02")
    assert split.code == "TRAIT_MISMATCH" and "no majority" in split.detail


def test_the_runner_asks_the_trait_three_times_at_temperature_with_seeds(tmp_path: Path):
    r, _, critic = _runner(tmp_path)
    asked = []

    def recording(messages, **kw):
        if "which single trait" in messages[0]["content"]:
            asked.append(kw)
        return critic(messages, **kw)

    r.critic = recording
    out = r.run_slot(next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A"))
    assert out.accepted and asked == [{"temperature": 0.7, "seed": 0}, {"temperature": 0.7, "seed": 1}, {"temperature": 0.7, "seed": 2}]


# ---- the puzzle answer -------------------------------------------------------------------------------------------
def test_the_critic_gets_the_puzzle_without_its_pattern_instruction():
    puzzle = ("The oak tree is taller than the ash. Draw your thinking as a tree.\nThe elm is shortest.\n"
              "Question: which is tallest? Show your working step by step.\nThe point of the exercise is pruning.")
    got = puzzle_for_critic(puzzle)
    assert "The oak tree is taller than the ash." in got and "The elm is shortest." in got and "Question: which is tallest?" in got
    assert "Draw your" not in got and "Show your working" not in got and "point of the exercise" not in got


def test_the_answer_prompt_asks_for_a_closed_line_in_the_keys_form():
    assert "name=value; name=value" in answer_messages("Facts.\nQuestion: who?", "Mira=Yellow; Noah=Red")[0]["content"]
    assert "ordered list" in answer_messages("Facts.\nQuestion: order?", "setup, test, deploy")[0]["content"]
    assert "supported, unsupported or cannot_tell" in answer_messages("Facts.\nQuestion: ?", "unsupported")[0]["content"]
    assert "single value" in answer_messages("Facts.\nQuestion: who?", "Zoe")[0]["content"]


def test_closed_answers_compare_by_form():
    assert compare_answer("1. Mia is out.\nANSWER: Zoe", "Zoe").ok
    assert compare_answer("ANSWER: Noah=red; Mira=yellow", "Mira=Yellow; Noah=Red").ok
    assert compare_answer("ANSWER: test, setup, deploy", "setup, test, deploy").code == "ANSWER_MISMATCH"
    assert compare_answer("ANSWER: No", "unsupported").code == "ANSWER_MISMATCH"  # never yes/no for a verdict
    assert compare_answer("ANSWER: supported because it says so", "supported").code == "VOID"
    assert same_answer(("list", ["zoe"]), ("value", "zoe"))


def test_a_void_answer_is_asked_once_more(tmp_path: Path):
    r, _, critic = _runner(tmp_path)
    calls = []

    def first_without_answer(messages, **kw):
        if "Solve the puzzle" in messages[0]["content"]:
            calls.append(1)
            if len(calls) == 1:
                return _said("I worked it out but forgot the line.")
        return critic(messages, **kw)

    r.critic = first_without_answer
    out = r.run_slot(next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "B"))
    assert out.accepted and len(calls) == 2


def b(i, key):
    # Every candidate the answer-leg tests use is named in the puzzle, as R&D's rule asks.
    return {"id": f"T02-B-{i:02d}", "kind": "B", "answer": key,
            "turns": [{"user": f"Facts {i}. The names: Zoe, Mia, setup, test, deploy, Mira, Noah, Yellow, Red.\nQuestion: which?", "thinking": "", "reply": ""}]}


# ---- critic-control's trait leg ----------------------------------------------------------------------------------
def _rows():
    def row(i, lesson):
        return {"id": f"{lesson}-A-{i:02d}", "lesson": lesson, "kind": "A", "turns": [{"user": "u", "thinking": "", "reply": f"reply {lesson} {i}"}]}

    return [row(i, "T02") for i in range(10)] + [row(i, "T14") for i in range(10)]


def test_the_trait_leg_scores_against_the_known_trait_with_a_confusion_matrix():
    def critic(messages, temperature=0.0, seed=0):
        text = messages[1]["content"]
        # T02 items: always right. T14 items: right except items 0-3, which read as T02 (and item 0 splits).
        if "reply T14 0" in text:
            return _said(f"TRAIT: {['T02', 'T14', 'T05'][seed]}")
        for i in (1, 2, 3):
            if f"reply T14 {i}" in text:
                return _said("TRAIT: T02")
        return _said("TRAIT: T02" if "reply T02" in text else "TRAIT: T14")

    misses = []
    leg = trait_leg(critic, _rows() + [{"id": "x", "lesson": "T99", "turns": []}], {"T02": "goes to the source", "T14": "doesn't fill gaps"},
                    concurrency=2, log=misses.append)
    assert leg["items"] == 20 and leg["agree"] == 16 and leg["overall"] == 0.8
    assert leg["per_trait"]["T14"] == {"items": 10, "agree": 6, "rate": 0.6, "thin": False} and leg["worst_trait"] == 0.6
    assert leg["confusion"]["T14"] == {"T02": 3, "T14": 6, "split": 1} and leg["stability_unanimous"] == 0.95
    assert leg["bar"] == TRAIT_BAR and leg["passes"] is False and len(misses) == 4


def test_the_trait_leg_passes_only_when_every_part_of_the_bar_is_met():
    def right(messages, temperature=0.0, seed=0):
        return _said("TRAIT: T02" if "reply T02" in messages[1]["content"] else "TRAIT: T14")

    leg = trait_leg(right, _rows(), {"T02": "goes to the source", "T14": "doesn't fill gaps"})
    assert leg["overall"] == 1.0 and leg["stability_unanimous"] == 1.0 and leg["passes"] is True
    assert leg["seconds_per_call"] == {"calls": 60, "p50": 0.0, "p95": 0.0, "total": 0.0}
    assert leg["per_trait"]["T02"]["thin"] is False


def test_three_unreadable_votes_are_not_unanimous_and_small_traits_are_thin():
    def mumbles(messages, temperature=0.0, seed=0):
        return _said("I'm not sure which one.")

    rows = _rows()[:3]
    leg = trait_leg(mumbles, rows, {"T02": "goes to the source"})
    assert leg["stability_unanimous"] == 0.0 and leg["with_majority"] == 0.0 and leg["confusion"]["T02"] == {"split": 3}
    assert leg["per_trait"]["T02"]["thin"] is True and leg["passes"] is False


def test_the_answer_leg_scores_closed_keys_and_lists_every_disagreement():
    from role_set.critic_control import ANSWER_BAR, answer_leg

    rows = [b(1, "Zoe"), b(2, "setup, test, deploy"), b(3, "Mira=Yellow; Noah=Red"), b(4, "unsupported"),
            b(5, "Noah ordered juice because it cost two pounds"), {"id": "T02-A-01", "kind": "A", "turns": [{"user": "u", "thinking": "", "reply": ""}]},
            b(6, "the red lamp")]
    answers = {"Facts 1.": "ANSWER: Zoe", "Facts 2.": "ANSWER: test, setup, deploy", "Facts 3.": "ANSWER: noah=red; mira=yellow", "Facts 4.": "No idea."}

    def critic(messages, **kw):
        return _said(next(v for k, v in answers.items() if messages[1]["content"].startswith(k)))

    leg = answer_leg(critic, rows)
    assert leg["puzzles"] == 6 and leg["keys_left_out"] == ["T02-B-05"] and leg["combined"] == {"items": 4, "agree": 2, "rate": 0.5}
    assert leg["descriptive_keys_left_out"] == ["T02-B-06"]  # "the red lamp" where the puzzle names only colours
    assert leg["bar"] == ANSWER_BAR and leg["passes"] is False and leg["voids"] == 2 and leg["as_written_below_bar"] is True
    assert [(d["item"], d["code"], d["critic_answer"]) for d in leg["disagreements"]] == [
        ("T02-B-02", "ANSWER_MISMATCH", "test, setup, deploy"), ("T02-B-04", "VOID", None)]


def test_older_label_reason_keys_count_by_their_label_and_are_reported_apart():
    from role_set.critic_control import answer_leg, leg_key

    assert leg_key("unsupported — a blue tile is square, hence unmarked") == ("unsupported", "label_head")
    assert leg_key("cannot_tell: the notice never gives a distance") == ("cannot_tell", "label_head")
    assert leg_key("Undetermined: Rhea or Ulf was fastest") == (None, None) and leg_key("No - it fails") == (None, None)
    assert leg_key("Zoe") == ("Zoe", "as_written")
    rows = [b(1, "Zoe"), b(2, "supported — both parts are backed"), b(3, "unsupported — the tile is square"), b(4, "Undetermined: either")]
    said = {"Facts 1.": "ANSWER: Mia", "Facts 2.": "ANSWER: supported", "Facts 3.": "ANSWER: unsupported"}

    def critic(messages, **kw):
        assert "Undetermined" not in messages[1]["content"]
        return _said(next(v for k, v in said.items() if messages[1]["content"].startswith(k)))

    leg = answer_leg(critic, rows)
    assert leg["subsets"] == {"as_written": {"items": 1, "agree": 0, "rate": 0.0}, "label_head": {"items": 2, "agree": 2, "rate": 1.0}}
    assert leg["combined"]["rate"] == 0.667 and leg["keys_left_out"] == ["T02-B-04"] and leg["as_written_below_bar"] is True
    assert leg["disagreements"][0]["subset"] == "as_written" and leg["disagreements"][0]["compared_as"] == "Zoe"
    assert json.dumps(leg)  # the report is plain JSON
