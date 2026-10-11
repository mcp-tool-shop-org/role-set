"""The closed-form rule is the gates' settings (aspire-si-gates 1.4.0): the brief states the run's setting, and the CLI
passes a run's own settings through to the gates."""

import argparse

import pytest

from role_set.cli import _closed_form_arg
from role_set.plan import plan_lesson
from role_set.prompts import closed_form_carry_rule, hard_rules
from test_runner import SIDES

SETTINGS = {"carry": "paraphrase", "share": 0.5, "distractor_max": 0.8, "distractor_margin": 0.15, "span": 8,
            "exempt_keyed_key_points": False}


def test_the_brief_states_the_runs_carry_setting():
    para = closed_form_carry_rule(SETTINGS)
    assert "in its own words" in para and "at least 50%" in para and "a run of 8 words from this item's question, thinking or reply" in para
    assert "word for word" not in para
    verb = closed_form_carry_rule({**SETTINGS, "carry": "verbatim"})
    assert "word for word" in verb and "own words" not in verb
    either = closed_form_carry_rule({**SETTINGS, "carry": "either", "share": 0.6, "span": 10})
    assert "word for word or" in either and "at least 60%" in either and "a run of 10 words" in either
    assert "as the run's gates require" in closed_form_carry_rule(None)


def test_the_written_e_rules_carry_the_setting_into_the_brief():
    slot = next(s for s in plan_lesson("T02", SIDES, seed=1) if s.kind == "E" and s.e_type == "written")
    rules = " ".join(hard_rules(slot, None, {**SETTINGS, "carry": "verbatim"}))
    assert "carries one of your key_points word for word" in rules
    rules = " ".join(hard_rules(slot, None, SETTINGS))
    assert "says one of your key_points in its own words" in rules


def test_the_closed_form_option_parses_a_json_object_and_nothing_else():
    assert _closed_form_arg(None) is None and _closed_form_arg("") is None
    assert _closed_form_arg('{"carry": "either", "share": 0.6}') == {"carry": "either", "share": 0.6}
    with pytest.raises(argparse.ArgumentTypeError, match="not JSON"):
        _closed_form_arg("{carry: either}")
    with pytest.raises(argparse.ArgumentTypeError, match="must be a JSON object"):
        _closed_form_arg('["either"]')
