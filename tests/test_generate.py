"""The kind-B generator, slice 1: generate-then-reduce, the shortcut registry, the T10 recipe and template wording
(docs/generator-design.md)."""

import json
import random
from pathlib import Path

import pytest

from role_set.generate.build import BuildError, metrics, solver_version
from role_set.generate.recipes import PACKS, build_t10
from role_set.generate.render import render_puzzle, sentence
from role_set.generate.shortcuts import REGISTRY, answer, differs, prove, rewrite
from role_set.puzzle import PuzzleError, Table, check_puzzle, parse_model

FIXTURES = Path(__file__).parent / "fixtures"
ROUND2 = json.loads((FIXTURES / "t10_round2_must_refuse.json").read_text(encoding="utf-8"))


def _rendered(b, pack, seed):
    return render_puzzle(b.model, PACKS[pack]["plural"], random.Random(seed))


@pytest.mark.parametrize("pack", sorted(PACKS))
@pytest.mark.parametrize("seed", [1, 2])
def test_a_t10_open_set_puzzle_is_proven_and_its_wording_round_trips(pack, seed):
    b = build_t10(pack, seed, question="set")
    text = _rendered(b, pack, seed)
    got = check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\nAnswer: " + b.answer},
                       "underdetermined_with_explicit_set")
    assert got.answer == b.answer
    # The count is on the deciding path, and a size loosening changes the set (design pin: the trait step decides).
    assert b.trait_clue in [f["rule"] for f in b.model["facts"]] and b.trait_clue[0] == "count"
    assert any(p["differs"] for p in b.shortcuts.values())
    without = dict(b.model, facts=[f for f in b.model["facts"] if f["rule"] != b.trait_clue])
    assert answer(without, "underdetermined_with_explicit_set") != answer(b.model, "underdetermined_with_explicit_set")


@pytest.mark.parametrize("pack", sorted(PACKS))
def test_a_t10_claim_is_proven(pack):
    b = build_t10(pack, 1, question="claim")
    text = _rendered(b, pack, 1)
    assert "Question: Claim:" in text
    assert check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\n" + b.answer},
                        "falsifier_first").answer == b.answer
    assert any(p["differs"] for p in b.shortcuts.values())


def test_the_same_seed_rebuilds_the_same_puzzle():
    a, b = build_t10("cups", 7, question="set"), build_t10("cups", 7, question="set")
    assert a.model == b.model and a.answer == b.answer and a.shortcuts == b.shortcuts
    assert build_t10("cups", 8, question="set").model != a.model


def test_t10_refuses_a_single_value_question_at_once():
    # R&D on #48, ruling 2: no reading can decide one (loosened keeps the key possible; tightened gives the key or a
    # contradiction), so it is refused before any search, with that reason; and the question has no default.
    import inspect

    with pytest.raises(BuildError, match="no single-value questions"):
        build_t10("lamps", 1, question="value")
    with pytest.raises(BuildError, match="one of set, claim"):
        build_t10("lamps", 1, question="which")
    assert inspect.signature(build_t10).parameters["question"].default is inspect.Parameter.empty


def test_a_contradiction_is_recorded_but_proves_nothing():
    # R&D on #48, ruling 1: kept on the receipt with differs false.
    model = {"entities": ["the oak door", "the pine door"], "attributes": {"state": {"values": ["open", "shut"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["count", "state", "open", "<=", 1]},
                       {"line": 3, "rule": ["is", "the oak door", "state", "shut"]},
                       {"line": 4, "rule": ["is", "the pine door", "state", "shut"]}], "flavour": [1],
             "claim": ["is", "the oak door", "state", "shut"]}
    proofs = prove(model, "falsifier_first", ["T10.ceiling_read_as_exact"])
    assert proofs["T10.ceiling_read_as_exact"] == {"answer": "no assignment fits", "differs": False}


def test_the_receipt_metrics_and_the_solver_pin():
    b = build_t10("lamps", 2, question="set")
    m = metrics(b.model)
    assert m == b.metrics and m["assignments_explored"] == 16
    assert m["load_bearing_facts"] == len(b.model["facts"]) and m["lookup_distance"] >= 0
    assert len(solver_version()["sha256"]) == 64


@pytest.mark.parametrize("fixture", [f for f in ROUND2 if f["id"] != "T10-B-01"], ids=lambda f: f["id"])
def test_round2_t10_puzzles_have_no_size_reading_that_decides(fixture):
    # aspire-si#106 round 2 (833463b): the size word decided nothing; every applicable reading leaves the answer as it was.
    proofs = prove(fixture["model"], fixture["structure"], list(REGISTRY))
    assert proofs and not any(p["differs"] for p in proofs.values())


def test_round2_t10_b01_is_refused_by_puzzle_check():
    # Under #45's grouped keys its flat key ("south, west") no longer stands, so it is refused before any reading runs.
    f = next(x for x in ROUND2 if x["id"] == "T10-B-01")
    with pytest.raises(PuzzleError):
        check_puzzle({"puzzle": f["puzzle"], "model": f["model"], "answer": f["answer"], "reply": "x\n" + f["answer"]},
                     f["structure"])


def test_the_divergence_rules_by_answer_type():
    claim = {"kind": "claim", "value": "supported"}
    assert differs(claim, {"kind": "claim", "value": "cannot_tell"})
    assert not differs(claim, {"kind": "claim", "value": "supported"})
    assert not differs(claim, {"kind": "none", "value": "no assignment fits"})
    single = {"kind": "values", "value": ["warm"]}
    assert differs(single, {"kind": "values", "value": ["hot"]})
    assert not differs(single, {"kind": "values", "value": ["hot", "warm"]})
    assert differs(single, {"kind": "values", "value": ["cold", "hot"]})
    assert differs({"kind": "set", "value": "a; b"}, {"kind": "set", "value": "a; a and b; b"})


def test_a_reading_with_nothing_to_rewrite_is_skipped():
    model = {"entities": ["the oak door", "the pine door"], "attributes": {"state": {"values": ["open", "shut"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["count", "state", "open", "=", 1]}], "flavour": [1],
             "claim": ["is", "the oak door", "state", "open"]}
    assert rewrite("T10.floor_read_as_exact", model) is None
    assert set(prove(model, "falsifier_first", list(REGISTRY))) == {"T10.exact_read_as_floor", "T10.exact_read_as_ceiling"}


def test_templates_agree_in_number_and_read_negative():
    rng = random.Random(0)
    assert sentence(["count", "state", "open", "=", 1], "doors", rng) == "Exactly one of the doors is open."
    assert sentence(["count", "state", "open", ">=", 2], "doors", rng) == "At least two of the doors are open."
    assert "not" in sentence(["not_in", "the oak door", "state", ["open", "shut"]], "doors", rng) or \
        "neither" in sentence(["not_in", "the oak door", "state", ["open", "shut"]], "doors", rng)
    assert sentence(["offset", "the oak door", "the pine door", "place", 1], "doors", rng) == "The pine door is next on from the oak door."
    t = Table(parse_model({"entities": ["a", "b"], "attributes": {"s": {"values": ["x", "y"], "distinct": False}},
                           "facts": [{"line": 2, "rule": ["is", "a", "s", "x"]}], "flavour": [1], "asks": {"attribute": "s", "of": "b"}}))
    assert len(t.rows) == 4


def test_a_receipt_carries_the_seed_readings_metrics_and_pins():
    from role_set.generate.build import receipt

    b = build_t10("doors", 3, question="set")
    r = receipt(b, "T10", PACKS["doors"], "underdetermined_with_explicit_set")
    assert r["seed"] == 3 and r["recipe"] == "T10" and r["answer"] == b.answer and r["shortcuts"] == b.shortcuts
    assert len(r["skeleton_sha256"]) == len(r["pack_sha256"]) == len(r["solver"]["sha256"]) == 64
    json.dumps(r)


def test_round5_t10_b01_carries_a_deciding_size_reading():
    # R&D's optional must-pass (aspire-si 0c181e4): two counts, and each bound read as exact changes the set.
    (f,) = json.loads((FIXTURES / "t10_round5_must_pass.json").read_text(encoding="utf-8"))
    check_puzzle({"puzzle": f["puzzle"], "model": f["model"], "answer": f["answer"], "reply": f["reply"]}, f["structure"])
    proofs = prove(f["model"], f["structure"], list(REGISTRY))
    assert any(p["differs"] for p in proofs.values())
