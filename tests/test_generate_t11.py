"""The T11 recipe: a claim decided only by reading across lines (find side), or held with a recorded contrast (confirm side)."""

import json
import random
from pathlib import Path

import pytest

from role_set.generate.build import BuildError, receipt
from role_set.generate.recipes import PACKS, T11_SIDES, _one_slot, build_t11, t11_refusal
from role_set.generate.render import render_puzzle
from role_set.generate.select import Floor, check_file, file_receipt, rests, select
from role_set.generate.shortcuts import prove, subject_reading
from role_set.puzzle import Table, check_puzzle, parse_model

FIXTURES = Path(__file__).parent / "fixtures"
MUST_REFUSE = json.loads((FIXTURES / "t11_round3_must_refuse.json").read_text(encoding="utf-8"))
MUST_PASS = json.loads((FIXTURES / "t11_batch6_must_pass.json").read_text(encoding="utf-8"))
READING = "T11.consistent_so_supported"


def _verdict(model):
    t = Table(parse_model(model))
    return t.verdict(t.all)


@pytest.mark.parametrize("pack", sorted(PACKS))
@pytest.mark.parametrize("side", T11_SIDES)
@pytest.mark.parametrize("seed", [1, 2])
def test_a_t11_puzzle_is_proven_and_its_wording_round_trips(pack, side, seed):
    b = build_t11(pack, seed, side)
    text = render_puzzle(b.model, PACKS[pack]["plural"], random.Random(seed))
    assert text.splitlines()[-1].startswith("Question: Claim: ")
    got = check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\n" + b.answer}, "falsifier_first")
    assert got.answer == b.answer and not got.lookup
    assert all(f["rule"][0] != "count" for f in b.model["facts"])
    assert t11_refusal(b.model, side) is None
    if side == "find":
        assert b.answer == "unsupported" and b.shortcuts[READING]["differs"] and b.contrast is None
    else:
        assert b.answer == "supported" and not b.shortcuts[READING]["differs"]
        # The contrast is real: swap the line and the claim is refuted, with an assignment left.
        c = b.contrast
        swapped = {**b.model, "facts": [{**f, "rule": c["to"] if f["line"] == c["line"] else f["rule"]} for f in b.model["facts"]]}
        assert c["from"] == next(f["rule"] for f in b.model["facts"] if f["line"] == c["line"])
        assert _one_slot(c["from"], c["to"]) and _verdict(swapped) == "unsupported"


def test_the_same_seed_rebuilds_the_same_t11_puzzle():
    a, b = build_t11("doors", 4, "find"), build_t11("doors", 4, "find")
    assert a.model == b.model and a.shortcuts == b.shortcuts
    assert build_t11("doors", 4, "confirm").model != a.model


def test_a_t11_side_is_named():
    with pytest.raises(BuildError, match="side"):
        build_t11("doors", 1, "value")


@pytest.mark.parametrize("fixture", MUST_REFUSE, ids=[f["id"] for f in MUST_REFUSE])
def test_round_3_count_decided_items_are_refused(fixture):
    assert "count" in t11_refusal(fixture["model"], fixture["side"])


@pytest.mark.parametrize("fixture", MUST_PASS, ids=[f["id"] for f in MUST_PASS])
def test_batch_6_find_side_items_pass_and_the_reading_differs(fixture):
    assert t11_refusal(fixture["model"], fixture["side"]) is None
    assert prove(fixture["model"], fixture["structure"], [READING])[READING]["differs"]


def _lamps(facts, claim):
    return {"entities": ["the tin lamp", "the iron lamp", "the lead lamp"],
            "attributes": {"light": {"values": ["lit", "dark"], "distinct": False}},
            "facts": [{"line": i + 2, "rule": r} for i, r in enumerate(facts)], "flavour": [1], "claim": claim}


def test_each_t11_refusal():
    tin_lit = ["is", "the tin lamp", "light", "lit"]
    assert t11_refusal({**_lamps([tin_lit], tin_lit), "claim": None, "asks": {"of": "the tin lamp", "attribute": "light"}},
                       "find") == "T11 items are claims"
    # Stated outright: one line decides it.
    assert "on its own" in t11_refusal(_lamps([["is_not", "the tin lamp", "light", "lit"],
                                               ["is", "the iron lamp", "light", "lit"]], tin_lit), "find")
    # Nothing decides the claim: neither side.
    open_model = _lamps([["is", "the iron lamp", "light", "lit"]], tin_lit)
    assert "is cannot_tell" in t11_refusal(open_model, "find") and "supported" in t11_refusal(open_model, "confirm")
    # Two lines that each name the tin lamp refute it together: the falsifier is in the subject's own lines.
    both = _lamps([["if", tin_lit, ["is", "the iron lamp", "light", "lit"]],
                   ["if", tin_lit, ["is_not", "the iron lamp", "light", "lit"]]], tin_lit)
    assert "already refute" in t11_refusal(both, "find")
    assert subject_reading(both)["value"] == "unsupported"


def test_one_slot_changes():
    a = ["if", ["is", "x", "s", "open"], ["is", "y", "s", "open"]]
    assert _one_slot(a, ["if", ["is", "x", "s", "open"], ["is", "y", "s", "shut"]])
    assert _one_slot(a, ["if", ["is", "x", "s", "open"], ["is_not", "y", "s", "open"]])
    assert not _one_slot(a, ["if", ["is", "x", "s", "shut"], ["is", "y", "s", "shut"]])
    assert _one_slot(["in", "x", "h", ["hot", "warm"]], ["in", "x", "h", ["hot", "cold"]])
    assert not _one_slot(["is", "x", "h", "hot"], ["in", "x", "h", ["hot", "cold"]])


def test_a_t11_file_mixes_the_falsifier_reading_and_contrasts():
    pool = [receipt(build_t11(p, s, side), "T11", PACKS[p], "falsifier_first")
            for p in sorted(PACKS) for side in T11_SIDES for s in (1, 2)]
    picks = select(pool)
    rested = [p.rests_on for p in picks]
    assert rested.count(READING) <= 2 and any(r.startswith("T11.contrast.") for r in rested)
    for p in picks:
        assert p.rests_on in rests(pool[p.index])
    record = file_receipt(pool, picks)
    assert record["problems"] == [] and [i["rests_on"] for i in record["items"]] == rested
    assert record["floor"]["min_newest_share"] == 0.2 and all(i["position"] is None for i in record["items"])
    assert check_file([(pool[p.index], p.rests_on) for p in picks]) == []


def test_the_newest_floor_is_a_share():
    assert Floor().min_newest == 1 and Floor(size=10).min_newest == 2 and Floor(size=4).min_newest == 1
    assert Floor(min_newest_share=0).min_newest == 0
