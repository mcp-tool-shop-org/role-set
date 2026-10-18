"""The T11 recipe: a claim decided only by reading across lines (find side), or held with a recorded contrast (confirm side)."""

import json
import random
from pathlib import Path

import pytest

from role_set.generate.build import BuildError, receipt
from role_set.generate.recipes import (PACKS, T11_FILE, T11_SIDES, T11_SIDES_BUILT, _one_slot, build_t11, t11_contrast,
                                      t11_pool, t11_refusal)
from role_set.generate.render import render_puzzle
from role_set.generate.select import Floor, check_file, file_receipt, rests, select
from role_set.generate.shortcuts import names_in, prove, subject_reading
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
        # The contrast is real: swap the line and the verdict moves off supported, to what the receipt records.
        c = b.contrast
        swapped = {**b.model, "facts": [{**f, "rule": c["to"] if f["line"] == c["line"] else f["rule"]} for f in b.model["facts"]]}
        assert c["from"] == next(f["rule"] for f in b.model["facts"] if f["line"] == c["line"])
        assert _one_slot(c["from"], c["to"]) and _verdict(swapped) == c["contrast_verdict"]
        assert c["contrast_verdict"] in ("cannot_tell", "unsupported")
        # Q3 (R&D on #52, as amended): the change is on a line that doesn't name the claim's subject, into one that doesn't.
        ents = set(b.model["entities"])
        subject = names_in(b.model["claim"], ents)
        assert not names_in(c["from"], ents) & subject and not names_in(c["to"], ents) & subject
        assert c["contrast_names_subject"] is False


def test_a_t11_receipt_names_its_file():
    # R&D on #52: the receipt says which file of seeds its catalogue row was drawn for.
    assert receipt(build_t11("cups", 20, "find"), "T11", PACKS["cups"], "falsifier_first")["file_index"] == 0
    assert receipt(build_t11("cups", 21, "confirm"), "T11", PACKS["cups"], "falsifier_first")["file_index"] == 1


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


# ---- variety: the skeleton catalogue (R&D's floor on #52) ----------------------------------------------------------
SWEEP_PACKS = ("lamps", "cups", "doors")  # the acceptance sweep: 3 packs x seeds 1-20 = 60 builds a side


def test_the_committed_catalogue_is_current():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    got = subprocess.run([sys.executable, str(root / "scripts" / "t11_catalogue.py"), "--check"], capture_output=True,
                         text=True, timeout=300)
    assert got.returncode == 0, got.stdout + got.stderr


# The slow-test split (R&D, 2026-10-18): a case that takes over ~10 s under coverage runs in the sweep job, the
# rest of the parametrization stays in the test job. Under coverage on a CI runner these cases passed the 240 s
# faulthandler limit, and its dump took down the xdist worker (#93, twice).
# confirm: 130 s under coverage locally, past 240 s on a CI runner; find stays as the representative.
@pytest.mark.parametrize("side", [pytest.param(x, marks=pytest.mark.sweep) if x == "confirm" else x for x in T11_SIDES_BUILT])
def test_every_catalogue_row_builds_and_proves(side):
    from role_set.generate.recipes import _t11_catalogue, _t11_from_row
    from role_set.puzzle_check import skeleton

    for nv, rows in _t11_catalogue()["sides"][side].items():
        assert len({r["skeleton_sha256"] for r in rows}) == len(rows)
        pack = next(p for p in sorted(PACKS) if len(PACKS[p]["values"]) == int(nv))
        keys = {skeleton(_t11_from_row(pack, side, i + 1, row).model) for i, row in enumerate(rows)}
        assert len(keys) == len(rows), (side, nv)  # instantiating keeps each row's skeleton


@pytest.mark.parametrize("side", T11_SIDES_BUILT)
def test_the_variety_floor_holds_on_the_sweep(side):
    from collections import Counter

    from role_set.puzzle_check import skeleton

    builds = [build_t11(p, s, side) for p in SWEEP_PACKS for s in range(1, 21)]
    tally = Counter(skeleton(b.model) for b in builds)
    assert len(tally) >= 45 and max(tally.values()) <= 3, (len(tally), max(tally.values()))


def test_a_file_never_repeats_a_skeleton_within_a_stratum():
    from role_set.puzzle_check import skeleton

    for nv in (2, 3):
        stratum = [p for p in sorted(PACKS) if len(PACKS[p]["values"]) == nv]
        keys = [skeleton(build_t11(p, s, "find").model) for p in stratum for s in range(1, T11_FILE + 1)]
        assert len(set(keys)) == len(keys), nv


def _contrast_case(kept):
    from role_set.generate.build import Shape, _key, _norm, assignments, truth
    from role_set.puzzle import holds

    names = ("the green cup", "the blue cup", "the grey cup")
    claim = ["is", "the green cup", "heat", "warm"]
    shape = Shape(names, {"heat": {"values": ["hot", "warm", "cold"], "distinct": False}}, claim=claim,
                  structure="falsifier_first")
    sols = assignments(shape)
    pool = t11_pool(shape)
    bits = {_key(c): truth(shape, sols, c) for c in pool}
    m = _norm(shape)
    return t11_contrast(shape, sols, kept, pool, bits, [holds(m, x, claim) for x in sols], random.Random(0))


def test_a_contrast_only_on_the_subjects_own_lines_is_refused():
    # Q3: every line names the green cup, so any flip would be on the subject's own lines: a lookup in disguise.
    assert _contrast_case([["in", "the green cup", "heat", ["hot", "warm"]], ["is_not", "the green cup", "heat", "hot"]]) is None


def test_an_off_subject_contrast_moves_the_verdict_off_supported():
    # The blue cup's stated heat is the only line without the green cup; changing it leaves the green cup's heat open.
    got = _contrast_case([["if", ["is", "the blue cup", "heat", "hot"], ["is", "the green cup", "heat", "warm"]],
                          ["is", "the blue cup", "heat", "hot"]])
    assert got["line"] == 3 and got["contrast_verdict"] == "cannot_tell" and got["contrast_names_subject"] is False
    assert got["to"][1] == "the blue cup"

