"""The T12 recipe: which edition was in force on a date, solved on `contains`, proven by R&D's four T12 readings."""

import copy
import json
import random
from pathlib import Path

import pytest

from role_set.generate.recipes import T12_PACKS, T12_TEMPLATES, build_t12, t12_calendar
from role_set.generate.render import render_puzzle
from role_set.generate.shortcuts import REGISTRY, answer_reading, prove
from role_set.puzzle import check_puzzle, parse_model

FIXTURES = Path(__file__).parent / "fixtures"
MUST_PASS = json.loads((FIXTURES / "span_t12_must_pass.json").read_text(encoding="utf-8"))
T12_READINGS = [n for n, s in REGISTRY.items() if s["lesson"] == "T12"]


@pytest.mark.parametrize("pack", sorted(T12_PACKS))
@pytest.mark.parametrize("n", [3, 4])
def test_a_t12_puzzle_is_proven_and_its_wording_round_trips(pack, n):
    b = build_t12(pack, 1, n=n)
    p = T12_PACKS[pack]
    text = render_puzzle(b.model, p["plural"], random.Random(1), T12_TEMPLATES, p["singular"])
    assert text.splitlines()[-1].startswith(f"Question: Which {p['singular']} was in force on ")
    got = check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\n" + b.answer}, "constraint_grid")
    assert got.answer == b.answer and not got.lookup
    assert any(r["differs"] for r in b.shortcuts.values())
    # The clues map editions to spans; none states one outright (R&D: span facts come from clues).
    assert all(f["rule"][0] != "is" for f in b.model["facts"])


def test_the_same_seed_rebuilds_the_same_t12_puzzle_and_packs_differ():
    a, b = build_t12("issues", 5), build_t12("issues", 5)
    assert a.model == b.model and a.shortcuts == b.shortcuts
    assert build_t12("cards", 5).model["span"] != a.model["span"]


def test_the_calendar_is_contiguous_typed_and_marks_a_wrap():
    for seed in range(30):
        spans, span, asked = t12_calendar(4, random.Random(seed))
        days = span["days"]
        assert [d["ordinal"] for d in days] == list(range(len(days)))
        assert asked in {d["label"] for d in days}
        wrapped = any(d.get("year") for d in days)
        assert all("year" in d for d in days) == wrapped or not wrapped


@pytest.mark.parametrize("fixture", MUST_PASS, ids=[f["id"] for f in MUST_PASS])
def test_the_t12_rebuilds_carry_a_reading_that_names_another_edition(fixture):
    proofs = prove(fixture["model"], fixture["structure"], T12_READINGS)
    assert any(r["differs"] for r in proofs.values())


def _by_id(rid):
    return copy.deepcopy(next(f["model"] for f in MUST_PASS if f["id"] == rid))


def test_each_t12_reading():
    # B-02: issue 4 holds 14 July; issue 5 ran last; issue 6 has the highest number.
    m = _by_id("T12-B-02")
    assert answer_reading("T12.newest", m)["value"] == ["issue 5"]
    assert answer_reading("T12.highest_number", m)["value"] == ["issue 6"]
    assert answer_reading("T12.boundary_next", m) is None  # 14 July is no span's end or start
    # B-01: 31 May closes the first span, so the boundary reading hands it to the white card; no numbers in the names.
    m = _by_id("T12-B-01")
    assert answer_reading("T12.boundary_next", m)["value"] == ["the white card"]
    assert answer_reading("T12.highest_number", m) is None
    # An opening day goes to the span before.
    m["asks"] = {"contains": "1 June"}
    assert answer_reading("T12.boundary_next", m)["value"] == ["the black card"]
    # Month alone: 9 August's month is named by no span end in B-04 (1 June to 30 September holds it), so nothing to act on;
    # in B-05, 2 November's month starts the last span, which also holds it.
    assert answer_reading("T12.month_alone", _by_id("T12-B-04")) is None
    assert answer_reading("T12.month_alone", _by_id("T12-B-05")) is None


def test_month_alone_names_a_span_that_only_shares_the_month():
    # 20 May sits in 15 May to 31 August, and the span before ends on 14 May: a reader matching the month alone has both.
    m = _by_id("T12-B-01")
    m["attributes"]["span"]["values"] = ["1 March to 14 May", "15 May to 31 August", "1 September to 30 November"]
    m["span"]["bounds"] = {s: s.split(" to ") for s in m["attributes"]["span"]["values"]}
    m["span"]["days"] = [{"label": "1 March", "day": 1, "month": 3, "ordinal": 0}, {"label": "14 May", "day": 14, "month": 5, "ordinal": 1},
                         {"label": "15 May", "day": 15, "month": 5, "ordinal": 2}, {"label": "20 May", "day": 20, "month": 5, "ordinal": 3},
                         {"label": "31 August", "day": 31, "month": 8, "ordinal": 4}, {"label": "1 September", "day": 1, "month": 9, "ordinal": 5},
                         {"label": "30 November", "day": 30, "month": 11, "ordinal": 6}]
    m["asks"] = {"contains": "20 May"}
    parse_model(m)
    got = answer_reading("T12.month_alone", m)["value"]
    assert "the black card" in got and "the white card" in got  # both May-named spans' holders, the true one among them


@pytest.mark.parametrize("pack", sorted(T12_PACKS))
def test_a_batch_can_ask_for_a_newest_edition_answer(pack):
    from role_set.generate.build import answer_position

    for n in (3, 4):
        for seed in (1, 2, 3):
            b = build_t12(pack, seed, n=n, position=n - 1)
            assert answer_position(b.model) == "newest" and any(r["differs"] for r in b.shortcuts.values())
        assert answer_position(build_t12(pack, 1, n=n, position=0).model) == "earliest"


def test_a_position_outside_the_calendar_is_refused():
    from role_set.generate.build import BuildError

    with pytest.raises(BuildError, match="span index"):
        build_t12("issues", 1, n=3, position=3)


def test_answer_position_is_none_without_a_dated_question():
    from role_set.generate.build import answer_position
    from role_set.generate.recipes import build_t10

    assert answer_position(build_t10("cups", 1, "claim").model) is None


# ---- the skeleton catalogue and claim questions (R&D's T12 claim ruling; the variety follow-up to role-set#52) ------
def test_the_committed_t12_catalogue_is_current():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    got = subprocess.run([sys.executable, str(root / "scripts" / "t12_catalogue.py"), "--check"], capture_output=True,
                         text=True, timeout=600)
    assert got.returncode == 0, got.stdout + got.stderr


@pytest.mark.parametrize("pack", sorted(T12_PACKS))
@pytest.mark.parametrize("n", [3, 4])
def test_a_t12_claim_is_keyed_and_a_reading_flips_it(pack, n):
    from role_set.generate.build import receipt

    for seed in (1, 2, 3):
        b = build_t12(pack, seed, n=n, form="claim")
        p = T12_PACKS[pack]
        text = render_puzzle(b.model, p["plural"], random.Random(seed), T12_TEMPLATES, p["singular"])
        assert text.splitlines()[-1].startswith("Question: Claim: ") and " was in force on " in text.splitlines()[-1]
        got = check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\n" + b.answer}, "constraint_grid")
        assert got.answer == b.answer in ("supported", "unsupported")
        assert any(r["differs"] for r in b.shortcuts.values())
        r = receipt(b, "T12", p, "constraint_grid")
        assert r["form"] == "claim" and r["position"] is None and r["t11_reading"] in ("supported", "unsupported", "cannot_tell")
        assert "T11.consistent_so_supported" not in r["shortcuts"]  # a record only, never a reading to rest on


def test_the_skeleton_records_the_form():
    from role_set.puzzle_check import skeleton

    ask = build_t12("issues", 1).model
    claim = {k: v for k, v in ask.items() if k != "asks"}
    claim["claim"] = ["contains", ask["entities"][0], ask["asks"]["contains"]]
    assert skeleton(ask) != skeleton(claim) and '"form": "ask"' in skeleton(ask) and '"form": "claim"' in skeleton(claim)


def test_a_reading_answers_a_claim_with_a_verdict():
    b = build_t12("editions", 4, form="claim")
    for name, r in b.shortcuts.items():
        assert r["answer"] in ("supported", "unsupported", "cannot_tell"), name


def test_claims_keep_their_unsupported_share_in_a_file():
    from role_set.generate.build import receipt
    from role_set.generate.select import Floor, check_file, select

    pool = [receipt(build_t12(p, s, form="claim"), "T12", T12_PACKS[p], "constraint_grid") for p in sorted(T12_PACKS)
            for s in range(1, 11)]
    picks = select(pool)
    claims = [pool[x.index] for x in picks]
    assert sum(r["answer"] == "unsupported" for r in claims) >= 0.4 * len(claims)
    supported = [r for r in pool if r["answer"] == "supported"]
    file = [(r, next(n for n, v in r["shortcuts"].items() if v["differs"])) for r in supported[:5]]
    assert any("claims are unsupported" in m for m in check_file(file))
    assert not any("claims are unsupported" in m for m in check_file(file, Floor(min_unsupported_share=0)))
    # The cap (Argon's review, settled by R&D): five unsupported claims of five are refused, the same 40/60 band as T11.
    unsupported = [r for r in pool if r["answer"] == "unsupported"]
    five = [(r, next(n for n, v in r["shortcuts"].items() if v["differs"])) for r in unsupported[:5]]
    assert len(five) == 5 and any("5 of 5 claims are unsupported" in m for m in check_file(five))
    assert not any("claims are unsupported" in m for m in check_file(five, Floor(max_unsupported_share=1)))
    picked = [pool[x.index] for x in select(pool)]
    assert sum(r["answer"] == "unsupported" for r in picked) <= 0.6 * len(picked)


@pytest.mark.parametrize("n, band", [(1, (0, 1)), (3, (1, 2)), (5, (2, 3))])
def test_the_claims_band_is_in_counts_so_every_size_can_meet_it(n, band):
    # R&D on #64: a 0.4-0.6 share can't be met at 1 or 3 claims, so the band is floor(0.4n) to ceil(0.6n).
    from role_set.generate.select import Floor, _claims_balanced, claims_band

    assert claims_band(n, Floor()) == band
    for unsupported in range(n + 1):
        claims = [{"answer": "unsupported"}] * unsupported + [{"answer": "supported"}] * (n - unsupported)
        assert _claims_balanced(claims, Floor()) == (band[0] <= unsupported <= band[1])


def test_the_t12_variety_floor_holds_per_form_and_combined():
    from collections import Counter

    from role_set.puzzle_check import skeleton

    builds = {f: [build_t12(p, s, n=3, form=f) for p in ("issues", "cards", "sheets") for s in range(1, 11)] for f in ("ask", "claim")}
    combined = Counter(skeleton(b.model) for bs in builds.values() for b in bs)
    assert len(combined) >= 45 and max(combined.values()) <= 3
    assert sum(b.answer == "unsupported" for b in builds["claim"]) >= 0.4 * len(builds["claim"])


def test_a_t12_form_is_named():
    from role_set.generate.build import BuildError

    with pytest.raises(BuildError, match="form"):
        build_t12("issues", 1, form="value")

