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


def _next_day(day, month):
    from role_set.generate.recipes import MONTH_DAYS

    return (day + 1, month) if day < MONTH_DAYS[month - 1] else (1, month % 12 + 1)


def test_spans_touch_and_some_boundaries_fall_mid_month():
    mid = shared_asks = 0
    for seed in range(60):
        spans, span, asked = t12_calendar(4, random.Random(seed))
        rec = {d["label"]: (d["day"], d["month"]) for d in span["days"]}
        for a, b in zip(spans, spans[1:]):
            end, start = rec[a.split(" to ")[1]], rec[b.split(" to ")[0]]
            assert _next_day(*end) == start  # no gap and no overlap
            mid += start[0] != 1
        # An asked day in a month a boundary splits, strictly inside its span.
        boundary_months = {rec[s.split(" to ")[0]][1] for s in spans if rec[s.split(" to ")[0]][0] != 1}
        shared_asks += rec[asked][1] in boundary_months and asked not in {x for s in spans for x in s.split(" to ")}
    assert mid >= 40 and shared_asks >= 10


def test_mid_month_boundaries_keep_newest_answers_near_their_share():
    builds = [build_t12(p, s, n=n) for p in sorted(T12_PACKS) for n in (3, 4) for s in range(1, 11)]
    newest = sum(not b.shortcuts["T12.newest"]["differs"] for b in builds)
    assert newest >= len(builds) // 5  # about 1 in n is expected; the keep filter used to leave 12 in 160
