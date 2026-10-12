"""`contains`: which edition was in force on a date, worked out by the solver (R&D, 2026-10-11 and 2026-10-12)."""

import copy
import json
from pathlib import Path

import pytest

from role_set.puzzle import PuzzleError, Table, check_puzzle, parse_model
from role_set.puzzle_check import skeleton

FIXTURES = Path(__file__).parent / "fixtures"
MUST_PASS = json.loads((FIXTURES / "span_t12_must_pass.json").read_text(encoding="utf-8"))
MUST_REFUSE = json.loads((FIXTURES / "t12_round3_must_refuse.json").read_text(encoding="utf-8"))


def _check(f):
    fields = {"puzzle": f["puzzle"], "model": copy.deepcopy(f["model"]), "answer": f["answer"], "reply": f["reply"]}
    return check_puzzle(fields, f.get("structure", "constraint_grid"))


def _days(*specs):
    return [{"label": label, "day": int(label.split()[0]), "month": month, "ordinal": i, **({"year": y} if y is not None else {})}
            for i, (label, month, y) in enumerate(specs)]


CARDS = ("1. Three ferry fare cards, the red card, the white card and the black card, were in force one after another, one "
         "card to each span, in this order: 1 March to 31 May, 1 June to 31 August and 1 September to 30 November.\n"
         "2. The red card and the black card were never in force back to back.\n"
         "3. The black card came into force before the red card.\n"
         "Question: Which card was in force on 14 April?")
SPANS = ["1 March to 31 May", "1 June to 31 August", "1 September to 30 November"]
CARDS_MODEL = {
    "entities": ["the red card", "the white card", "the black card"],
    "attributes": {"span": {"values": SPANS, "ordered": True}},
    "facts": [{"line": 2, "rule": ["not_adjacent", "the red card", "the black card", "span"]},
              {"line": 3, "rule": ["before", "the black card", "the red card", "span"]}],
    "flavour": [1],
    "span": {"attribute": "span",
             "days": _days(("1 March", 3, None), ("14 April", 4, None), ("31 May", 5, None), ("1 June", 6, None),
                           ("31 August", 8, None), ("1 September", 9, None), ("30 November", 11, None)),
             "bounds": {s: s.split(" to ") for s in SPANS}},
    "asks": {"contains": "14 April"},
}


def _ask(day):
    """CARDS asking about another day, with the day records naming only the days that puzzle names."""

    m = copy.deepcopy(CARDS_MODEL)
    keep = [d for d in m["span"]["days"] if d["label"] != "14 April" or day == "14 April"]
    m["span"]["days"] = [dict(d, ordinal=i) for i, d in enumerate(keep)]
    m["asks"] = {"contains": day}
    return CARDS.replace("14 April", day), m


def _fields(puzzle, model, answer):
    return {"puzzle": puzzle, "model": copy.deepcopy(model), "answer": answer, "reply": "x\n" + answer}


@pytest.mark.parametrize("fixture", MUST_PASS, ids=[f["id"] for f in MUST_PASS])
def test_the_t12_rebuilds_pass_on_contains(fixture):
    # The Specialist's T12-B-01..05 (aspire-si 0c181e4), re-modelled so the solver does the date matching itself.
    got = _check(fixture)
    assert got.answer == fixture["answer"] and not got.lookup


@pytest.mark.parametrize("fixture", MUST_REFUSE, ids=[f["id"] for f in MUST_REFUSE])
def test_the_round3_placeholder_puzzles_stay_refused(fixture):
    # aspire-si#106 round 3: a date as an entity with a made-up span value, and a conditional handing it the fee.
    with pytest.raises(PuzzleError, match="is a date or a time, not an entity"):
        _check(fixture)


def test_the_solver_finds_the_holder_inclusive_at_both_ends():
    assert check_puzzle(_fields(CARDS, CARDS_MODEL, "the black card"), "constraint_grid").answer == "the black card"
    puzzle, m = _ask("31 May")
    assert check_puzzle(_fields(puzzle, m, "the black card"), "constraint_grid").answer == "the black card"
    puzzle, m = _ask("1 September")
    assert check_puzzle(_fields(puzzle, m, "the red card"), "constraint_grid").answer == "the red card"


def test_contains_is_never_a_fact_line_and_names_only_declared_days():
    m = copy.deepcopy(CARDS_MODEL)
    m["facts"].append({"line": 3, "rule": ["contains", "the white card", "14 April"]})
    with pytest.raises(PuzzleError, match="never state it as a fact line"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    m["asks"] = {"contains": "15 April"}
    with pytest.raises(PuzzleError, match="not one of the item's named days"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    del m["span"]
    with pytest.raises(PuzzleError, match="asks.contains needs model.span"):
        parse_model(m)
    with pytest.raises(PuzzleError, match="doesn't name that day"):
        check_puzzle(_fields(CARDS.replace("on 14 April", "in spring"), CARDS_MODEL, "the black card"), "constraint_grid")


def test_day_records_follow_the_calendar_and_mark_a_year_wrap():
    m = copy.deepcopy(CARDS_MODEL)
    m["span"]["days"][1] = {"label": "14 April", "day": 14, "month": 3, "ordinal": 1}
    with pytest.raises(PuzzleError, match="its label should read '14 March'"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    days = m["span"]["days"]
    days[2], days[3] = dict(days[3], ordinal=2), dict(days[2], ordinal=3)  # 1 June listed before 31 May
    with pytest.raises(PuzzleError, match="list the days in calendar order"):
        parse_model(m)
    wrap = ["1 October to 31 December", "1 January to 31 March"]
    wm = {"entities": ["A", "B"], "attributes": {"span": {"values": wrap, "ordered": True}},
          "facts": [{"line": 2, "rule": ["before", "A", "B", "span"]}], "flavour": [1],
          "span": {"attribute": "span", "bounds": {s: s.split(" to ") for s in wrap},
                   "days": _days(("1 October", 10, None), ("31 December", 12, None), ("1 January", 1, None), ("31 March", 3, None))},
          "asks": {"contains": "1 January"}}
    with pytest.raises(PuzzleError, match="mark a year wrap"):
        parse_model(wm)
    wm["span"]["days"] = _days(("1 October", 10, 0), ("31 December", 12, 0), ("1 January", 1, 1), ("31 March", 3, 1))
    assert parse_model(wm).span.day("1 January") == 2


def test_spans_are_well_formed_and_strictly_ordered():
    m = copy.deepcopy(CARDS_MODEL)
    m["span"]["bounds"]["1 June to 31 August"] = ["31 August", "1 June"]
    with pytest.raises(PuzzleError, match="starts after it ends"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    m["span"]["bounds"]["1 June to 31 August"] = ["1 June", "30 November"]
    with pytest.raises(PuzzleError, match="its label should read '1 June to 30 November'"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    m["attributes"]["span"]["values"] = [SPANS[1], SPANS[0], SPANS[2]]
    with pytest.raises(PuzzleError, match="listed in start order"):
        parse_model(m)
    shared = ["1 March to 31 May", "1 March to 31 August", "1 September to 30 November"]
    m = copy.deepcopy(CARDS_MODEL)
    m["attributes"]["span"]["values"] = shared
    m["span"]["bounds"] = {s: s.split(" to ") for s in shared}
    with pytest.raises(PuzzleError, match="share a start day"):
        parse_model(m)
    m = copy.deepcopy(CARDS_MODEL)
    m["attributes"]["span"]["ordered"] = False
    with pytest.raises(PuzzleError, match="ordered attribute whose values are the spans"):
        parse_model(m)


OVER = ["1 March to 30 June", "1 June to 31 August", "1 September to 30 November"]


def _overlap(owner_line=None, rule="earlier", word="earlier"):
    puzzle = ("1. Three ferry fare cards, the red card, the white card and the black card, one card to each span, in this "
              "order: 1 March to 30 June, 1 June to 31 August and 1 September to 30 November.\n"
              "2. The red card and the black card were never in force back to back.\n"
              "3. The black card came into force before the red card.\n"
              + (f"4. A day two cards both list belongs to the {word} card.\n" if owner_line else "")
              + "Question: Which card was in force on 14 June?")
    m = copy.deepcopy(CARDS_MODEL)
    m["attributes"]["span"]["values"] = OVER
    m["span"]["bounds"] = {s: s.split(" to ") for s in OVER}
    m["span"]["days"] = _days(("1 March", 3, None), ("1 June", 6, None), ("14 June", 6, None), ("30 June", 6, None),
                              ("31 August", 8, None), ("1 September", 9, None), ("30 November", 11, None))
    m["asks"] = {"contains": "14 June"}
    if owner_line:
        m["span"]["owner"] = {"rule": rule, "line": owner_line}
    return puzzle, m


def test_an_overlap_has_both_holders_unless_the_item_states_an_owner():
    puzzle, m = _overlap()
    got = check_puzzle(_fields(puzzle, m, "the black card, the white card"), "constraint_grid")
    assert set(got.answer.split(", ")) == {"the black card", "the white card"}
    puzzle, m = _overlap(owner_line=4)
    assert check_puzzle(_fields(puzzle, m, "the black card"), "constraint_grid").answer == "the black card"
    puzzle, m = _overlap(owner_line=4, rule="later")
    with pytest.raises(PuzzleError, match="never says 'later'"):
        check_puzzle(_fields(puzzle, m, "the white card"), "constraint_grid")
    puzzle, m = _overlap(owner_line=5)
    with pytest.raises(PuzzleError, match="span.owner cites line 5"):
        check_puzzle(_fields(puzzle, m, "the black card"), "constraint_grid")


def test_a_day_no_span_holds():
    gap = ["1 March to 31 May", "1 July to 31 August", "1 September to 30 November"]
    puzzle = CARDS.replace("1 June to 31 August", "1 July to 31 August").replace("14 April", "14 June")
    m = copy.deepcopy(CARDS_MODEL)
    m["attributes"]["span"]["values"] = gap
    m["span"]["bounds"] = {s: s.split(" to ") for s in gap}
    m["span"]["days"] = _days(("1 March", 3, None), ("31 May", 5, None), ("14 June", 6, None), ("1 July", 7, None),
                              ("31 August", 8, None), ("1 September", 9, None), ("30 November", 11, None))
    m["asks"] = {"contains": "14 June"}
    with pytest.raises(PuzzleError, match="no span holds '14 June'"):
        check_puzzle(_fields(puzzle, m, "none"), "constraint_grid")
    # As a claim it is decided like any claim: no card held it in any placing, so unsupported. (With every fact idle,
    # puzzle-check refuses it as a puzzle; the verdict itself is the solver's.)
    cm = copy.deepcopy(m)
    del cm["asks"]
    cm["claim"] = ["contains", "the white card", "14 June"]
    t = Table(parse_model(cm))
    assert t.verdict(t.all) == "unsupported"


def test_a_claim_on_contains_is_keyed_by_the_computed_verdict():
    _, m = _ask("1 September")
    claim = CARDS.replace("Question: Which card was in force on 14 April?", "Question: Claim: the red card was in force on 1 September.")
    del m["asks"]
    m["claim"] = ["contains", "the red card", "1 September"]
    assert check_puzzle(_fields(claim, m, "supported"), "falsifier_first").answer == "supported"
    open_claim = claim.replace("3. The black card came into force before the red card.\n", "")
    om = copy.deepcopy(m)
    om["facts"] = om["facts"][:1]
    assert check_puzzle(_fields(open_claim, om, "cannot_tell"), "falsifier_first").answer == "cannot_tell"


def test_the_skeleton_records_the_span_contains_and_the_owner():
    _, owned = _overlap(owner_line=4)
    _, plain = _overlap()
    assert skeleton(owned) != skeleton(plain)
    assert '"contains": true' in skeleton(plain) and '"owner": "earlier"' in skeleton(owned)
    no_span = copy.deepcopy(CARDS_MODEL)
    del no_span["span"]
    no_span["asks"] = {"entity_with": {"span": SPANS[1]}}
    assert "|span:" not in skeleton(no_span) and skeleton(no_span) != skeleton(CARDS_MODEL)


@pytest.mark.parametrize("change, match", [
    (lambda s: "not a dict", "model.span is"),
    (lambda s: s.update(days=s["days"][:1]), "two or more"),
    (lambda s: s["days"][0].pop("ordinal"), "a day record is"),
    (lambda s: s["days"][0].update(day=31, month=2, label="31 February"), "is not a calendar date"),
    (lambda s: s["days"].append(dict(s["days"][-1], ordinal=99)), "is declared twice"),
    (lambda s: s["days"][1].update(ordinal=0), "ordinals rise strictly"),
    (lambda s: s["bounds"].pop("1 March to 31 May"), "for every value"),
    (lambda s: s["bounds"].update({"1 March to 31 May": ["1 March"]}), r"needs \[start day, end day\]"),
    (lambda s: s.update(owner={"rule": "first", "line": 2}), "span.owner is"),
])
def test_a_malformed_span_declaration_is_refused(change, match):
    m = copy.deepcopy(CARDS_MODEL)
    out = change(m["span"])
    if isinstance(out, str):
        m["span"] = out
    with pytest.raises(PuzzleError, match=match):
        parse_model(m)


def test_an_owner_rule_needs_two_spans_that_share_a_day():
    # R&D on #47: "the later card counts" over spans that never overlap was accepted and counted against nothing.
    puzzle = CARDS.replace("Question:", "4. A day two cards both list belongs to the later card.\nQuestion:")
    m = copy.deepcopy(CARDS_MODEL)
    m["span"]["owner"] = {"rule": "later", "line": 4}
    with pytest.raises(PuzzleError, match="no two declared spans share a day"):
        check_puzzle(_fields(puzzle, m, "the black card"), "constraint_grid")


def test_a_time_is_a_clock_time_and_a_bare_decimal_is_not():
    from role_set.puzzle import DATE_ENTITY

    for t in ("14:00", "2.40 pm", "9 am", "noon", "14 May", "May 14"):
        assert DATE_ENTITY.fullmatch(t), t
    for t in ("2.40", "the 9th edition", "issue 4", "Mayfield"):
        assert not DATE_ENTITY.fullmatch(t), t


def test_the_skeleton_records_whether_spans_overlap():
    _, over = _overlap()
    assert '"overlap": true' in skeleton(over) and '"overlap": false' in skeleton(CARDS_MODEL)
