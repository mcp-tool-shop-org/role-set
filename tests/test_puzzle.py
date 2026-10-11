"""Kind B's mechanical key (PR H): the model's shape, its tie to the text, the solver, and the key."""

import copy

import pytest

from role_set.puzzle import SEARCH_CAP, PuzzleError, check_puzzle, fact_lines, holds, parse_model, rule_names

GRID = """1. Three friends, Ana, Ben and Cy, each ordered one dish: soup, salad or pasta.
2. Ben ordered salad.
3. Cy didn't order pasta.
Question: Who ordered what?"""
GRID_MODEL = {"entities": ["Ana", "Ben", "Cy"], "attributes": {"dish": {"values": ["soup", "salad", "pasta"]}},
              "facts": [{"line": 2, "rule": ["is", "Ben", "dish", "salad"]}, {"line": 3, "rule": ["is_not", "Cy", "dish", "pasta"]}],
              "flavour": [1], "asks": {"attribute": "dish"}}
GRID_KEY = "Ana=pasta; Ben=salad; Cy=soup"


def _f(puzzle=GRID, model=None, answer=GRID_KEY, reply=None):
    return {"puzzle": puzzle, "model": copy.deepcopy(model or GRID_MODEL), "answer": answer,
            "reply": reply if reply is not None else f"1. Ben has salad.\n2. Cy isn't pasta, so soup.\nAnswer: {answer}"}


def _refused(match, structure="constraint_grid", **kw):
    with pytest.raises(PuzzleError, match=match):
        check_puzzle(_f(**kw), structure)


def test_a_closed_grid_passes_and_its_key_is_computed():
    got = check_puzzle(_f(), "constraint_grid")
    assert got.answer == GRID_KEY and got.solutions == 1 and not got.lookup
    # The key may list the assignments in any order and case.
    assert check_puzzle(_f(answer="cy=soup; ana=pasta; ben=salad"), "constraint_grid").answer == GRID_KEY


def test_two_solutions_are_refused_with_a_counterexample():
    m = copy.deepcopy(GRID_MODEL)
    m["facts"] = m["facts"][:1]
    _refused(r"more than one answer fits every fact, for example 'Ana=pasta; Ben=salad; Cy=soup' and 'Ana=soup",
             puzzle=GRID.replace("3. Cy didn't order pasta.\n", ""), model=m)


def test_a_wrong_key_and_a_reply_without_the_answer_line_are_refused():
    _refused("the facts give 'Ana=pasta; Ben=salad; Cy=soup', and the key says 'Ana=soup", answer="Ana=soup; Ben=salad; Cy=pasta",
             reply="x\nAnswer: Ana=soup; Ben=salad; Cy=pasta")
    _refused("the reply ends with the answer on its own line", reply="So Cy had soup and Ana had pasta.")


def test_contradictory_facts_are_refused():
    m = copy.deepcopy(GRID_MODEL)
    m["facts"].append({"line": 4, "rule": ["is", "Cy", "dish", "pasta"]})
    _refused("contradict each other", puzzle=GRID.replace("Question:", "4. Cy ordered pasta.\nQuestion:"), model=m)


def test_a_fact_that_changes_nothing_is_refused_naming_its_line():
    m = copy.deepcopy(GRID_MODEL)
    m["facts"].append({"line": 4, "rule": ["is_not", "Ana", "dish", "salad"]})
    _refused("line 4 changes nothing about the answer", puzzle=GRID.replace("Question:", "4. Ana didn't order salad.\nQuestion:"), model=m)


def test_a_lookup_is_flagged():
    puzzle = GRID.replace("3. Cy didn't order pasta.", "3. Cy ordered soup.")
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][1] = {"line": 3, "rule": ["is", "Cy", "dish", "soup"]}
    got = check_puzzle(_f(puzzle=puzzle, model=m), "constraint_grid")
    assert got.lookup


# ---- the tie to the text -------------------------------------------------------------------------------------------
def test_every_line_is_numbered_cited_once_and_names_what_its_rule_uses():
    _refused("every line before the Question line is numbered", puzzle="Three friends.\n" + GRID)
    _refused("numbered 1, 2, 3", puzzle=GRID.replace("3. Cy", "4. Cy"))
    _refused("ends with a line that begins 'Question:'", puzzle=GRID.split("Question:")[0])
    m = copy.deepcopy(GRID_MODEL)
    m["flavour"] = []
    _refused("line 1 has no rule in the model", model=m)
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][0]["line"] = 9
    _refused("a fact cites line 9", model=m)
    m = copy.deepcopy(GRID_MODEL)
    m["flavour"] = [1, 2]
    _refused("line 2 is both a fact and flavour", model=m)
    m = copy.deepcopy(GRID_MODEL)
    m["flavour"] = [7]
    _refused("flavour cites line 7", model=m)
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][0]["rule"] = ["is", "Ana", "dish", "salad"]
    _refused("line 2's rule uses 'Ana', which line 2 doesn't name", model=m)
    _refused("'pasta' is in the model but the puzzle never names it", puzzle=GRID.replace("soup, salad or pasta", "soup, salad or rice")
             .replace("didn't order pasta", "didn't order rice"))


def test_flavour_that_names_part_of_a_domain_reads_as_a_fact():
    puzzle = GRID.replace("Question:", "4. Ana loves soup.\nQuestion:")
    m = copy.deepcopy(GRID_MODEL)
    m["flavour"] = [1, 4]
    _refused("flavour line 4 names 'soup'", puzzle=puzzle, model=m)
    m["flavour"] = [1, 2, 3]
    with pytest.raises(PuzzleError, match="at most 2 flavour lines"):
        parse_model(m)


def test_polarity_must_match_the_line():
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][1]["rule"] = ["is", "Cy", "dish", "pasta"]
    _refused("line 3 is negative but its rule is positive", model=m)
    puzzle = GRID.replace("Cy didn't order pasta.", "Cy ordered pasta.")
    m = copy.deepcopy(GRID_MODEL)
    _refused("line 3 is positive but its rule is negative", puzzle=puzzle, model=m)


ROW = """1. Ana, Ben and Cy sit in seats 1 to 3, left to right.
2. Ana sits left of Ben.
3. Cy sits after Ben.
Question: Who sits in which seat?"""
ROW_MODEL = {"entities": ["Ana", "Ben", "Cy"], "attributes": {"seat": {"values": ["1", "2", "3"], "ordered": True}},
             "facts": [{"line": 2, "rule": ["before", "Ana", "Ben", "seat"]}, {"line": 3, "rule": ["after", "Cy", "Ben", "seat"]}],
             "flavour": [1], "asks": {"attribute": "seat"}}


def test_order_words_map_to_their_form_in_the_line_order():
    got = check_puzzle(_f(puzzle=ROW, model=ROW_MODEL, answer="Ana=1; Ben=2; Cy=3"), "constraint_grid")
    assert got.answer == "Ana=1; Ben=2; Cy=3"
    m = copy.deepcopy(ROW_MODEL)
    m["facts"][0]["rule"] = ["after", "Ben", "Ana", "seat"]
    _refused("line 2 says 'left of', so its rule is before", puzzle=ROW, model=m, answer="Ana=1; Ben=2; Cy=3")
    m = copy.deepcopy(ROW_MODEL)
    m["facts"][0]["rule"] = ["before", "Ben", "Ana", "seat"]
    _refused("opposite order to the line", puzzle=ROW, model=m, answer="Ana=1; Ben=2; Cy=3")
    m = copy.deepcopy(ROW_MODEL)
    m["attributes"]["seat"]["ordered"] = False
    _refused("is not ordered", puzzle=ROW, model=m)


def test_left_right_first_last_need_the_order_stated():
    _refused("states the direction once", puzzle=ROW.replace(", left to right", ""), model=ROW_MODEL, answer="Ana=1; Ben=2; Cy=3")
    puzzle = ROW.replace(", left to right", "").replace("sits left of", "sits before").replace("seats 1 to 3", "three seats 1, 2, 3") \
        .replace("Cy sits after Ben.", "Cy sits last, after Ben.")
    m = copy.deepcopy(ROW_MODEL)
    _refused("says first or last, so one line states the order", puzzle=puzzle, model=m, answer="Ana=1; Ben=2; Cy=3")


# ---- structures ----------------------------------------------------------------------------------------------------
CUPS = """1. Three cups stand in a row: Red, Blue, Gold.
2. Exactly one cup holds the pen; the rest are empty.
3. The pen is not in Red.
4. The pen is not in Blue.
Question: {q}"""
CUPS_MODEL = {"entities": ["Red", "Blue", "Gold"], "attributes": {"holds": {"values": ["pen", "empty"], "distinct": False}},
              "facts": [{"line": 2, "rule": ["count", "holds", "pen", "=", 1]}, {"line": 3, "rule": ["is_not", "Red", "holds", "pen"]},
                        {"line": 4, "rule": ["is_not", "Blue", "holds", "pen"]}],
              "flavour": [1], "asks": {"entity_with": {"holds": "pen"}}}


def test_elimination_with_a_count():
    assert check_puzzle(_f(puzzle=CUPS.format(q="Which cup holds the pen?"), model=CUPS_MODEL, answer="Gold"), "elimination_from_a_set").answer == "Gold"


def test_a_stopping_problem_is_keyed_by_the_computed_line():
    q = CUPS.format(q="After which line is it settled which cup holds the pen?")
    assert check_puzzle(_f(puzzle=q, model=CUPS_MODEL, answer="4", reply="x\nAnswer: after line 4"), "stopping_problem").stop_at == 4
    with pytest.raises(PuzzleError, match="settled after line 4, so the key is 4, not '3'"):
        check_puzzle(_f(puzzle=q, model=CUPS_MODEL, answer="3"), "stopping_problem")
    # Lines after the stop may follow; only those up to it must be needed.
    later = q.replace("Question:", "5. The pen is not in Red, again.\nQuestion:")
    m = copy.deepcopy(CUPS_MODEL)
    m["facts"].append({"line": 5, "rule": ["is_not", "Red", "holds", "pen"]})
    assert check_puzzle(_f(puzzle=later, model=m, answer="4", reply="x\nAnswer: 4"), "stopping_problem").stop_at == 4
    m = copy.deepcopy(CUPS_MODEL)
    m["facts"] = m["facts"][:2]
    with pytest.raises(PuzzleError, match="never settle"):
        check_puzzle(_f(puzzle=q.replace("4. The pen is not in Blue.\n", ""), model=m, answer="3"), "stopping_problem")


def test_an_underdetermined_set_is_computed_and_must_have_two_members():
    q = CUPS.format(q="Which cups could hold the pen?").replace("4. The pen is not in Blue.\n", "")
    m = copy.deepcopy(CUPS_MODEL)
    m["facts"] = m["facts"][:2]
    got = check_puzzle(_f(puzzle=q, model=m, answer="Gold, Blue", reply="x\nAnswer: Gold, Blue"), "underdetermined_with_explicit_set")
    assert got.answer == "Blue, Gold"
    with pytest.raises(PuzzleError, match="leave exactly 'Blue, Gold' possible"):
        check_puzzle(_f(puzzle=q, model=m, answer="Gold"), "underdetermined_with_explicit_set")
    with pytest.raises(PuzzleError, match="these facts settle it"):
        check_puzzle(_f(puzzle=CUPS.format(q="Which cups could hold the pen?"), model=CUPS_MODEL, answer="Gold"), "underdetermined_with_explicit_set")
    uq = CUPS.format(q="Is the cup that holds the pen determined?").replace("4. The pen is not in Blue.\n", "")
    assert check_puzzle(_f(puzzle=uq, model=m, answer="undetermined", reply="x\nAnswer: undetermined"), "underdetermined_with_explicit_set")
    with pytest.raises(PuzzleError, match="answers a question that asks whether the answer is unique"):
        check_puzzle(_f(puzzle=q, model=m, answer="undetermined", reply="x\nAnswer: undetermined"), "underdetermined_with_explicit_set")
    mm = copy.deepcopy(GRID_MODEL)
    with pytest.raises(PuzzleError, match="asks for one value"):
        check_puzzle(_f(model=mm), "underdetermined_with_explicit_set")


def test_a_claim_is_keyed_by_the_computed_verdict():
    q = CUPS.format(q="Is it true that Gold holds the pen?")
    m = copy.deepcopy(CUPS_MODEL)
    m.pop("asks")
    m["claim"] = ["is", "Gold", "holds", "pen"]
    assert check_puzzle(_f(puzzle=q, model=m, answer="supported", reply="x\nAnswer: supported"), "tree_with_pruning").answer == "supported"
    open_q = q.replace("4. The pen is not in Blue.\n", "")
    m2 = copy.deepcopy(m)
    m2["facts"] = m2["facts"][:2]
    assert check_puzzle(_f(puzzle=open_q, model=m2, answer="cannot_tell", reply="x\nAnswer: cannot_tell"), "falsifier_first").answer == "cannot_tell"
    with pytest.raises(PuzzleError, match="the Question line doesn't name"):
        check_puzzle(_f(puzzle=CUPS.format(q="Is the claim true?"), model=m, answer="supported"), "tree_with_pruning")
    with pytest.raises(PuzzleError, match="keyed by a verdict: give model.claim"):
        check_puzzle(_f(puzzle=q, model=CUPS_MODEL, answer="Gold"), "falsifier_first")
    m3 = copy.deepcopy(m)
    m3["facts"].append({"line": 5, "rule": ["is_not", "Red", "holds", "pen"]})
    with pytest.raises(PuzzleError, match="line 3 changes nothing"):  # a repeated fact makes each copy idle
        check_puzzle(_f(puzzle=q.replace("Question:", "5. The pen is not in Red, again.\nQuestion:"), model=m3, answer="supported"), "tree_with_pruning")
    m4 = copy.deepcopy(m)
    m4["facts"].append({"line": 5, "rule": ["is", "Red", "holds", "pen"]})
    with pytest.raises(PuzzleError, match="contradict"):
        check_puzzle(_f(puzzle=q.replace("Question:", "5. The pen is in Red.\nQuestion:"), model=m4, answer="supported"), "tree_with_pruning")


# ---- shape -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("change, match", [
    (lambda m: m.update(entities=["Ana"]), "2 to 6 distinct names"),
    (lambda m: m.update(attributes={}), "1 to 3 attributes"),
    (lambda m: m["attributes"].update(dish={"values": ["soup"]}), "2 or more distinct values"),
    (lambda m: m["attributes"].update(dish={"values": ["soup", "salad"]}), "needs exactly 3 values"),
    (lambda m: m.update(facts=[]), "one rule per fact line"),
    (lambda m: m.update(facts=[{"rule": ["is", "Ben", "dish", "salad"]}]), "each fact is"),
    (lambda m: m.update(flavour="1"), "list of line numbers"),
    (lambda m: m.update(asks=None), "model.asks says what the question asks"),
    (lambda m: m.update(asks={"which": 1}), "model.asks is"),
    (lambda m: m.update(asks={"entity_with": {"a": 1, "b": 2}}), "entity_with is"),
    (lambda m: m["facts"].append({"line": 2, "rule": "is"}), "a rule is a list"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["exactly", 1]}), "unknown rule form 'exactly'"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["is", "Ben"]}), "is takes 3 arguments"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["and", ["is", "Ben", "dish", "salad"]]}), "joins two or more rules"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["is", "Dee", "dish", "salad"]}), "'Dee' is not one of model.entities"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["is", "Ben", "drink", "salad"]}), "'drink' is not one of model.attributes"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["is", "Ben", "dish", "rice"]}), "'rice' is not a value"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["in", "Ben", "dish", []]}), "takes a list of values"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["is", {"dish": "soup", "x": 1}, "dish", "soup"]}), "an entity name or"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["count", "dish", "soup", "~", 1]}), "count is"),
    (lambda m: m["facts"].append({"line": 2, "rule": ["offset", "Ana", "Ben", "dish", 1]}), "is not ordered"),
])
def test_a_malformed_model_is_refused(change, match):
    m = copy.deepcopy(GRID_MODEL)
    change(m)
    with pytest.raises(PuzzleError, match=match):
        parse_model(m)
    with pytest.raises(PuzzleError, match="must be an object"):
        parse_model("model")


def test_offset_needs_a_whole_number_and_a_repeating_attribute_names_no_one():
    m = copy.deepcopy(ROW_MODEL)
    m["facts"].append({"line": 2, "rule": ["offset", "Ana", "Ben", "seat", "one"]})
    with pytest.raises(PuzzleError, match="whole number"):
        parse_model(m)
    m = copy.deepcopy(CUPS_MODEL)
    m["facts"].append({"line": 3, "rule": ["is", {"holds": "pen"}, "holds", "pen"]})
    with pytest.raises(PuzzleError, match="can't name one entity"):
        parse_model(m)


def test_every_form_evaluates():
    m = parse_model(copy.deepcopy(ROW_MODEL))
    sol = {"seat": ("1", "2", "3")}  # Ana 1, Ben 2, Cy 3
    cases = [(["in", "Ana", "seat", ["1", "3"]], True), (["not_in", "Ana", "seat", ["1"]], False), (["same", "Ana", {"seat": "1"}], True),
             (["different", "Ana", "Ben"], True), (["adjacent", "Ana", "Cy", "seat"], False), (["not_adjacent", "Ana", "Cy", "seat"], True),
             (["offset", "Ana", "Cy", "seat", 2], True), (["count", "seat", "1", ">=", 1], True), (["count", "seat", "1", "<=", 0], False),
             (["not", ["is", "Ana", "seat", "2"]], True), (["or", ["is", "Ana", "seat", "2"], ["is", "Ben", "seat", "2"]], True),
             (["if", ["is", "Ana", "seat", "1"], ["is", "Cy", "seat", "3"]], True), (["and", ["is", "Ana", "seat", "1"], ["is", "Cy", "seat", "1"]], False),
             (["after", "Ana", "Ben", "seat"], False)]
    for rule, want in cases:
        assert holds(m, sol, rule) is want, rule
    assert rule_names(["count", "seat", "1", "=", 1]) == ["1"]
    assert rule_names(["in", "Ana", "seat", ["1", "3"]]) == ["Ana", "1", "3"]


def test_a_puzzle_too_large_to_check_is_refused():
    names = ["A1", "B1", "C1", "D1", "E1", "F1"]
    m = {"entities": names, "attributes": {a: {"values": [f"{a}{i}" for i in range(6)]} for a in ("x", "y")},
         "facts": [{"line": 1, "rule": ["is", "A1", "x", "x0"]}], "flavour": [], "asks": {"attribute": "x"}}
    with pytest.raises(PuzzleError, match=f"keep it under {SEARCH_CAP:,}"):
        check_puzzle({"puzzle": "1. " + " ".join(names + [f"{a}{i}" for a in "xy" for i in range(6)]) + "\nQuestion: ?", "model": m,
                      "answer": "x", "reply": "Answer: x"}, "constraint_grid")


def test_fact_lines_reads_the_numbered_lines():
    lines, q = fact_lines(GRID)
    assert lines[2] == "Ben ordered salad." and q == "Who ordered what?"


def test_the_runner_refuses_a_bad_puzzle_and_caps_lookups_per_lesson(tmp_path):
    import json
    import types

    from role_set.plan import plan_lesson
    from test_runner import SIDES, _runner

    r, _, _ = _runner(tmp_path)
    b_slots = [s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "B" and s.structure in ("constraint_grid", "scheduling_with_exclusions")]
    lookup_puzzle = GRID.replace("3. Cy didn't order pasta.", "3. Cy ordered soup.")
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][1] = {"line": 3, "rule": ["is", "Cy", "dish", "soup"]}
    drafts = {"lookup": {"puzzle": lookup_puzzle, "model": m, "thinking": "1. Ben has salad.\n2. Cy has soup.\n3. Ana has pasta.",
                         "reply": "1. Ben has salad.\n2. Cy has soup.\n3. Ana has pasta.\nAnswer: " + GRID_KEY, "answer": GRID_KEY}}

    def writer(messages, seed=0):
        return types.SimpleNamespace(text=json.dumps(drafts["lookup"]), prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1)

    r.writer = writer
    first = r.run_slot(b_slots[0])
    second = r.run_slot(b_slots[1])
    assert first.accepted and r.lookups == {"T02": 1}
    assert not second.accepted and second.refusals[0]["code"] == "PUZZLE" and "a lookup" in second.refusals[0]["detail"]

    bad = dict(drafts["lookup"], answer="Ana=soup; Ben=salad; Cy=pasta")
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text=json.dumps(bad), prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1)
    out = r.run_slot(b_slots[1])
    assert out.refusals[0]["code"] == "PUZZLE" and "the facts give" in out.refusals[0]["detail"]
