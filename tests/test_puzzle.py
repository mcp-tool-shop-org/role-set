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


TIMES = """1. Three ferries, Ash, Birch and Cove, leave at 08:15, 12:15 and 16:15, one each.
2. Ash leaves at 08:15.
3. Cove does not leave at 12:15.
Question: Which ferry leaves at which time?"""
TIMES_MODEL = {"entities": ["Ash", "Birch", "Cove"], "attributes": {"time": {"values": ["08:15", "12:15", "16:15"], "ordered": True}},
               "facts": [{"line": 2, "rule": ["is", "Ash", "time", "08:15"]}, {"line": 3, "rule": ["is_not", "Cove", "time", "12:15"]}],
               "flavour": [1], "asks": {"attribute": "time"}}
TIMES_KEY = "Ash=08:15; Birch=12:15; Cove=16:15"


def test_a_bare_key_with_colons_in_its_values_ends_the_reply():
    # Only a leading "Answer:" label is stripped, so a time-valued key passes bare or labelled (R&D on aspire-si#99).
    for last in (TIMES_KEY, f"Answer: {TIMES_KEY}", f"answer:{TIMES_KEY}"):
        assert check_puzzle(_f(puzzle=TIMES, model=TIMES_MODEL, answer=TIMES_KEY, reply=f"x\n{last}"), "constraint_grid").answer == TIMES_KEY
    _refused("the reply ends with the answer", puzzle=TIMES, model=TIMES_MODEL, answer=TIMES_KEY,
             reply="x\nAsh=08:15; Birch=16:15; Cove=12:15")


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


def test_an_underdetermined_set_of_groups_keeps_which_names_go_together():
    # aspire-si#106 round 3: "Which sets of cups could be hot?" flattened to "red, blue, red, green", and a token set
    # compare would have passed "red, blue, green" too.
    cups = ("1. The red cup, the blue cup and the green cup are each hot or cold.\n2. Exactly two of the three cups are hot.\n"
            "3. The red cup is hot.\nQuestion: Which sets of cups could be hot?")
    m = {"entities": ["red", "blue", "green"], "attributes": {"heat": {"values": ["hot", "cold"], "distinct": False}},
         "facts": [{"line": 2, "rule": ["count", "heat", "hot", "=", 2]}, {"line": 3, "rule": ["is", "red", "heat", "hot"]}],
         "flavour": [1], "asks": {"entity_with": {"heat": "hot"}}}
    key = "red and blue; red and green"
    got = check_puzzle(_f(puzzle=cups, model=m, answer="green and red; blue and red", reply="x\nAnswer: green and red; blue and red"),
                       "underdetermined_with_explicit_set")
    assert got.answer == key
    for wrong in ("red, blue, red, green", "red, blue, green", "red and blue", "red and blue; red and teal"):
        with pytest.raises(PuzzleError, match="write each possible group with 'and'"):
            check_puzzle(_f(puzzle=cups, model=m, answer=wrong, reply=f"x\nAnswer: {wrong}"), "underdetermined_with_explicit_set")
    # R&D on #45: the form follows the question. One-name groups stay groups ("elm; oak; yew"), never the single
    # group of three, and a name with "and" in it stays whole.
    one = cups.replace("Exactly two", "Exactly one").replace("3. The red cup is hot.", "3. The red cup is cold.")
    m1 = copy.deepcopy(m)
    m1["facts"] = [{"line": 2, "rule": ["count", "heat", "hot", "=", 1]}, {"line": 3, "rule": ["is", "red", "heat", "cold"]}]
    assert check_puzzle(_f(puzzle=one, model=m1, answer="green; blue", reply="x\nAnswer: green; blue"),
                        "underdetermined_with_explicit_set").answer == "blue; green"
    with pytest.raises(PuzzleError, match="write each possible group with 'and'"):
        check_puzzle(_f(puzzle=one, model=m1, answer="blue, green", reply="x\nAnswer: blue, green"), "underdetermined_with_explicit_set")
    sp = one.replace("the green cup", "the salt and pepper cup")
    m2 = copy.deepcopy(m1)
    m2["entities"] = ["red", "blue", "salt and pepper"]
    assert check_puzzle(_f(puzzle=sp, model=m2, answer="salt and pepper; blue", reply="x\nAnswer: salt and pepper; blue"),
                        "underdetermined_with_explicit_set").answer == "blue; salt and pepper"


def test_an_underdetermined_set_is_computed_and_must_have_two_members():
    q = CUPS.format(q="Which cups could hold the pen?").replace("4. The pen is not in Blue.\n", "")
    m = copy.deepcopy(CUPS_MODEL)
    m["facts"] = m["facts"][:2]
    # "holds" may repeat, so each possible answer is a group, here of one cup (R&D on role-set#45).
    got = check_puzzle(_f(puzzle=q, model=m, answer="Gold; Blue", reply="x\nAnswer: Gold; Blue"), "underdetermined_with_explicit_set")
    assert got.answer == "Blue; Gold"
    with pytest.raises(PuzzleError, match="leave exactly 'Blue; Gold' possible"):
        check_puzzle(_f(puzzle=q, model=m, answer="Gold"), "underdetermined_with_explicit_set")
    with pytest.raises(PuzzleError, match="these facts settle it"):
        check_puzzle(_f(puzzle=CUPS.format(q="Which cups could hold the pen?"), model=CUPS_MODEL, answer="Gold"), "underdetermined_with_explicit_set")
    uq = CUPS.format(q="Is the cup that holds the pen determined?").replace("4. The pen is not in Blue.\n", "")
    assert check_puzzle(_f(puzzle=uq, model=m, answer="undetermined", reply="x\nAnswer: undetermined"), "underdetermined_with_explicit_set")
    with pytest.raises(PuzzleError, match="answers a question that asks whether the answer is unique"):
        check_puzzle(_f(puzzle=q, model=m, answer="undetermined", reply="x\nAnswer: undetermined"), "underdetermined_with_explicit_set")
    # One value per answer (asks.of on a distinct attribute) keeps the comma list.
    om = copy.deepcopy(GRID_MODEL)
    om["facts"] = om["facts"][:1]
    om["asks"] = {"attribute": "dish", "of": "Cy"}
    oq = GRID.replace("3. Cy didn't order pasta.\n", "").replace("Who ordered what?", "What could Cy have ordered?")
    assert check_puzzle(_f(puzzle=oq, model=om, answer="soup, pasta", reply="x\nAnswer: soup, pasta"),
                        "underdetermined_with_explicit_set").answer == "pasta, soup"
    with pytest.raises(PuzzleError, match="leave exactly 'pasta, soup' possible, and the key says 'soup'"):
        check_puzzle(_f(puzzle=oq, model=om, answer="soup", reply="x\nAnswer: soup"), "underdetermined_with_explicit_set")
    # R&D on aspire-si#106 round 4: a key that lists every possibility answers a question about what could be.
    with pytest.raises(PuzzleError, match="the Question asks what could be"):
        check_puzzle(_f(puzzle=oq.replace("What could Cy have ordered?", "What did Cy order?"), model=om, answer="soup, pasta",
                        reply="x\nAnswer: soup, pasta"), "underdetermined_with_explicit_set")
    # R&D on #46: a name or value never counts as the possibility word ("the can", "14 May"), and "may" counts in lower case.
    from role_set.puzzle import asks_possibility, parse_model
    cans = parse_model({"entities": ["the can", "the jar"], "attributes": {"day": {"values": ["14 May", "15 May"]}},
                        "facts": [{"line": 2, "rule": ["is", "the can", "day", "14 May"]}], "flavour": [1],
                        "asks": {"attribute": "day", "of": "the jar"}})
    assert not asks_possibility(cans, "Which day is the can on?")
    assert not asks_possibility(cans, "What was in force on 14 May?")
    assert not asks_possibility(cans, "May the jar be first?")
    assert asks_possibility(cans, "Which days may the can be on?") and asks_possibility(cans, "Which days can the jar take?")
    for word in ("can", "might", "possible"):
        q2 = oq.replace("What could Cy have ordered?", {"can": "Which dishes can Cy have?", "might": "What might Cy have ordered?",
                                                         "possible": "Which dishes are possible for Cy?"}[word])
        assert check_puzzle(_f(puzzle=q2, model=om, answer="soup, pasta", reply="x\nAnswer: soup, pasta"),
                            "underdetermined_with_explicit_set").answer == "pasta, soup"
    mm = copy.deepcopy(GRID_MODEL)
    with pytest.raises(PuzzleError, match="asks for one value"):
        check_puzzle(_f(model=mm), "underdetermined_with_explicit_set")


def test_the_question_names_the_entity_the_model_asks_about():
    # aspire-si#106, T12-B-01..05: the model asked the fee of an edition, and the text asked the fee on a date.
    m = copy.deepcopy(GRID_MODEL)
    m["asks"] = {"attribute": "dish", "of": "Cy"}
    asked = GRID.replace("Who ordered what?", "What did Cy order?")
    assert check_puzzle(_f(puzzle=asked, model=m, answer="soup"), "constraint_grid").answer == "soup"
    _refused("the model asks about 'Cy', and the Question line doesn't name it",
             puzzle=GRID.replace("Who ordered what?", "What did the third friend order?"), model=m, answer="soup")


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
    # The gates saw the accepted puzzle's own model (R&D, 2026-10-15: check_against signs puzzles by their model).
    accepted_id = r.accepted[-1]["id"]
    assert r.models == {accepted_id: m} and r.gates.module.models_seen[0] == {accepted_id: m}
    assert not second.accepted and second.refusals[0]["code"] == "PUZZLE" and "a lookup" in second.refusals[0]["detail"]

    bad = dict(drafts["lookup"], answer="Ana=soup; Ben=salad; Cy=pasta")
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text=json.dumps(bad), prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1)
    out = r.run_slot(b_slots[1])
    assert out.refusals[0]["code"] == "PUZZLE" and "the facts give" in out.refusals[0]["detail"]


def test_right_to_left_is_refused_because_it_reverses_left_and_right():
    _refused("right to left reverses every left/right fact", puzzle=ROW.replace("left to right", "right to left"), model=ROW_MODEL,
             answer="Ana=1; Ben=2; Cy=3")


def test_in_with_one_value_and_same_with_a_value_count_as_direct_facts():
    for rule, line in ((["in", "Cy", "dish", ["soup"]], "3. Cy ordered one of: soup."),
                       (["same", "Cy", {"dish": "soup"}], "3. Cy is the one who ordered soup.")):
        m = copy.deepcopy(GRID_MODEL)
        m["facts"][1] = {"line": 3, "rule": rule}
        assert check_puzzle(_f(puzzle=GRID.replace("3. Cy didn't order pasta.", line), model=m), "constraint_grid").lookup, rule


def test_a_fact_that_narrows_a_decided_claim_without_moving_it_is_idle():
    # Gold holds the pen whatever Red holds: with the claim decided, "not in Red" only narrows the solutions.
    q = """1. Three cups stand in a row: Red, Blue, Gold.
2. Gold holds the pen.
3. Blue is empty.
4. Red is empty.
Question: Is it true that Gold holds the pen?"""
    m = {"entities": ["Red", "Blue", "Gold"], "attributes": {"holds": {"values": ["pen", "empty"], "distinct": False}},
         "facts": [{"line": 2, "rule": ["is", "Gold", "holds", "pen"]}, {"line": 3, "rule": ["is", "Blue", "holds", "empty"]},
                   {"line": 4, "rule": ["is", "Red", "holds", "empty"]}],
         "flavour": [1], "claim": ["is", "Gold", "holds", "pen"]}
    with pytest.raises(PuzzleError, match="line 3 changes nothing"):
        check_puzzle(_f(puzzle=q, model=m, answer="supported", reply="x\nAnswer: supported"), "tree_with_pruning")


def test_the_lookup_cap_is_read_again_under_the_accept_lock(tmp_path):
    import json
    import types

    from role_set.plan import plan_lesson
    from test_runner import SIDES, _runner

    r, _, _ = _runner(tmp_path)
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "B" and s.structure == "constraint_grid")
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][1] = {"line": 3, "rule": ["is", "Cy", "dish", "soup"]}
    draft = {"puzzle": GRID.replace("3. Cy didn't order pasta.", "3. Cy ordered soup."), "model": m,
             "thinking": "1. Ben has salad.\n2. Cy has soup.\n3. Ana has pasta.",
             "reply": "1. Ben has salad.\n2. Cy has soup.\n3. Ana has pasta.\nAnswer: " + GRID_KEY, "answer": GRID_KEY}
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text=json.dumps(draft), prompt_tokens=1, completion_tokens=1, model="m", seconds=0.1)
    real = r.gates.module.check_against

    def meanwhile(row, accepted, models=None):
        r.lookups["T02"] = 1  # another B slot of the lesson was accepted after this one's first read
        return real(row, accepted, models=models)

    r.gates.module.check_against = meanwhile
    out = r.run_slot(slot)
    assert not out.accepted and out.refusals[0]["code"] == "PUZZLE" and "a lookup" in out.refusals[0]["detail"]


def test_a_b_trace_that_trips_the_stuck_rule_is_refused(tmp_path):
    from role_set.plan import plan_lesson
    from test_runner import SIDES, _runner

    r, _, _ = _runner(tmp_path, stuck=lambda thinking, material, tier, tokens: {"stuck": True, "bursts": 2, "hedges": 4})
    slot = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "B")
    out = r.run_slot(slot)
    assert not out.accepted and {x["code"] for x in out.refusals} == {"STUCK"} and "bursts 2, hedges 4" in out.refusals[0]["detail"]


def test_every_text_problem_comes_in_one_numbered_refusal():
    m = copy.deepcopy(GRID_MODEL)
    m["facts"][0]["rule"] = ["is", "Ana", "dish", "salad"]
    m["flavour"] = [1, 9]
    with pytest.raises(PuzzleError) as e:
        check_puzzle(_f(puzzle="Three friends order food.\n" + GRID.replace("Question:", "Q:"), model=m), "constraint_grid")
    msg = str(e.value)
    assert msg.startswith("1) a puzzle ends with a line that begins 'Question:'") and "2) every line before the Question line is numbered" in msg
    assert "flavour cites line 9" in msg and "line 2's rule uses 'Ana'" in msg


def test_a_puzzle_that_repeats_the_briefs_instruction_is_refused():
    leaked = GRID.replace("Question: Who ordered what?", "Question: Who ordered what? Show your thinking as a constraint grid.")
    _refused("repeats the brief's instruction", puzzle=leaked)


def test_the_briefs_worked_example_is_a_model_the_code_accepts():
    from role_set.prompts import B_EXAMPLE_KEY, B_EXAMPLE_MODEL, B_EXAMPLE_PUZZLE, B_MODEL_EXAMPLE, hard_rules
    from role_set.plan import plan_lesson
    from test_runner import SIDES

    got = check_puzzle({"puzzle": B_EXAMPLE_PUZZLE, "model": copy.deepcopy(B_EXAMPLE_MODEL), "answer": B_EXAMPLE_KEY,
                        "reply": "x\nAnswer: " + B_EXAMPLE_KEY}, "constraint_grid")
    assert got.answer == B_EXAMPLE_KEY and not got.lookup
    b = next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "B")
    assert B_MODEL_EXAMPLE in hard_rules(b) and "elimination" in B_MODEL_EXAMPLE


def test_halts_key_assembly_and_puzzle_refusals_by_their_rule():
    from role_set.runner import refusal_rule

    a = refusal_rule({"code": "ASSEMBLY", "detail": "the answer names 'Zed', which the puzzle doesn't; use the puzzle's own names"})
    b = refusal_rule({"code": "ASSEMBLY", "detail": "the answer names 'Ann', which the puzzle doesn't; use the puzzle's own names"})
    c = refusal_rule({"code": "ASSEMBLY", "detail": "reply writes its own VERDICT, DECIDING or ESCALATE line"})
    assert a == b != c and a.startswith("ASSEMBLY: ")
    p1 = refusal_rule({"code": "PUZZLE", "detail": "1) flavour cites line 3, and the puzzle has no line 3; 2) x"})
    p2 = refusal_rule({"code": "PUZZLE", "detail": "flavour cites line 4, and the puzzle has no line 4"})
    assert p1 == p2 == "PUZZLE: flavour cites line N, and the puzzle has no line N"


# ---- fix round 2's shapes (R&D on aspire-si #95) -----------------------------------------------------------------
PAIRS = ["Ana and Bo", "Ana and Cal", "Ana and Dee", "Bo and Cal", "Bo and Dee", "Cal and Dee"]


def _pairs(line2, count):
    puzzle = ("1. Six pairs are " + ", ".join(PAIRS[:-1]) + ", and " + PAIRS[-1] + ". A pair is yes or no.\n2. " + line2 +
              "\nQuestion: Which pairs may sit in front?")
    model = {"entities": PAIRS, "attributes": {"seat": {"values": ["yes", "no"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["count", "seat", "yes", "=", count]}], "flavour": [1], "asks": {"entity_with": {"seat": "yes"}}}
    return puzzle, model


def test_t05_b01_a_count_written_as_model_syntax_or_settling_every_entity_is_refused():
    puzzle, model = _pairs("The count of yes is 6.", 6)
    key = ", ".join(PAIRS)
    with pytest.raises(PuzzleError, match="reads as model syntax"):
        check_puzzle(_f(puzzle=puzzle, model=model, answer=key, reply="x\nAnswer: " + key), "elimination_from_a_set")
    puzzle, model = _pairs("Every pair says yes.", 6)
    with pytest.raises(PuzzleError, match="settles every entity at once"):
        check_puzzle(_f(puzzle=puzzle, model=model, answer=key, reply="x\nAnswer: " + key), "elimination_from_a_set")


def test_t06_b01_a_stopping_line_that_is_a_count_of_everyone_is_refused():
    pegs = [f"peg {c}" for c in "ABCDEF"]
    puzzle = ("1. Six pegs are " + ", ".join(pegs) + ". A peg may or may not take the cord.\n2. Every peg may take the cord.\n"
              "Question: After which line is it settled which pegs may take the cord?")
    model = {"entities": pegs, "attributes": {"take": {"values": ["may", "may not"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["count", "take", "may", "=", 6]}], "flavour": [1], "asks": {"entity_with": {"take": "may"}}}
    with pytest.raises(PuzzleError, match="settles every entity at once"):
        check_puzzle(_f(puzzle=puzzle, model=model, answer="2", reply="x\nAnswer: 2"), "stopping_problem")


def test_t03_b01_is_not_on_a_two_value_attribute_is_a_lookup():
    puzzle = ("1. Eight bins stand in a row. The third bin and the catalogue use the words tin and container.\n"
              "2. The third bin is not marked container.\nQuestion: Which word does the row paint on the third bin?")
    model = {"entities": ["the third bin", "the catalogue"], "attributes": {"word": {"values": ["tin", "container"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["is_not", "the third bin", "word", "container"]}], "flavour": [1],
             "asks": {"attribute": "word", "of": "the third bin"}}
    assert check_puzzle(_f(puzzle=puzzle, model=model, answer="tin", reply="x\nAnswer: tin"), "elimination_from_a_set").lookup


def test_t13_b07_an_if_on_its_own_stated_antecedent_is_a_lookup():
    puzzle = ("1. Two markers are the seal and the red crate. A marker sits at bay 1 or bay 2.\n"
              "2. If the seal is at bay 2, then the red crate is at bay 2.\n3. The seal is at bay 2.\nQuestion: Which bay sends the red crate?")
    model = {"entities": ["the seal", "the red crate"], "attributes": {"bay": {"values": ["bay 1", "bay 2"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["if", ["is", "the seal", "bay", "bay 2"], ["is", "the red crate", "bay", "bay 2"]]},
                       {"line": 3, "rule": ["is", "the seal", "bay", "bay 2"]}], "flavour": [1], "asks": {"attribute": "bay", "of": "the red crate"}}
    assert check_puzzle(_f(puzzle=puzzle, model=model, answer="bay 2", reply="x\nAnswer: bay 2"), "scheduling_with_exclusions").lookup


def test_any_single_fact_that_settles_the_answer_makes_a_lookup_and_a_claim_can_be_one():
    from role_set.puzzle import parse_model, solve

    m = parse_model({"entities": ["Ana", "Ben", "Cy"], "attributes": {"dish": {"values": ["soup", "salad", "pasta"]}},
                     "facts": [{"line": 2, "rule": ["not_in", "Ben", "dish", ["soup", "pasta"]]}],
                     "flavour": [1], "asks": {"attribute": "dish", "of": "Ben"}})
    assert solve(m, "constraint_grid", "").lookup is True  # line 2 alone fixes Ben's dish
    claim = parse_model({"entities": ["the white boat", "the grey boat"], "attributes": {"time": {"values": ["dusk", "dawn"]}},
                         "facts": [{"line": 2, "rule": ["is", "the white boat", "time", "dusk"]}], "flavour": [1],
                         "claim": ["is", "the grey boat", "time", "dawn"]})
    assert solve(claim, "tree_with_pruning", "").lookup is True


def test_a_bare_number_as_a_predicate_reads_as_model_syntax():
    puzzle = "1. Two lines are the choir line and the margin. A count is 1 or 2.\n2. The choir line is 1.\nQuestion: What count does the choir line give?"
    model = {"entities": ["the choir line", "the margin"], "attributes": {"count": {"values": ["1", "2"], "distinct": False}},
             "facts": [{"line": 2, "rule": ["is", "the choir line", "count", "1"]}], "flavour": [1], "asks": {"attribute": "count", "of": "the choir line"}}
    with pytest.raises(PuzzleError, match="reads as model syntax"):
        check_puzzle(_f(puzzle=puzzle, model=model, answer="1", reply="x\nAnswer: 1"), "elimination_from_a_set")


def test_t03_b06_an_if_on_a_stated_before_is_a_lookup():
    from role_set.puzzle import parse_model, solve

    # The entities were "4pm" and "6pm"; a clock time is no longer an entity (#47), so the sessions carry names.
    m = parse_model({"entities": ["the early session", "the late session"],
                     "attributes": {"slot": {"values": ["slot 1", "slot 2"], "ordered": True},
                                    "rate": {"values": ["2 pounds an hour", "unlisted"]}},
                     "facts": [{"line": 2, "rule": ["if", ["before", "the early session", "the late session", "slot"],
                                                    ["is", "the early session", "rate", "2 pounds an hour"]]},
                               {"line": 3, "rule": ["before", "the early session", "the late session", "slot"]}],
                     "flavour": [1], "asks": {"attribute": "rate", "of": "the early session"}})
    assert solve(m, "scheduling_with_exclusions", "").lookup is True
