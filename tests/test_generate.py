"""The kind-B generator, slice 1: generate-then-reduce, the shortcut registry, the T10 recipe and template wording
(docs/generator-design.md)."""

import functools
import json
import random
import re
from pathlib import Path

import pytest

from role_set.generate.build import CLUE_FORMS, BuildError, metrics, solver_version
from role_set.generate.recipes import PACKS, build_t10
from role_set.generate.render import CLAUSES, NUMBER_WORDS, TEMPLATES, render_puzzle, sentence
from role_set.generate.shortcuts import REGISTRY, answer, differs, prove, rewrite
from role_set.puzzle import AFTER_WORDS, BEFORE_WORDS, NEGATION, PuzzleError, Table, check_puzzle, check_text, parse_model

FIXTURES = Path(__file__).parent / "fixtures"
ROUND2 = json.loads((FIXTURES / "t10_round2_must_refuse.json").read_text(encoding="utf-8"))


def _rendered(b, pack, seed):
    return render_puzzle(b.model, PACKS[pack]["plural"], random.Random(seed))


# The slow-test split (R&D, 2026-10-18): a case that takes over ~10 s under coverage runs in the sweep job, the
# rest of the parametrization stays in the test job. Under coverage on a CI runner these cases passed the 240 s
# faulthandler limit, and its dump took down the xdist worker (#93, twice).
SLOW_OPEN_SET = {(1, "trains"), (2, "machines")}  # 35 s and 19 s under coverage, locally


@pytest.mark.parametrize("seed,pack", [pytest.param(seed, pack, marks=[pytest.mark.sweep] if (seed, pack) in SLOW_OPEN_SET else [])
                                       for seed in [1, 2] for pack in sorted(PACKS)])
def test_a_t10_open_set_puzzle_is_proven_and_its_wording_round_trips(pack, seed):
    b = _proven(pack, seed, "set")
    # The count is on the deciding path, and a size loosening changes the set (design pin: the trait step decides).
    assert b.trait_clue in [f["rule"] for f in b.model["facts"]] and b.trait_clue[0] == "count"
    assert any(p["differs"] for p in b.shortcuts.values())
    without = dict(b.model, facts=[f for f in b.model["facts"] if f["rule"] != b.trait_clue])
    assert answer(without, "underdetermined_with_explicit_set") != answer(b.model, "underdetermined_with_explicit_set")


@pytest.mark.parametrize("pack", [pytest.param(p, marks=pytest.mark.sweep) if p == "beds" else p for p in sorted(PACKS)])
def test_a_t10_claim_is_proven(pack):  # beds: 59 s under coverage, in the sweep (the split, above)
    b = _proven(pack, 1, "claim")
    text = _rendered(b, pack, 1)
    assert "Question: Claim:" in text
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
    # The cell chooses how many entities, so the explored space is the pack's values raised to that, not a fixed 16.
    vals = len(next(iter(b.model["attributes"].values()))["values"])
    assert m == b.metrics and m["assignments_explored"] == vals ** len(b.model["entities"])
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
    assert set(prove(model, "falsifier_first", [n for n in REGISTRY if n.startswith("T10.")])) == {"T10.exact_read_as_floor", "T10.exact_read_as_ceiling"}


def test_templates_agree_in_number_and_read_negative():
    rng = random.Random(0)
    one = sentence(["count", "state", "open", "=", 1], "doors", rng)
    assert "one" in one and "is" in one and "open" in one and "doors" in one
    two = sentence(["count", "state", "open", ">=", 2], "doors", rng)
    assert "two" in two and "are" in two and "open" in two
    denied = sentence(["not_in", "the oak door", "state", ["open", "shut"]], "doors", rng)
    assert any(w in denied for w in ("not", "never", "neither", "nor", "n't"))
    moved = sentence(["offset", "the oak door", "the pine door", "place", 1], "doors", rng)
    assert "before" not in moved and "after" not in moved and "oak" in moved and "pine" in moved
    t = Table(parse_model({"entities": ["a", "b"], "attributes": {"s": {"values": ["x", "y"], "distinct": False}},
                           "facts": [{"line": 2, "rule": ["is", "a", "s", "x"]}], "flavour": [1], "asks": {"attribute": "s", "of": "b"}}))
    assert len(t.rows) == 4


_FORMS = ("is", "is_not", "in", "not_in", "before", "after", "adjacent", "not_adjacent", "offset1", "offset2",
          "count=", "count>=", "count<=", "if")
_BASELINE = {"lamps", "cups", "doors"}
_X, _Y = "the oak door", "the pine door"


def _directions(form, line):
    before, after = BEFORE_WORDS.search(line), AFTER_WORDS.search(line)
    if form == "before":
        assert before and not after, line
    elif form == "after":
        assert after and not before, line
    else:
        assert not before and not after, line


def _round(rule, fact, ordered=False):
    """One fact line, a setting line that names the whole domain, and check_text."""

    if ordered:
        ents = ["the oak door", "the pine door", "the ash door"]
        spec = {"values": ["1st", "2nd", "3rd"], "ordered": True}
        setting = ("The oak door, the pine door and the ash door are the doors. "
                   "Each has its own place, one of 1st, 2nd and 3rd.")
        attr = "place"
    else:
        ents = ["the oak door", "the pine door"]
        spec = {"values": ["open", "shut"], "distinct": False}
        setting = "The oak door and the pine door are the doors. Each is open or shut."
        attr = "state"
    raw = {"entities": ents, "attributes": {attr: spec}, "facts": [{"line": 2, "rule": rule}], "flavour": [1],
           "asks": {"attribute": attr, "of": "the pine door"}}
    text = f"1. {setting}\n2. {fact}\nQuestion: What holds for the pine door?"
    check_text(parse_model(raw), text)


def test_every_form_has_three_templates_and_four_packs_beyond_the_baseline():
    assert set(TEMPLATES) == set(_FORMS)
    for form in _FORMS:
        assert len(TEMPLATES[form]) >= 3, form
    assert set(CLAUSES) == {"is", "is_not"}
    for form in ("is", "is_not"):
        assert len(CLAUSES[form]) >= 3, form
    assert _BASELINE <= set(PACKS)
    assert len(set(PACKS) - _BASELINE) >= 4


def test_pack_names_are_whole_words_no_other_name_contains():
    for key, pack in PACKS.items():
        assert {"plural", "names", "attribute", "values"} <= set(pack), key
        names = pack["names"]
        assert len(names) >= 4 and len(set(names)) == len(names)
        for a in names:
            for b in names:
                if a == b:
                    continue
                assert re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", b) is None, (key, a, b)


# What a sense block may hold. Every key is optional. counts holds only min and max_share.
_SENSE_KEYS = {"forms", "counts", "unsayable", "value_words", "question_value"}
_COUNT_KEYS = {"min", "max_share"}


def test_every_sense_block_has_only_the_sense_keys_known_forms_and_real_unsayable_pairs():
    for key, pack in PACKS.items():
        if key not in _BASELINE:
            assert "sense" in pack, key
        sense = pack.get("sense")
        if sense is None:
            continue
        assert isinstance(sense, dict) and sense and set(sense) <= _SENSE_KEYS, key
        if "forms" in sense:
            assert isinstance(sense["forms"], list) and sense["forms"], key
            assert set(sense["forms"]) <= set(CLUE_FORMS), (key, sense["forms"])
            # before, after, adjacent, not_adjacent and offset name an order.
            # These attributes are states, so a block does not list them.
            assert {"before", "after", "adjacent", "not_adjacent", "offset"}.isdisjoint(sense["forms"]), key
        if "counts" in sense:
            counts = sense["counts"]
            assert isinstance(counts, dict) and set(counts) <= _COUNT_KEYS, key
            if "min" in counts:
                assert counts["min"] >= 1, key
            if "max_share" in counts:
                assert 0 < counts["max_share"] <= 1, key
        if "unsayable" in sense:
            assert isinstance(sense["unsayable"], list), key
            for pair in sense["unsayable"]:
                assert set(pair) == {"entity", "value"}, (key, pair)
                assert pair["entity"] in pack["names"], (key, pair["entity"])
                assert pair["value"] in pack["values"], (key, pair["value"])
        if "value_words" in sense:
            for label, word in sense["value_words"].items():
                assert label in pack["values"], (key, label)
                assert isinstance(word, str) and word and word != label, (key, label)
        if "question_value" in sense:
            assert sense["question_value"] in pack["values"], key


def test_every_template_round_trips_through_check_text():
    for body in CLAUSES["is"]:
        assert NEGATION.search(body.format(x=_X, v="open")) is None, body
    for body in CLAUSES["is_not"]:
        assert NEGATION.search(body.format(x=_X, v="open")), body
    for tmpl in TEMPLATES["is"]:
        line = tmpl.format(X="The oak door", v="open")
        _directions("is", line)
        _round(["is", _X, "state", "open"], line)
    for tmpl in TEMPLATES["is_not"]:
        line = tmpl.format(X="The oak door", v="open")
        _directions("is_not", line)
        _round(["is_not", _X, "state", "open"], line)
    for tmpl in TEMPLATES["in"]:
        line = tmpl.format(X="The oak door", v1="open", v2="shut")
        _directions("in", line)
        _round(["in", _X, "state", ["open", "shut"]], line)
    for tmpl in TEMPLATES["not_in"]:
        line = tmpl.format(X="The oak door", v1="open", v2="shut")
        _directions("not_in", line)
        _round(["not_in", _X, "state", ["open", "shut"]], line)
    for form in ("before", "after", "adjacent", "not_adjacent"):
        for tmpl in TEMPLATES[form]:
            line = tmpl.format(X="The oak door", y=_Y)
            _directions(form, line)
            _round([form, _X, _Y, "place"], line, ordered=True)
    for k in (1, 2):
        for tmpl in TEMPLATES[f"offset{k}"]:
            line = tmpl.format(Y="The pine door", x=_X)
            _directions(f"offset{k}", line)
            _round(["offset", _X, _Y, "place", k], line, ordered=True)
    for op, key in (("=", "count="), (">=", "count>="), ("<=", "count<=")):
        for tmpl in TEMPLATES[key]:
            for n, be in ((1, "is"), (2, "are")):
                line = tmpl.format(k=NUMBER_WORDS[n], plural="doors", v="open", be=be)
                _directions(key, line)
                _round(["count", "state", "open", op, n], line)
    for frame in TEMPLATES["if"]:
        for p_form, p_bodies in CLAUSES.items():
            for q_form, q_bodies in CLAUSES.items():
                for p_body in p_bodies:
                    for q_body in q_bodies:
                        line = frame.format(p=p_body.format(x=_X, v="open"), q=q_body.format(x=_Y, v="shut"))
                        _directions("if", line)
                        _round(["if", [p_form, _X, "state", "open"], [q_form, _Y, "state", "shut"]], line)


@functools.lru_cache(maxsize=None)
def _proven(pack, seed, question):
    b = build_t10(pack, seed, question=question)
    text = _rendered(b, pack, seed)
    if question == "claim":
        structure, reply = "falsifier_first", "x\n" + b.answer
    else:
        structure, reply = "underdetermined_with_explicit_set", "x\nAnswer: " + b.answer
    got = check_puzzle({"puzzle": text, "model": b.model, "answer": b.answer, "reply": reply}, structure)
    assert got.answer == b.answer
    return b


def _cell_shape_cases():
    """One fixed seed for each cell shape in seeds 1-20.

    A shape is the value-count, the question, the entity count, the count word, and the if-target.
    The seed is the earliest one in pack order. It is not chosen for speed.
    """

    from role_set.generate.t10_variety import cell_for

    found = {}
    for pack in sorted(PACKS):
        kind = "two" if len(PACKS[pack]["values"]) == 2 else "three"
        for question in ("set", "claim"):
            for seed in range(1, 21):
                cell = cell_for(pack, question, seed)
                shape = (kind, question, cell["n"], cell["op"], cell["ifs"])
                found.setdefault(shape, (pack, seed, question))
    cases = []
    for (kind, question, n, op, ifs), (pack, seed, question) in sorted(found.items()):
        shape = f"{kind}-{question}-n{n}-{op}-ifs{ifs}"
        marks = [pytest.mark.sweep] if shape in SLOW_CELL_SHAPES else []  # the split, above
        cases.append(pytest.param(pack, seed, question, id=shape, marks=marks))
    return cases


# 59, 15, 13 and 11 s under coverage, locally: in the sweep.
SLOW_CELL_SHAPES = {"three-claim-n5-=-ifs2", "three-set-n5-<=-ifs1", "three-set-n5->=-ifs1", "three-set-n4-<=-ifs2"}


@pytest.mark.sweep
@pytest.mark.parametrize("pack", sorted(PACKS))
@pytest.mark.parametrize("seed", range(1, 21))
@pytest.mark.parametrize("question", ["set", "claim"])
def test_every_pack_and_seed_renders_a_proven_puzzle(pack, seed, question):
    from role_set.generate.t10_variety import canonical_skeleton, cell_for, clues_if_shape, record_index

    b = _proven(pack, seed, question)
    expected = record_index()[(pack, question, seed)]
    actual = {
        "pack": pack,
        "question": question,
        "seed": seed,
        "built": True,
        "cell": cell_for(pack, question, seed),
        "if_shape": clues_if_shape([f["rule"] for f in b.model["facts"]]),
        "tries": b.tries,
        "canonical_skeleton": canonical_skeleton(b.model),
    }
    assert actual == expected


@pytest.mark.parametrize("pack,seed,question", _cell_shape_cases())
def test_one_fixed_seed_covers_each_cell_shape(pack, seed, question):
    _proven(pack, seed, question)


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


@pytest.mark.parametrize("pack", ["lamps", "cups"])
@pytest.mark.parametrize(
    "shortcut,want_op",
    [
        ("T10.floor_read_as_exact", ">="),
        ("T10.ceiling_read_as_exact", "<="),
    ],
)
@pytest.mark.parametrize("ifs", [0, 1, 2], ids=["d2", "d3", "d4"])
def test_floor_and_ceiling_cells_build_for_lamps_and_cups_at_d2_to_d4(pack, shortcut, want_op, ifs):
    from role_set.generate.build import receipt
    from role_set.generate.select import differing
    from role_set.generate.t10_variety import load_cells
    from role_set.puzzle import solve

    matching = [c for c in load_cells()["cells"][pack]["set"] if c["op"] == want_op]
    seed, cell = next((i + 1, c) for i, c in enumerate(matching) if c["ifs"] == ifs)
    b = build_t10(pack, seed, question="set", cell=cell, target_shortcut=shortcut)
    assert b.trait_clue[0] == "count" and b.trait_clue[3] == want_op
    assert sum(1 for f in b.model["facts"] if f["rule"][0] == "if") == ifs
    solved = solve(parse_model(b.model), "underdetermined_with_explicit_set", "Which")
    assert solved.answer == b.answer and solved.solutions >= 2
    text = _rendered(b, pack, seed)
    checked = check_puzzle(
        {"puzzle": text, "model": b.model, "answer": b.answer, "reply": "x\nAnswer: " + b.answer},
        "underdetermined_with_explicit_set",
    )
    assert checked.answer == b.answer
    rec = receipt(b, "T10", PACKS[pack], "underdetermined_with_explicit_set")
    assert differing(rec) == [shortcut]
    b_from_seed = build_t10(pack, seed, question="set", target_shortcut=shortcut)
    assert b_from_seed.model == b.model and b_from_seed.answer == b.answer


@pytest.mark.parametrize("shortcut", ["T10.exact_read_as_floor", "T10.exact_read_as_ceiling"])
@pytest.mark.parametrize("question", ["set", "claim"])
def test_target_shortcut_selects_requested_exact_reading(shortcut, question):
    from role_set.generate.build import receipt
    from role_set.generate.select import differing

    structure = "falsifier_first" if question == "claim" else "underdetermined_with_explicit_set"
    b = build_t10("lamps", 1, question=question, target_shortcut=shortcut)
    rec = receipt(b, "T10", PACKS["lamps"], structure)
    assert shortcut in differing(rec)


def test_target_shortcut_refusals_and_default_identity():
    from role_set.generate.build import receipt

    with pytest.raises(BuildError, match="unknown shortcut"):
        build_t10("lamps", 1, question="set", target_shortcut="T11.consistent_so_supported")
    with pytest.raises(BuildError, match="does not match"):
        build_t10(
            "lamps",
            1,
            question="set",
            cell={"n": 3, "op": "=", "k": 1, "ifs": 1},
            target_shortcut="T10.floor_read_as_exact",
        )
    for shortcut, op in (("T10.floor_read_as_exact", ">="), ("T10.ceiling_read_as_exact", "<=")):
        with pytest.raises(BuildError, match="a claim tests exactly"):
            build_t10("lamps", 1, question="claim", target_shortcut=shortcut)
        with pytest.raises(BuildError, match="a claim tests exactly"):
            build_t10(
                "lamps",
                1,
                question="claim",
                cell={"n": 3, "op": op, "k": 1, "ifs": 1},
                target_shortcut=shortcut,
            )
    with pytest.raises(ValueError, match="T10 seed is 1 or more"):
        build_t10("lamps", 0, question="set", target_shortcut="T10.floor_read_as_exact")
    a = build_t10("lamps", 2, question="set")
    b = build_t10("lamps", 2, question="set", target_shortcut=None)
    assert receipt(a, "T10", PACKS["lamps"], "underdetermined_with_explicit_set") == receipt(
        b, "T10", PACKS["lamps"], "underdetermined_with_explicit_set"
    )

