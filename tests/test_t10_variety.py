"""T10 skeleton variety, and that each item tests its trait clue's own size word.

The floors are asserted on the locked seeds 41-60. Tuning used 1-40, which the previous variety fix had already seen.
This range is not built until the Phase 1 code is committed.

Distinct canonical skeletons at least 75 percent, no skeleton more than 3 times in a stratum, and no skeleton twice in one
pack's file. Every built puzzle has one count clue, and that clue's own reading differs. The solver's own check is unchanged.
"""

import json
import random
from collections import Counter

import pytest

from role_set.generate.build import BuildError, Shape, assignments, check_model, deletion_order
from role_set.generate.recipes import PACKS, T10, build_t10
from role_set.generate.shortcuts import REGISTRY, prove
from role_set.generate.t10_variety import canonical_skeleton, cell_for, load_cells

# Locked seeds. Not built until the Phase 1 code was committed. Tuning is 1-40.
VERIFY = range(41, 61)


def test_deletion_tries_a_unit_fact_before_an_if_and_the_count_last():
    count = ["count", "s", "lit", "=", 1]
    unit = ["is", "a", "s", "lit"]
    iff = ["if", ["is", "a", "s", "lit"], ["is", "b", "s", "dark"]]
    rng_order = deletion_order([iff, unit, count], random.Random(0), keep=count)
    assert [c[0] for c in rng_order] == ["is", "if", "count"]


def test_twenty_seeds_in_one_file_take_twenty_cells():
    for pack in PACKS:
        for question in ("set", "claim"):
            got = [json.dumps(cell_for(pack, question, seed), sort_keys=True) for seed in range(1, 21)]
            assert len(set(got)) == 20, (pack, question)


def test_claim_cells_are_exactly_and_set_cells_keep_three_size_words():
    table = load_cells()
    for pack in table["packs"]:
        claim = table["cells"][pack]["claim"]
        setting = table["cells"][pack]["set"]
        assert len(claim) >= 20 and len(setting) >= 20
        assert {c["op"] for c in claim} == {"="}
        assert {c["op"] for c in setting} == {"=", ">=", "<="}


def test_a_claim_refuses_an_at_least_cell():
    with pytest.raises(BuildError, match="exactly"):
        build_t10("cups", 1, "claim", cell={"n": 3, "op": ">=", "k": 1, "ifs": 0})


def _kind(pack: str) -> str:
    return "two" if len(PACKS[pack]["values"]) == 2 else "three"


def _hidden_count(model: dict, hidden: int, trait: list) -> int:
    sols = assignments(Shape(tuple(model["entities"]), model["attributes"]))
    return sum(1 for value in sols[hidden][trait[1]] if value == trait[2])


def _rows():
    rows = []
    for pack in PACKS:
        for question in ("set", "claim"):
            for seed in VERIFY:
                cell = cell_for(pack, question, seed)
                built = build_t10(pack, seed, question)
                structure = "falsifier_first" if question == "claim" else "underdetermined_with_explicit_set"
                assert check_model(built.model, structure) == built.answer
                proofs = prove(built.model, structure, list(T10.shortcuts))
                assert any(p["differs"] for p in proofs.values())
                op = built.trait_clue[3]
                own = [name for name in T10.shortcuts if REGISTRY[name].get("from") == op]
                trait_differs = any(proofs.get(name, {}).get("differs") for name in own)
                assert trait_differs, (pack, question, seed, op)
                counts = [fact["rule"] for fact in built.model["facts"] if fact["rule"][0] == "count"]
                assert counts == [built.trait_clue]
                if question == "claim":
                    assert op == "="
                ifs = sum(1 for fact in built.model["facts"] if fact["rule"][0] == "if")
                assert ifs == cell["ifs"]
                assert built.trait_clue[3:] == [cell["op"], cell["k"]]
                assert len(built.model["entities"]) == cell["n"]
                hidden = _hidden_count(built.model, built.hidden, built.trait_clue)
                rows.append({"pack": pack, "question": question, "kind": _kind(pack), "seed": seed,
                             "skeleton": canonical_skeleton(built.model), "op": op,
                             "n": cell["n"], "ifs": ifs, "refusals": built.tries - 1,
                             "trait_differs": trait_differs,
                             "on_boundary": op != "=" and hidden == cell["k"]})
    return rows


def _report(rows: list[dict]) -> None:
    """The figures section 5 asks for. The floors are asserted separately; this print is the record."""

    for question in ("set", "claim"):
        for kind in ("two", "three", None):
            subset = [r for r in rows if r["question"] == question and (kind is None or r["kind"] == kind)]
            skins = [r["skeleton"] for r in subset]
            top = Counter(skins).most_common(1)[0][1]
            within = {}
            for row in subset:
                within.setdefault(row["pack"], []).append(row["skeleton"])
            extras = {pack: sum(n - 1 for n in Counter(skins_p).values()) for pack, skins_p in within.items()}
            ineq = [r for r in subset if r["op"] != "="]
            missed = sum(not r["trait_differs"] for r in subset)
            print(f"{question} {kind or 'all'}: {len(set(skins))}/{len(skins)} top {top} within {extras} "
                  f"ifs {sum(r['ifs'] >= 1 for r in subset)}/{len(subset)} "
                  f"ops {dict(Counter(r['op'] for r in subset))} n {dict(Counter(r['n'] for r in subset))} "
                  f"boundary {sum(r['on_boundary'] for r in ineq)}/{len(ineq)} "
                  f"trait_not_tested {missed} "
                  f"refusals {sum(r['refusals'] for r in subset)}")


def _floors(rows: list[dict]) -> None:
    skins = [r["skeleton"] for r in rows]
    assert skins and len(set(skins)) / len(skins) >= 0.75
    assert Counter(skins).most_common(1)[0][1] <= 3
    by_pack: dict[str, list[str]] = {}
    for row in rows:
        by_pack.setdefault(row["pack"], []).append(row["skeleton"])
    for skins_p in by_pack.values():
        assert sum(n - 1 for n in Counter(skins_p).values()) == 0


def test_locked_seeds_clear_the_variety_floors_in_every_stratum():
    rows = _rows()
    _report(rows)
    for question in ("set", "claim"):
        for kind in ("two", "three"):
            _floors([r for r in rows if r["question"] == question and r["kind"] == kind])
