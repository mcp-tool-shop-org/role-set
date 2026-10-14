"""T10 skeleton variety, and that each item tests its trait clue's own size word.

The floors are asserted on the locked seeds 61-80 and 81-100. Fresh locked seeds 81-100 are not built until the Phase 1
if_shape code and tables are frozen in a commit.

Distinct canonical skeletons at least 75 percent, no skeleton more than 3 times in a stratum, and no skeleton twice in one
pack's file. A zero-if cell stays within its visit cap (2 in every stratum). Every built puzzle has one count clue, and
that clue's own reading differs. The solver's own check is unchanged.
"""

import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from role_set.generate.build import MAX_TRIES, BuildError, Shape, assignments, check_model, deletion_order
from role_set.generate.recipes import PACKS, T10, build_t10
from role_set.generate.shortcuts import REGISTRY, prove
from role_set.generate.t10_variety import (
    IF_SHAPES,
    RECORD_HASH_FILES,
    RECORD_SEEDS,
    REGENERATE_RECORD_CMD,
    _max_placed,
    assign_cells,
    assignment,
    canonical_skeleton,
    capacity_report,
    cell_for,
    cell_id,
    cell_key,
    check_record_hash,
    clues_if_shape,
    load_capacity,
    load_cells,
    load_record,
    placed_builds,
    record_index,
    record_sha256,
    zero_if_cap,
)

FILES = (range(1, 21), range(61, 81), range(81, 101), range(101, 121))
LOCKED_FILES = (range(61, 81), range(81, 101), range(101, 121))
VERIFY = range(101, 121)


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


def test_claim_cells_are_exactly_With_if_shape_and_set_cells_keep_three_size_words():
    table = load_cells()
    for pack in table["packs"]:
        claim = table["cells"][pack]["claim"]
        setting = table["cells"][pack]["set"]
        assert len(claim) >= 20 and len(setting) >= 20
        assert {c["op"] for c in claim} == {"="}
        assert all(c.get("if_shape") in IF_SHAPES[c["ifs"]] for c in claim)
        assert {c["op"] for c in setting} == {"=", ">=", "<="}
        assert all("if_shape" not in c for c in setting)


def test_a_claim_refuses_an_at_least_cell():
    with pytest.raises(BuildError, match="exactly"):
        build_t10("cups", 1, "claim", cell={"n": 3, "op": ">=", "k": 1, "ifs": 0, "if_shape": "none"})


def test_a_seed_below_one_is_refused():
    with pytest.raises(ValueError, match="1 or more"):
        cell_for("lamps", "set", 0)


def test_three_value_claims_fit_at_a_cap_of_two():
    assert placed_builds("claim", 3, 2) == (120, 120)
    assert zero_if_cap("three", "claim") == 2
    assert zero_if_cap("two", "claim") == 2
    assert zero_if_cap("three", "set") == 2
    assert zero_if_cap("two", "set") == 2


def test_the_capacity_gate_reports_supply_demand_and_headroom():
    rows = {(row["question"], row["kind"]): row for row in capacity_report()}
    assert rows[("set", "two")]["ok"] and rows[("set", "two")]["headroom_any"] is None
    assert rows[("set", "three")]["supply"] == 48 and rows[("set", "three")]["demand"] == 0
    assert rows[("set", "three")]["headroom_shared"] == 42 and rows[("set", "three")]["headroom_any"] is None
    claim_two = rows[("claim", "two")]
    assert claim_two["ok"] and claim_two["supply"] == 18 and claim_two["demand"] == 0
    assert claim_two["headroom_shared"] is None and claim_two["headroom_any"] is None
    claim = rows[("claim", "three")]
    assert claim["ok"] and claim["zero_if_cap"] == 2
    assert claim["supply"] == 18 and claim["demand"] == 0 and claim["placed"] == 120
    assert claim["headroom_shared"] is None and claim["headroom_any"] is None
    by_pack = {pack["pack"]: pack for pack in claim["packs"]}
    assert by_pack == {
        "cups": {"pack": "cups", "if_cells": 43, "zero_if_demand": 0},
        "pans": {"pack": "pans", "if_cells": 43, "zero_if_demand": 0},
        "machines": {"pack": "machines", "if_cells": 47, "zero_if_demand": 0},
        "trains": {"pack": "trains", "if_cells": 43, "zero_if_demand": 0},
        "beds": {"pack": "beds", "if_cells": 46, "zero_if_demand": 0},
        "clinics": {"pack": "clinics", "if_cells": 43, "zero_if_demand": 0},
    }


def test_twelve_three_value_packs_allocate_with_no_within_file_repeats():
    table = load_cells()
    cap_data = load_capacity().get("cells", {})
    real_packs = ["cups", "pans", "machines", "trains", "beds", "clinics"]
    packs_12 = real_packs + [f"{name}_clone" for name in real_packs]
    for question in ("set", "claim"):
        cells_of = {
            pack: table["cells"][pack.removesuffix("_clone")][question]
            for pack in packs_12
        }
        cap = zero_if_cap("three", question)
        assert _max_placed(packs_12, cells_of, cap) == 20 * len(packs_12)
        q_cap = cap_data.get(question, {}).get("three", {})
        weights = {
            cell_id(cell): max(1, int(q_cap[cell_key(cell)]["distinct"]))
            for pack in real_packs
            for cell in cells_of[pack]
            if cell_key(cell) in q_cap and q_cap[cell_key(cell)].get("builds")
        }
        plan = assign_cells(packs_12, cells_of, cap, weights, file_index=0)
        zero_visits: Counter = Counter()
        for pack in packs_12:
            chosen = plan[pack]
            assert len(chosen) == 20
            assert len({cell_id(cell) for cell in chosen}) == 20
            for cell in chosen:
                if cell["ifs"] == 0:
                    zero_visits[cell_id(cell)] += 1
        assert all(count <= cap for count in zero_visits.values())


def test_a_short_claim_stratum_fails_and_names_the_next_step():
    table = json.loads(json.dumps(load_cells()))
    three_packs = ("cups", "pans", "machines", "trains", "beds", "clinics")
    for pack in three_packs:
        zeros = [c for c in table["cells"][pack]["claim"] if c["ifs"] == 0]
        ifs = [c for c in table["cells"][pack]["claim"] if c["ifs"] != 0][:16]
        table["cells"][pack]["claim"] = zeros + ifs
    counts = {pack: len(PACKS[pack]["values"]) for pack in PACKS}
    rows = {(row["question"], row["kind"]): row for row in capacity_report(table, counts)}
    claim = rows[("claim", "three")]
    assert claim["ok"] is False and claim["placed"] < claim["needed"]
    assert claim["supply"] == 18 and claim["demand"] == 24
    assert "polarity_mode or in/not_in" in claim["reason"]
    assert "18" in claim["reason"] and "24" in claim["reason"]
    assert rows[("set", "three")]["ok"] and rows[("claim", "two")]["ok"] and rows[("set", "two")]["ok"]


def test_one_file_stays_inside_the_zero_if_cap():
    reported = {(row["question"], row["kind"]): row for row in capacity_report()}
    for question in ("set", "claim"):
        for kind in ("two", "three"):
            visits: Counter = Counter()
            packs = [name for name in PACKS if _kind(name) == kind]
            for pack in packs:
                for seed in range(1, 21):
                    cell = cell_for(pack, question, seed)
                    if cell["ifs"] == 0:
                        visits[cell_id(cell)] += 1
            assert sum(visits.values()) == reported[(question, kind)]["demand"]
            assert all(count <= zero_if_cap(kind, question) for count in visits.values())


def test_the_next_file_rotates_the_assignment():
    for question, nvalues in (("set", 2), ("set", 3), ("claim", 2), ("claim", 3)):
        assert assignment(question, nvalues, 0) != assignment(question, nvalues, 1)
    assert cell_for("clinics", "claim", 1) != cell_for("clinics", "claim", 21)


def test_visits_prefer_cells_that_yield_more_skeletons():
    cells = [{"n": 3, "op": "=", "k": i, "ifs": 1} for i in range(1, 7)]
    packs = ["a", "b"]
    weights = {cell_id(cell): (8 if cell["k"] >= 5 else 1) for cell in cells}
    plan = assign_cells(packs, {pack: cells for pack in packs}, 2, weights, 0, per_pack=4)
    for pack in packs:
        chosen = [cell["k"] for cell in plan[pack]]
        assert len(chosen) == 4 and len(set(chosen)) == 4
        assert {5, 6} <= set(chosen)
    fat = sum(1 for pack in packs for cell in plan[pack] if cell["k"] >= 5)
    assert fat == 4


def test_an_impossible_file_is_refused_before_a_build():
    cell = {"n": 3, "op": "=", "k": 1, "ifs": 0}
    with pytest.raises(ValueError, match="zero-if cap"):
        assign_cells(["a"], {"a": [cell]}, 1, {}, 0, per_pack=2)


def test_capacity_weights_come_from_the_tuning_seeds():
    data = load_capacity()
    assert data["seeds"] == [1, 21, 41]
    assert set(data["packs"]) == set(PACKS)
    for question in ("set", "claim"):
        for kind in ("two", "three"):
            assert data["cells"][question][kind]
            for row in data["cells"][question][kind].values():
                assert row["distinct"] <= row["builds"]
                assert row["builds"] + row["failed"] > 0


def test_canonical_skeleton_renames_entities_and_values():
    model = {
        "entities": ["the red cup", "the blue cup"],
        "attributes": {"heat": {"values": ["hot", "warm", "cold"]}},
        "facts": [
            {"rule": ["is", "the red cup", "heat", "hot"]},
            {"rule": ["count", "heat", "hot", "=", 1]},
            {"rule": ["if", ["is", "the red cup", "heat", "hot"], ["is_not", "the blue cup", "heat", "hot"]]},
        ],
    }
    text = canonical_skeleton(model)
    assert "cup" not in text and "hot" not in text
    assert "e1" in text and "v1" in text


def test_the_usable_cell_script_checks_the_committed_table():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    done = subprocess.run(
        [sys.executable, str(root / "scripts" / "t10_usable_cells.py"), "--check"],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "claim three: supply 18 demand 0" in done.stdout
    assert "clinics: 43 if cells, zero-if demand 0" in done.stdout


def _kind(pack: str) -> str:
    return "two" if len(PACKS[pack]["values"]) == 2 else "three"


def _hidden_count(model: dict, hidden: int, trait: list) -> int:
    sols = assignments(Shape(tuple(model["entities"]), model["attributes"]))
    return sum(1 for value in sols[hidden][trait[1]] if value == trait[2])


def _rows(question: str, kind: str, seeds: range = VERIFY):
    rec = record_index(check_record_hash())
    rows = []
    for pack in PACKS:
        if _kind(pack) != kind:
            continue
        for seed in seeds:
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
            rules = [fact["rule"] for fact in built.model["facts"]]
            counts = [rule for rule in rules if rule[0] == "count"]
            assert counts == [built.trait_clue]
            if question == "claim":
                assert op == "="
            shape_got = clues_if_shape(rules)
            if "if_shape" in cell:
                assert shape_got == cell["if_shape"]
            ifs = sum(1 for rule in rules if rule[0] == "if")
            assert ifs == cell["ifs"]
            assert built.trait_clue[3:] == [cell["op"], cell["k"]]
            assert len(built.model["entities"]) == cell["n"]
            skel = canonical_skeleton(built.model)
            assert rec[(pack, question, seed)] == {
                "pack": pack,
                "question": question,
                "seed": seed,
                "built": True,
                "cell": cell,
                "if_shape": shape_got,
                "tries": built.tries,
                "canonical_skeleton": skel,
            }
            hidden = _hidden_count(built.model, built.hidden, built.trait_clue)
            rows.append({"pack": pack, "question": question, "kind": _kind(pack), "seed": seed,
                         "skeleton": skel, "op": op, "cell": cell,
                         "n": cell["n"], "k": cell["k"], "ifs": ifs, "if_shape": shape_got,
                         "refusals": built.tries - 1,
                         "trait_differs": trait_differs,
                         "on_boundary": op != "=" and hidden == cell["k"]})
    return rows


def _report(rows: list[dict], question: str, kind: str) -> str:
    """The figures for one stratum. The floors are asserted separately; this print is the record."""

    skins = [r["skeleton"] for r in rows]
    top_skeleton, top = Counter(skins).most_common(1)[0]
    top_row = next(r for r in rows if r["skeleton"] == top_skeleton)
    top_cell = cell_id(top_row["cell"])
    zero_visits = Counter((r["n"], r["op"], r["k"]) for r in rows if r["ifs"] == 0)
    if zero_visits:
        busiest, busiest_n = zero_visits.most_common(1)[0]
        zero_note = f"{sum(zero_visits.values())} max {busiest_n} at {busiest}"
    else:
        zero_note = "0"
    within: dict[str, list[str]] = {}
    for row in rows:
        within.setdefault(row["pack"], []).append(row["skeleton"])
    extras = {pack: sum(n - 1 for n in Counter(skins_p).values()) for pack, skins_p in within.items()}
    ineq = [r for r in rows if r["op"] != "="]
    missed = sum(not r["trait_differs"] for r in rows)
    line = (f"{question} {kind}: {len(set(skins))}/{len(skins)} top {top} cell {top_cell} "
            f"zero-if {zero_note} within {extras} "
            f"ifs {sum(r['ifs'] >= 1 for r in rows)}/{len(rows)} "
            f"ops {dict(Counter(r['op'] for r in rows))} n {dict(Counter(r['n'] for r in rows))} "
            f"boundary {sum(r['on_boundary'] for r in ineq)}/{len(ineq)} "
            f"trait_not_tested {missed} "
            f"refusals {sum(r['refusals'] for r in rows)}")
    print(line, flush=True)
    return line


def _floors(rows: list[dict]) -> None:
    skins = [r["skeleton"] for r in rows]
    assert skins and len(set(skins)) / len(skins) >= 0.75
    assert Counter(skins).most_common(1)[0][1] <= 3
    by_pack: dict[str, list[str]] = {}
    for row in rows:
        by_pack.setdefault(row["pack"], []).append(row["skeleton"])
    for skins_p in by_pack.values():
        assert sum(n - 1 for n in Counter(skins_p).values()) == 0


def _zero_if_cap(rows: list[dict], cap: int) -> None:
    """No zero-if cell is visited more than its cap inside one file of the locked seeds."""

    visits: Counter = Counter()
    for row in rows:
        if row["ifs"] == 0:
            visits[((row["seed"] - 1) // 20, row["n"], row["op"], row["k"])] += 1
    over = {cell: count for cell, count in visits.items() if count > cap}
    assert not over


def _record_rows(record: dict, question: str, kind: str, seeds: range) -> list[dict]:
    seed_set = set(seeds)
    rows = []
    for entry in record["builds"]:
        if entry["question"] != question or _kind(entry["pack"]) != kind or entry["seed"] not in seed_set:
            continue
        assert entry["built"] is True
        cell = entry["cell"]
        rows.append({
            "pack": entry["pack"],
            "question": question,
            "kind": kind,
            "seed": entry["seed"],
            "skeleton": entry["canonical_skeleton"],
            "op": cell["op"],
            "n": cell["n"],
            "k": cell["k"],
            "ifs": cell["ifs"],
            "if_shape": entry["if_shape"],
        })
    return rows


def test_committed_record_hash_and_proven_entries():
    record = check_record_hash(load_record())
    assert record["regenerate"] == REGENERATE_RECORD_CMD
    assert record["files"] == list(RECORD_HASH_FILES)
    assert record["sha256"] == record_sha256()
    expected_keys = [(pack, question, seed) for pack in PACKS for question in ("set", "claim") for seed in RECORD_SEEDS]
    builds = record["builds"]
    assert [(row["pack"], row["question"], row["seed"]) for row in builds] == expected_keys
    assert list(record_index(record)) == expected_keys
    for entry in builds:
        assert entry["built"] is True
        cell = cell_for(entry["pack"], entry["question"], entry["seed"])
        assert entry["cell"] == cell
        assert isinstance(entry["tries"], int) and 1 <= entry["tries"] <= MAX_TRIES
        skel = json.loads(entry["canonical_skeleton"])
        assert skel["n"] == cell["n"]
        counts = [clue for clue in skel["clues"] if clue[0] == "count"]
        assert len(counts) == 1 and counts[0][3:] == [cell["op"], cell["k"]]
        assert sum(1 for clue in skel["clues"] if clue[0] == "if") == cell["ifs"]
        assert entry["if_shape"] == clues_if_shape(skel["clues"])
        if entry["question"] == "claim":
            assert cell["op"] == "="
            assert cell["if_shape"] == entry["if_shape"]


def test_a_stale_record_hash_fails_and_names_the_regenerate_command():
    stale = dict(load_record(), sha256="0" * 64)
    with pytest.raises(ValueError, match=REGENERATE_RECORD_CMD):
        check_record_hash(stale)
    bad_files = dict(load_record(), files=["src/role_set/puzzle.py"])
    with pytest.raises(ValueError, match=REGENERATE_RECORD_CMD):
        check_record_hash(bad_files)


def test_the_record_hash_list_is_closed_under_role_set_imports():
    import ast

    root = Path(__file__).resolve().parents[1]
    hashed = set(RECORD_HASH_FILES)
    for rel in RECORD_HASH_FILES:
        if not rel.endswith(".py"):
            continue
        tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [alias.name for alias in node.names if alias.name.startswith("role_set")]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("role_set"):
                mods = [node.module]
                for alias in node.names:
                    cand = f"{node.module}.{alias.name}"
                    if (root / "src" / (*cand.split("."),)).with_suffix(".py").exists():
                        mods.append(cand)
            for mod in mods:
                parts = mod.split(".")
                py_rel = "/".join(["src", *parts]) + ".py"
                init_rel = "/".join(["src", *parts, "__init__.py"])
                target = py_rel if (root / py_rel).exists() else init_rel
                assert target in hashed, f"{rel} imports {mod} ({target}), which is missing from RECORD_HASH_FILES"


@pytest.mark.parametrize(
    ("question", "kind", "expected_total"),
    [
        ("set", "two", 40),
        ("set", "three", 120),
        ("claim", "two", 40),
        ("claim", "three", 120),
    ],
    ids=["set-two", "set-three", "claim-two", "claim-three"],
)
def test_committed_record_clears_the_variety_floors_in_every_stratum(
    question, kind, expected_total
):
    record = check_record_hash(load_record())
    for seed_range in FILES:
        rows = _record_rows(record, question, kind, seed_range)
        assert len(rows) == expected_total
        _floors(rows)
        _zero_if_cap(rows, zero_if_cap(kind, question))


def test_the_variety_record_script_checks_the_committed_record():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    done = subprocess.run(
        [sys.executable, str(root / "scripts" / "t10_variety_record.py"), "--check"],
        cwd=root, env=env, capture_output=True, text=True, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "set two: 40/40 top 1" in done.stdout
    assert "set three: 120/120 top 1" in done.stdout


@pytest.mark.sweep
@pytest.mark.parametrize("seed_range", LOCKED_FILES, ids=["61-80", "81-100", "101-120"])
@pytest.mark.parametrize(
    ("question", "kind"),
    [("set", "two"), ("set", "three"), ("claim", "two"), ("claim", "three")],
    ids=["set-two", "set-three", "claim-two", "claim-three"],
)
def test_locked_seeds_clear_the_variety_floors_in_every_stratum(question, kind, seed_range, request):
    rows = _rows(question, kind, seed_range)
    request.node.user_properties.append(("stratum", _report(rows, question, kind)))
    _floors(rows)
    _zero_if_cap(rows, zero_if_cap(kind, question))
