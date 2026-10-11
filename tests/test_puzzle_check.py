"""`role-set puzzle-check`: reviewed data files plus their models sidecars."""

import json
from pathlib import Path

from role_set.cli import main
from role_set.puzzle_check import check_file, models_path, row_sha256

GRID = """1. Three friends, Ana, Ben and Cy, each ordered one dish: soup, salad or pasta.
2. Ben ordered salad.
3. Cy didn't order pasta.
Question: Who ordered what?"""
MODEL = {"entities": ["Ana", "Ben", "Cy"], "attributes": {"dish": {"values": ["soup", "salad", "pasta"]}},
         "facts": [{"line": 2, "rule": ["is", "Ben", "dish", "salad"]}, {"line": 3, "rule": ["is_not", "Cy", "dish", "pasta"]}],
         "flavour": [1], "asks": {"attribute": "dish"}}
KEY = "Ana=pasta; Ben=salad; Cy=soup"


def _row(rid, puzzle=GRID, answer=KEY, reply=None, lesson="T02"):
    return {"id": rid, "kind": "B", "lesson": lesson, "answer": answer,
            "turns": [{"user": puzzle, "thinking": "1. Ben has salad.", "reply": reply or f"1. Ben has salad.\n{answer}"}]}


def _write(tmp: Path, rows, side, stale=()) -> Path:
    """Sidecar lines get the row's sha256 unless they carry one, or their id is in `stale` (hashed before an edit)."""

    data = tmp / "batch9-t02.jsonl"
    lines = {r["id"]: json.dumps(r) for r in rows}
    data.write_text("".join(x + "\n" for x in lines.values()), encoding="utf-8")
    for m in side:
        if m["id"] in lines and "row_sha256" not in m:
            m["row_sha256"] = row_sha256(lines[m["id"]] + (" " if m["id"] in stale else ""))
    models_path(data).write_text("".join(json.dumps(m) + "\n" for m in side), encoding="utf-8")
    return data


def test_a_modelled_puzzle_passes_and_a_verdict_item_is_out_of_scope(tmp_path: Path):
    verdict = _row("T02-B-06", puzzle="Material: The desk opens at 9.\nStatement: The desk opens at 9.", answer="supported",
                   reply="The line says so.\nVERDICT: supported\nDECIDING: The desk opens at 9.\nESCALATE: no")
    other = {"id": "T02-A-01", "kind": "A", "turns": [{"user": "x", "reply": "y"}]}
    data = _write(tmp_path, [_row("T02-B-01"), verdict, other],
                  [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL}, {"id": "T02-B-06", "structure": "verdict"}])
    r = check_file(data)
    assert r["ok"] and (r["b_items"], r["pass"], r["verdict"], r["refused"], r["missing"]) == (2, 1, 1, 0, 0)
    assert models_path(data).name == "batch9-t02.models.jsonl"


def test_refusals_missing_lines_stray_ids_and_a_second_lookup(tmp_path: Path):
    lookup_puzzle = GRID.replace("3. Cy didn't order pasta.", "3. Cy ordered soup.")
    lookup_model = json.loads(json.dumps(MODEL))
    lookup_model["facts"][1] = {"line": 3, "rule": ["is", "Cy", "dish", "soup"]}
    rows = [_row("T02-B-01", answer="Ana=soup; Ben=salad; Cy=pasta"), _row("T02-B-02"), _row("T02-B-03", puzzle=lookup_puzzle),
            _row("T02-B-04", puzzle=lookup_puzzle), _row("T02-B-05", reply="no verdict here", answer="supported")]
    side = [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL},
            {"id": "T02-B-03", "structure": "constraint_grid", "model": lookup_model},
            {"id": "T02-B-04", "structure": "constraint_grid", "model": lookup_model},
            {"id": "T02-B-05", "structure": "verdict"}, {"id": "T02-B-99", "structure": "verdict"}]
    r = check_file(_write(tmp_path, rows, side))
    by = {i["id"]: i for i in r["items"]}
    assert not r["ok"] and r["stray"] == ["T02-B-99"]
    assert by["T02-B-01"]["result"] == "refused" and "the facts give" in by["T02-B-01"]["detail"]
    assert by["T02-B-02"]["result"] == "missing"
    assert by["T02-B-03"]["result"] == "pass" and by["T02-B-04"]["result"] == "refused" and "second lookup" in by["T02-B-04"]["detail"]
    assert by["T02-B-05"]["result"] == "refused" and "no VERDICT line" in by["T02-B-05"]["detail"]


def test_the_cli_reports_and_exits_on_the_result(tmp_path: Path, capsys):
    data = _write(tmp_path, [_row("T02-B-01")], [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL}])
    assert main(["puzzle-check", str(data)]) == 0
    out = capsys.readouterr().out
    assert "pass 1, verdict, outside solver scope: 0, refused 0, missing 0" in out and "by structure: constraint_grid: pass 1" in out
    assert main(["puzzle-check", str(data), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["ok"] is True
    models_path(data).unlink()
    assert main(["puzzle-check", str(data)]) == 1
    assert "MISSING" in capsys.readouterr().out
    odd = tmp_path / "plain.txt"
    odd.write_text("", encoding="utf-8")
    assert models_path(odd).name == "plain.models.jsonl"


def test_a_row_changed_after_its_model_was_written_is_refused(tmp_path: Path):
    data = _write(tmp_path, [_row("T02-B-01")], [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL}], stale={"T02-B-01"})
    r = check_file(data)
    assert not r["ok"] and r["items"][0]["result"] == "refused" and "row_sha256" in r["items"][0]["detail"]
    assert r["by_structure"] == {"constraint_grid: refused": 1}


def test_stamp_writes_the_hashes_then_a_later_edit_shows_up(tmp_path: Path, capsys):
    data = _write(tmp_path, [_row("T02-B-01")], [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL, "row_sha256": "old"}])
    assert main(["puzzle-check", str(data)]) == 1
    assert main(["puzzle-check", str(data), "--stamp"]) == 0
    assert "stamped 1 sidecar line(s)" in capsys.readouterr().out
    data.write_text(data.read_text(encoding="utf-8").replace("Ben has salad", "Ben has the salad"), encoding="utf-8")
    assert main(["puzzle-check", str(data)]) == 1


def test_structure_must_be_a_menu_name_and_verdict_needs_a_verdict_key(tmp_path: Path):
    from role_set.puzzle_check import STRUCTURES

    value_keyed = _row("T02-B-02", reply="1. Ben has salad.\nVERDICT: supported\nDECIDING: NONE\nESCALATE: no")
    data = _write(tmp_path, [_row("T02-B-01"), value_keyed],
                  [{"id": "T02-B-01", "structure": "constraint-grid", "model": MODEL}, {"id": "T02-B-02", "structure": "verdict"}])
    by = {i["id"]: i for i in check_file(data)["items"]}
    assert by["T02-B-01"]["result"] == "refused" and "is not one of constraint_grid" in by["T02-B-01"]["detail"]
    assert by["T02-B-02"]["result"] == "refused" and "keyed by a value" in by["T02-B-02"]["detail"]
    assert len(STRUCTURES) == 9 and STRUCTURES[-1] == "verdict"


# The same skeleton as GRID ("X is A. Y is not B." over three values), renamed and with its two facts in the other order.
BOATS = """1. Three boats, Oak, Elm and Ash, each take one berth: north, east or west.
2. Ash doesn't take west.
3. Elm takes east.
Question: Which boat takes which berth?"""
BOATS_MODEL = {"entities": ["Oak", "Elm", "Ash"], "attributes": {"berth": {"values": ["north", "east", "west"]}},
               "facts": [{"line": 2, "rule": ["is_not", "Ash", "berth", "west"]}, {"line": 3, "rule": ["is", "Elm", "berth", "east"]}],
               "flavour": [1], "asks": {"attribute": "berth"}}
BOATS_KEY = "Oak=west; Elm=east; Ash=north"


def test_a_skeleton_repeated_in_another_file_is_refused_naming_the_first(tmp_path: Path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _write(tmp_path / "a", [_row("T02-B-01")], [{"id": "T02-B-01", "structure": "constraint_grid", "model": MODEL}])
    second = _write(tmp_path / "b", [_row("T05-B-03", puzzle=BOATS, answer=BOATS_KEY, lesson="T05")],
                    [{"id": "T05-B-03", "structure": "constraint_grid", "model": BOATS_MODEL}])
    assert check_file(first)["ok"] and check_file(second)["ok"]  # each file alone passes
    assert main(["puzzle-check", str(first), str(second)]) == 1
    out = capsys.readouterr().out
    assert "T05-B-03: REFUSED: repeats the fact skeleton of T02-B-01 (batch9-t02.jsonl)" in out
    assert "pass 1, verdict, outside solver scope: 0, refused 0" in out  # the first stays a pass


def test_skeletons_ignore_names_order_and_direction_but_keep_attribute_shape():
    from role_set.puzzle_check import skeleton

    assert skeleton(MODEL) == skeleton(BOATS_MODEL)
    chain = {"entities": ["A", "B", "C"], "attributes": {"seat": {"values": ["1", "2", "3"], "ordered": True}},
             "facts": [{"line": 2, "rule": ["before", "A", "B", "seat"]}, {"line": 3, "rule": ["before", "B", "C", "seat"]}]}
    mirrored = {"entities": ["P", "Q", "R"], "attributes": {"place": {"values": ["x", "y", "z"], "ordered": True}},
                "facts": [{"line": 2, "rule": ["after", "R", "Q", "place"]}, {"line": 3, "rule": ["after", "Q", "P", "place"]}]}
    assert skeleton(chain) == skeleton(mirrored)
    two_values = {"entities": ["Ana", "Ben"], "attributes": {"dish": {"values": ["soup", "salad"]}},
                  "facts": [{"line": 2, "rule": ["is", "Ben", "dish", "salad"]}, {"line": 3, "rule": ["is_not", "Ana", "dish", "salad"]}]}
    assert skeleton(two_values) != skeleton(MODEL)
    nested = {"entities": ["A", "B", "C"], "attributes": {"s": {"values": ["1", "2", "3"], "ordered": True}},
              "facts": [{"line": 2, "rule": ["if", ["is", "A", "s", "1"], ["not", ["in", {"s": "2"}, "s", ["2", "3"]]]]},
                        {"line": 3, "rule": ["offset", "A", "B", "s", 1]}, {"line": 4, "rule": ["count", "s", "1", "=", 1]},
                        {"line": 5, "rule": ["same", "C", {"s": "3"}]}, {"line": 6, "rule": ["not_adjacent", "A", "C", "s"]}]}
    assert skeleton(nested).startswith("[")


def test_a_symmetric_clue_in_either_order_and_a_negative_offset_are_one_skeleton():
    from role_set.puzzle_check import skeleton

    def m(*rules):
        return {"entities": ["X", "Y", "Z"], "attributes": {"h": {"values": ["1st", "2nd", "3rd"], "ordered": True}},
                "facts": [{"line": i + 2, "rule": r} for i, r in enumerate(rules)]}

    # R&D on role-set#37: swapping the names in one adjacent clue let a copy through.
    assert skeleton(m(["is_not", "X", "h", "3rd"], ["adjacent", "Z", "X", "h"])) == skeleton(m(["is_not", "X", "h", "3rd"], ["adjacent", "X", "Z", "h"]))
    assert skeleton(m(["if", ["is", "Y", "h", "1st"], ["different", "X", "Z"]], ["is_not", "X", "h", "1st"])) == \
        skeleton(m(["is_not", "Y", "h", "1st"], ["if", ["is", "X", "h", "1st"], ["different", "Z", "Y"]]))
    assert skeleton(m(["offset", "X", "Y", "h", -1])) == skeleton(m(["offset", "Y", "X", "h", 1]))
    # R&D on aspire-si#106: on a two-value attribute, "is not lit" and "is dark" are one clue. Three values keep them apart.
    def binary(attr, values, *rules):
        return {"entities": ["P", "Q", "R"], "attributes": {attr: {"values": values}},
                "facts": [{"line": i + 2, "rule": r} for i, r in enumerate(rules)]}

    lamps = binary("state", ["lit", "dark"], ["count", "state", "lit", ">=", 2], ["is_not", "P", "state", "lit"])
    mugs = binary("fill", ["full", "empty"], ["count", "fill", "full", ">=", 2], ["is", "R", "fill", "empty"])
    assert skeleton(lamps) == skeleton(mugs)
    assert skeleton(binary("state", ["lit", "dark"], ["not", ["is_not", "P", "state", "dark"]], ["is", "Q", "state", "lit"])) == \
        skeleton(binary("state", ["lit", "dark"], ["not", ["is", "P", "state", "lit"]], ["is", "Q", "state", "lit"]))
    three = binary("fill", ["full", "half", "empty"], ["count", "fill", "full", ">=", 2], ["is", "R", "fill", "empty"])
    assert skeleton(binary("fill", ["full", "half", "empty"], ["count", "fill", "full", ">=", 2], ["is_not", "P", "fill", "full"])) != skeleton(three)
    # Past the reading cap the facts are compared as written, still renamed.
    many = m(*[["not_adjacent", "X", "Y", "h"]] * 9)
    assert skeleton(many) == skeleton(m(*[["not_adjacent", "Y", "Z", "h"]] * 9))
