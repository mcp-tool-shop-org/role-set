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
