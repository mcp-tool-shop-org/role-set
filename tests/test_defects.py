"""The defect catalog: its shape, and every open class listed on the readout so the gaps stay visible."""

import json
from pathlib import Path

import pytest

from role_set.cli import main
from role_set.defects import CATALOG, HANDLED, DefectError, find, load, open_classes
from role_set.plan import plan_lesson
from role_set.readout import open_defects_markdown, readout
from test_runner import SIDES, _runner

REPO_CATALOG = Path(__file__).resolve().parents[1] / CATALOG


def test_the_catalog_loads_and_every_row_holds_its_shape():
    rows = load(REPO_CATALOG)
    assert len(rows) >= 30
    assert [r["id"] for r in rows] == [f"D{i:03d}" for i in range(1, len(rows) + 1)]
    assert {r["handled"] for r in rows} <= set(HANDLED) and open_classes(rows)
    assert find(Path(__file__).resolve().parents[1]) == REPO_CATALOG


def _run(tmp_path: Path) -> Path:
    r, _, _ = _runner(tmp_path)
    r.run([next(s for s in plan_lesson("T02", SIDES, seed=0) if s.kind == "A")])
    return tmp_path / "out"


def test_every_open_class_is_listed_in_the_readout(tmp_path: Path, capsys):
    rows = load(REPO_CATALOG)
    run = _run(tmp_path)
    data, md = readout(run, catalog=rows)
    for r in open_classes(rows):
        assert f"- {r['name']} ({r['id']}):" in md and r["name"] in data["open_defects"]
    assert main(["readout", "--run", str(run), "--catalog", str(REPO_CATALOG)]) == 0
    out = capsys.readouterr().out
    assert all(r["name"] in out for r in open_classes(rows))
    assert "open_defects" not in readout(run)[0]  # no catalog given, no section


def test_a_catalog_with_no_open_class_says_so():
    assert open_defects_markdown([]).strip() == "Open defect classes (no check yet): none."


ROW = {"id": "D001", "name": "a_class", "description": "d", "first_source": "s", "handled": "brief", "anchor": "a"}


@pytest.mark.parametrize("rows, match", [
    (["{not json"], "is not JSON"),
    ([{**ROW, "anchor": ""}], "fields must be exactly"),
    ([{**ROW, "extra": 1}], "fields must be exactly"),
    ([{**ROW, "handled": "maybe"}], "is not one of"),
    ([{**ROW, "name": "Bad-Name"}], "not snake_case"),
    ([ROW, {**ROW, "id": "D002"}], "repeats an earlier row"),
])
def test_a_malformed_catalog_is_refused(tmp_path: Path, rows, match):
    p = tmp_path / "defects.jsonl"
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(DefectError, match=match):
        load(p)
