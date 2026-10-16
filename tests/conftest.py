"""Test hooks shared by the suite."""


def pytest_runtest_logreport(report):
    """xdist keeps a worker's print on the worker. The controller repeats a stratum line."""

    if report.when != "call" or not hasattr(report, "node"):
        return
    for key, value in report.user_properties:
        if key == "stratum":
            print("\n" + value, flush=True)


# ---- the fixture domain table (BUILD-SPEC B2) ---------------------------------------------------------------
# No real NAICS titles until the pinned census.gov file is downloaded (R&D, 2026-10-16): the fixture's titles are
# "Fixture title <code>".
import importlib.util  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAPPED_CODES = ["23", "52", "2211", "325", "481", "482", "5411", "5412", "5413", "621", "622", "623", "9221"]


def naics_script():
    spec = importlib.util.spec_from_file_location("naics_table", ROOT / "scripts" / "naics_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_xlsx(path: Path, codes: list[str]) -> Path:
    """A minimal workbook in the source's shape: a header row, then Seq. No., code, title."""

    def cell(ref, text):
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    rows = ['<row r="1">' + cell("A1", "Seq. No.") + cell("B1", "2022 NAICS US Code") + cell("C1", "2022 NAICS US Title") + "</row>"]
    for i, code in enumerate(codes, start=2):
        rows.append(f'<row r="{i}"><c r="A{i}"><v>{i - 1}</v></c><c r="B{i}"><v>{code}</v></c>' + cell(f"C{i}", f"Fixture title {code}") + "</row>")
    sheet = ('<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             "<sheetData>" + "".join(rows) + "</sheetData></worksheet>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return path


@pytest.fixture(scope="session")
def domain_table(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("naics")
    xlsx = fixture_xlsx(d / "2-6 digit_2022_Codes.xlsx", MAPPED_CODES + [str(c) for c in range(100, 160)])
    out = d / "naics-2022.json"
    assert naics_script().main([str(xlsx), "--out", str(out)]) == 0
    return out
