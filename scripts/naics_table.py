"""Build the domain table (BUILD-SPEC B2) from the pinned NAICS 2022 source file.

    uv run python scripts/naics_table.py "<path to 2-6 digit_2022_Codes.xlsx>" --out roles/domains/naics-2022.json

The source is the US Census Bureau's 2022 NAICS code list, downloaded from SOURCE_URL. The table keeps the source's
sha256 and URL, rnd's licence registry id (census-naics-2022), the 3-digit subsector titles verbatim, and the ESCALATE
mapping R&D ruled on 2026-10-16, with each mapped code's title verbatim from the same file. The xlsx is read with the
standard library (it is a zip of XML parts); no spreadsheet package is needed.

Standards: PIN_PER_STEP (the source file's sha256 goes in the table; the table's own sha256 goes on every plan),
ANDON_AUTHORITY (a mapped code missing from the source stops the build). The script writes only --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_URL = "https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx"
REGISTRY_ID = "census-naics-2022"
# Left out of the ordinary pool (R&D, 2026-10-16): Public Administration's subsectors are government offices, odd
# material for everyday notices. 9221 stays reachable through the safety mapping.
EXCLUDED = {"921", "922", "923", "924", "925", "926", "927", "928"}
EXCLUDED_REASON = "921-928 (Public Administration) excluded from the ordinary pool: government offices, odd material for everyday notices; 9221 stays in the safety mapping (R&D, 2026-10-16)"
# R&D, 2026-10-16: each escalation domain maps to the trades that hold that qualification.
ESCALATE_CODES = {
    "medical": ("621", "622", "623"),
    "legal": ("5411",),
    "money": ("52", "5412"),
    "safety": ("23", "2211", "325", "481", "482", "5413", "9221"),
}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")) for si in root.findall("m:si", NS)]


def rows(xlsx: Path) -> list[list[str]]:
    """The first sheet's cell values, row by row, as text."""

    with zipfile.ZipFile(xlsx) as z:
        strings = _shared_strings(z)
        sheet = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))[0]
        root = ET.fromstring(z.read(sheet))
    out = []
    for r in root.iter(f"{{{NS['m']}}}row"):
        cells = []
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS)
            if c.get("t") == "inlineStr":
                text = "".join(t.text or "" for t in c.iter(f"{{{NS['m']}}}t"))
            elif v is None:
                text = ""
            elif c.get("t") == "s":
                text = strings[int(v.text)]
            else:
                text = v.text or ""
            cells.append(text.strip())
        out.append(cells)
    return out


def codes(xlsx: Path) -> dict[str, str]:
    """code -> title, verbatim, from the columns the header row names (a code column and a title column; the
    source's sequence-number column is never read as a code). Sector ranges like 31-33 are kept under their range."""

    table = rows(xlsx)
    for h, header in enumerate(table):
        code_col = next((i for i, c in enumerate(header) if re.search(r"code", c, re.I)), None)
        title_col = next((i for i, c in enumerate(header) if re.search(r"title", c, re.I)), None)
        if code_col is not None and title_col is not None:
            break
    else:
        raise SystemExit("REFUSED: no header row naming a code column and a title column")
    found: dict[str, str] = {}
    for cells in table[h + 1:]:
        if max(code_col, title_col) >= len(cells):
            continue
        code, title = cells[code_col].split(".")[0], cells[title_col]
        if re.fullmatch(r"\d{2,6}|\d{2}-\d{2}", code) and title:
            found.setdefault(code, title)
    return found


def build(xlsx: Path) -> dict:
    data = xlsx.read_bytes()
    table = codes(xlsx)
    subsectors = [{"code": c, "title": t} for c, t in sorted(table.items()) if len(c) == 3 and c.isdigit() and c not in EXCLUDED]
    missing = sorted({c for cs in ESCALATE_CODES.values() for c in cs} - set(table))
    if missing:
        raise SystemExit(f"REFUSED: mapped codes not in the source file: {', '.join(missing)}")
    return {
        "_note": "The NAICS 2022 domain table (BUILD-SPEC B2). Built by scripts/naics_table.py from the source file below; titles are verbatim. Never hand-edit: rebuild from the pinned file. " + EXCLUDED_REASON + ".",
        "source": {"url": SOURCE_URL, "file": xlsx.name, "sha256": hashlib.sha256(data).hexdigest(), "registry_id": REGISTRY_ID,
                   "owner": "US Census Bureau", "edition": "NAICS 2022 (US)"},
        "provenance": {"subsectors": "US Census Bureau (public data, census-naics-2022)",
                       "escalate_mapping": "R&D ruling, 2026-10-16 (codes); titles from the source file"},
        "subsectors": subsectors,
        "escalate": {d: [{"code": c, "title": table[c]} for c in cs] for d, cs in ESCALATE_CODES.items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("xlsx", help="the pinned 2-6 digit_2022_Codes.xlsx")
    p.add_argument("--out", default="roles/domains/naics-2022.json")
    args = p.parse_args(argv)
    table = build(Path(args.xlsx))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(table['subsectors'])} subsectors and {sum(len(v) for v in table['escalate'].values())} mapped codes to {out}; "
          f"source sha256 {table['source']['sha256']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
