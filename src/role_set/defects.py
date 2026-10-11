"""The defect catalog (`roles/verifier/defects.jsonl`): one row per class of defect that a review found in the hand-
or Grok-written batches and the pilots, and how role-set handles it today.

Each row is `{"id", "name", "description", "first_source", "handled", "anchor"}`:
- `handled` is `gate` (a mechanical refusal: a check in code, or a critic check that refuses), `puzzle-check`,
  `brief` (a writer instruction only), `critic` (an advisory critic check, logged and never refusing) or `none`;
- `anchor` names the check or the brief section, or for `none` the candidate check.

A review that finds a new class adds a row; a class that becomes mechanically checkable moves from `brief` to `gate`
or `puzzle-check`. `readout` lists every `none` row, so the open gaps stay in sight on every run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

FIELDS = ("id", "name", "description", "first_source", "handled", "anchor")
HANDLED = ("gate", "puzzle-check", "brief", "critic", "none")
CATALOG = Path("roles") / "verifier" / "defects.jsonl"
NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")


class DefectError(ValueError):
    """The catalog file doesn't hold to its shape."""


def load(path: Path) -> list[dict]:
    rows, ids, names = [], set(), set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise DefectError(f"line {n} is not JSON: {e}") from None
        missing = [f for f in FIELDS if not str(row.get(f) or "").strip()]
        if missing or set(row) - set(FIELDS):
            raise DefectError(f"line {n}: fields must be exactly {', '.join(FIELDS)}, all non-empty (missing {missing})")
        if row["handled"] not in HANDLED:
            raise DefectError(f"line {n}: handled {row['handled']!r} is not one of {', '.join(HANDLED)}")
        if not NAME.fullmatch(row["name"]):
            raise DefectError(f"line {n}: name {row['name']!r} is not snake_case")
        if row["id"] in ids or row["name"] in names:
            raise DefectError(f"line {n}: {row['id']} / {row['name']} repeats an earlier row")
        ids.add(row["id"])
        names.add(row["name"])
        rows.append(row)
    return rows


def find(start: Path | None = None) -> Path | None:
    """The catalog beside the working directory, else the one in this checkout."""

    for base in ((start or Path.cwd()), Path(__file__).resolve().parents[2]):
        if (base / CATALOG).exists():
            return base / CATALOG
    return None


def open_classes(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["handled"] == "none"]
