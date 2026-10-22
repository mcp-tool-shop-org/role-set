"""Verify that re-planning T10 at seed 0 with buffers changes only T10-B-05, T10-B-07, and T10-B-08."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_CHANGED = ("T10-B-05", "T10-B-07", "T10-B-08")
ALLOWED_FIELDS = frozenset({"b_path", "b_skeleton"})


def _file_sha256(path: Path) -> str:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def check_plans(old_path: Path, new_path: Path) -> tuple[bool, list[str]]:
    old_data = json.loads(old_path.read_text(encoding="utf-8"))
    new_data = json.loads(new_path.read_text(encoding="utf-8"))
    old_slots = old_data.get("slots") or []
    new_slots = new_data.get("slots") or []
    lines: list[str] = [
        f"old: {old_path.name} (sha256={_file_sha256(old_path)})",
        f"new: {new_path.name} (sha256={_file_sha256(new_path)})",
    ]
    old_ids = [s.get("id") for s in old_slots]
    new_ids = [s.get("id") for s in new_slots]
    if old_ids != new_ids:
        lines.append(f"FAIL: slot ID order differs (old={len(old_ids)} slots, new={len(new_ids)} slots)")
        return False, lines

    changed_ids: list[str] = []
    unexpected_fields: dict[str, list[str]] = {}
    for old_s, new_s in zip(old_slots, new_slots):
        sid = str(old_s.get("id"))
        if json.dumps(old_s, ensure_ascii=False) == json.dumps(new_s, ensure_ascii=False):
            continue
        changed_ids.append(sid)
        keys = list(dict.fromkeys([*old_s.keys(), *new_s.keys()]))
        diffs = [k for k in keys if json.dumps(old_s.get(k), sort_keys=True) != json.dumps(new_s.get(k), sort_keys=True)]
        lines.append(f"{sid} ({new_s.get('structure')}):")
        for k in diffs:
            lines.append(f"  {k}: {old_s.get(k)!r} -> {new_s.get(k)!r}")
        bad = [k for k in diffs if k not in ALLOWED_FIELDS]
        if bad:
            unexpected_fields[sid] = bad

    unchanged = len(old_slots) - len(changed_ids)
    lines.append(f"unchanged: {unchanged}/{len(old_slots)} slots byte-identical")
    if tuple(changed_ids) != EXPECTED_CHANGED:
        lines.append(f"FAIL: changed slots {changed_ids} != expected {list(EXPECTED_CHANGED)}")
        return False, lines
    if unexpected_fields:
        lines.append(f"FAIL: unexpected fields changed: {unexpected_fields}")
        return False, lines
    return True, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path, help="Path to frozen old plan.json")
    parser.add_argument("--new", required=True, type=Path, help="Path to new plan.json")
    args = parser.parse_args(argv)
    ok, lines = check_plans(args.old, args.new)
    stream = sys.stdout if ok else sys.stderr
    for line in lines:
        print(line, file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
