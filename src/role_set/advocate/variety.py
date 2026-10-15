"""Variety floors for the Advocate plan, gated the way T10 is: a committed record and a hash.

The three comparisons are at least 75% distinct skeletons, a top repeat of at most 3,
and no skeleton twice in one trait's file. The hash is closed over everything the
planner imports. It does not cover this module or the tests.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

RECORD_HASH_FILES = (
    "src/role_set/advocate/kinds.py",
    "src/role_set/advocate/ledger.py",
    "src/role_set/advocate/plan.py",
    "src/role_set/advocate/table.py",
    "src/role_set/pack.py",
)
REGENERATE = "uv run python -m role_set.advocate.variety"
_RECORD = Path(__file__).with_name("variety_record.json")
_TRAITS = ("failed-gate-first", "exact-figures", "faithful-relay")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def record_sha256(root: Path | None = None) -> str:
    base = repo_root() if root is None else root
    digest = hashlib.sha256()
    for rel in RECORD_HASH_FILES:
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update((base / rel).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def load_record() -> dict:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


def skeletons_by_trait(slots) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {trait: [] for trait in _TRAITS}
    for slot in slots:
        grouped.setdefault(slot.trait, []).append(slot.skeleton)
    return grouped


def floor_report(skeletons: list[str], by_file: dict[str, list[str]]) -> dict:
    """The three comparisons. `within` is how many extra copies sit inside one file."""

    top = Counter(skeletons).most_common(1)[0][1]
    within = {
        name: sum(count - 1 for count in Counter(skins).values())
        for name, skins in by_file.items()
    }
    return {
        "distinct": len(set(skeletons)) / len(skeletons),
        "top": top,
        "within": within,
    }


def floors_hold(report: dict) -> bool:
    return report["distinct"] >= 0.75 and report["top"] <= 3 and all(n == 0 for n in report["within"].values())


def check_record_hash(root: Path | None = None) -> dict:
    data = load_record()
    if data.get("regenerate") != REGENERATE or data.get("files") != list(RECORD_HASH_FILES):
        raise ValueError(f"Advocate variety record metadata mismatch; regenerate with: {REGENERATE}")
    actual = record_sha256(root)
    if data.get("sha256") != actual:
        raise ValueError(
            f"Advocate variety record sha256 mismatch ({data.get('sha256')} != {actual}); "
            f"regenerate with: {REGENERATE}"
        )
    return data


def planned_rows(seed: int = 0) -> list[dict]:
    from role_set.advocate.plan import plan_batch

    pack = repo_root() / "roles" / "advocate"
    slots = plan_batch(pack, list(_TRAITS), seed=seed)
    return [{"trait": slot.trait, "skeleton": slot.skeleton} for slot in slots]


def main() -> None:
    rows = planned_rows(0)
    data = {
        "regenerate": REGENERATE,
        "files": list(RECORD_HASH_FILES),
        "sha256": record_sha256(),
        "seed": 0,
        "slots": rows,
    }
    _RECORD.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} skeletons to {_RECORD.name}")


if __name__ == "__main__":  # pragma: no cover
    main()
