"""How many distinct skeletons each T10 cell yields on the tuning seeds.

The allocator uses the counts as weights. An if cell that kept producing new skeletons
can take another visit before one that repeated. Tuning seeds only: 1, 21 and 41.
Locked seeds are not built. Re-run to refresh src/role_set/generate/t10_capacity.json.

    uv run python scripts/t10_cell_capacity.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from role_set.generate.recipes import PACKS, build_t10
from role_set.generate.t10_variety import canonical_skeleton, cell_key, load_cells

SEEDS = (1, 21, 41)
OUT = Path(__file__).resolve().parents[1] / "src" / "role_set" / "generate" / "t10_capacity.json"


def _jobs() -> list[tuple[str, str, dict, int]]:
    table = load_cells()
    jobs = []
    for pack in table["packs"]:
        for question in ("set", "claim"):
            for cell in table["cells"][pack][question]:
                for seed in SEEDS:
                    jobs.append((pack, question, cell, seed))
    return jobs


def _kind(pack: str) -> str:
    return "two" if len(PACKS[pack]["values"]) == 2 else "three"


def one(job: tuple[str, str, dict, int]) -> tuple[str, str, str, str | None]:
    pack, question, cell, seed = job
    kind = _kind(pack)
    try:
        built = build_t10(pack, seed, question, cell=cell)
    except Exception:
        return question, kind, cell_key(cell), None
    return question, kind, cell_key(cell), canonical_skeleton(built.model)


def main() -> None:
    jobs = _jobs()
    counts: dict[str, dict[str, dict[str, dict]]] = {
        "set": {"two": {}, "three": {}},
        "claim": {"two": {}, "three": {}},
    }
    workers = min(16, os.cpu_count() or 4)
    print(f"0/{len(jobs)} workers {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (question, kind, key, skeleton) in enumerate(pool.map(one, jobs), 1):
            row = counts[question][kind].setdefault(key, {"builds": 0, "failed": 0, "skins": set()})
            if skeleton is None:
                row["failed"] += 1
            else:
                row["builds"] += 1
                row["skins"].add(skeleton)
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)}", flush=True)
    cells: dict[str, dict[str, dict]] = {"set": {}, "claim": {}}
    for question, by_kind in counts.items():
        for kind, rows in by_kind.items():
            cells[question][kind] = {
                key: {"distinct": len(row["skins"]), "builds": row["builds"], "failed": row["failed"]}
                for key, row in sorted(rows.items())
            }
    payload = {
        "seeds": list(SEEDS),
        "packs": list(PACKS),
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for question, by_kind in cells.items():
        for kind, rows in by_kind.items():
            distinct = sorted(row["distinct"] for row in rows.values())
            print(
                f"{question} {kind}: {len(rows)} cells, distinct skeletons "
                f"min {distinct[0]} median {distinct[len(distinct)//2]} max {distinct[-1]}"
            )


if __name__ == "__main__":
    main()
