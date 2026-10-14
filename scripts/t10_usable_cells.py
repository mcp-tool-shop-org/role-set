"""Which T10 cells build from the probe seed with tries to spare.

A cell is n in {3, 4, 5}, a count word (exactly, at least, at most), k in 1..n-1, and an if count in {0, 1, 2}.
The probe seed is 0, which is neither a tuning seed nor a locked seed. A cell is recorded only when that seed builds it
in at most MAX_TRIES // 10 attempts. A cell that spends most of the budget on the probe then misses on a later seed:
cups claim, seed 13, n=5 at least 2 with two if clues, took 72 probe tries and did not build in 400. The list is the
data in src/role_set/generate/t10_cells.json. Re-run this script to make that file again.

    uv run python scripts/t10_usable_cells.py
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from role_set.generate.build import MAX_TRIES, BuildError
from role_set.generate.recipes import PACKS, build_t10

PROBE_SEED = 0
# A later seed has MAX_TRIES attempts of its own. Keep a cell only when the probe needed a tenth of that.
RELIABLE_TRIES = MAX_TRIES // 10
OPS = ("=", ">=", "<=")
OUT = Path(__file__).resolve().parents[1] / "src" / "role_set" / "generate" / "t10_cells.json"


def every_cell() -> list[dict]:
    cells = []
    for n in (3, 4, 5):
        for op in OPS:
            for k in range(1, n):
                for ifs in (0, 1, 2):
                    cells.append({"n": n, "op": op, "k": k, "ifs": ifs})
    return cells


def probe(job: tuple[str, str, dict]) -> tuple[str, str, dict, bool, int | str]:
    pack, question, cell = job
    try:
        built = build_t10(pack, PROBE_SEED, question, cell=cell, tries=RELIABLE_TRIES)
    except (BuildError, Exception) as exc:  # a cell that cannot be built is recorded, not raised
        return pack, question, cell, False, f"{type(exc).__name__}: {exc}"
    return pack, question, cell, True, built.tries


def main() -> None:
    jobs = [(pack, question, cell) for pack in PACKS for question in ("set", "claim") for cell in every_cell()]
    found: dict[str, dict[str, list[dict]]] = {pack: {"set": [], "claim": []} for pack in PACKS}
    with ProcessPoolExecutor() as pool:
        for done, (pack, question, cell, ok, detail) in enumerate(pool.map(probe, jobs), 1):
            if ok:
                found[pack][question].append(cell)
            mark = f"ok tries={detail}" if ok else "no " + str(detail).split("last refusal: ")[-1][:90]
            print(f"{done}/{len(jobs)} {pack} {question} n={cell['n']} {cell['op']} k={cell['k']} ifs={cell['ifs']} {mark}",
                  flush=True)
    payload = {"probe_seed": PROBE_SEED, "tries": RELIABLE_TRIES, "packs": list(PACKS), "cells": found}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    short = False
    for pack, questions in found.items():
        for question, cells in questions.items():
            print(f"{pack} {question}: {len(cells)} usable")
            if len(cells) < 20:
                short = True
    if short:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
