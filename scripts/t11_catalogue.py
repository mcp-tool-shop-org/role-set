"""Write src/role_set/generate/t11_catalogue.json: one proven T11 item per fact skeleton, by value count and side.

Enumerates every set of 2 or 3 clues over three entities (e1, e2 and the claim's subject s) and one unordered attribute
with 2 or 3 values, on the T11 forms (`is`, `is_not`, `in`, `not_in`, and `if` with an `is` or `is_not` antecedent). An
item is kept when the solver's own checks pass on it (`check_model`, `t11_refusal`, and for the find side a T11 reading
that differs), and the first item of each skeleton is recorded. Builds draw distinct skeletons from this table, so
variety is set where the puzzle is made, not filtered after (R&D's variety floor on role-set#52).

    uv run python scripts/t11_catalogue.py           # write
    uv run python scripts/t11_catalogue.py --check   # fail if the committed table is stale
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from role_set.generate.build import Shape, _key, _norm, assignments, check_model, truth
from role_set.generate.recipes import T11_SIDES_BUILT, t11_contrast, t11_pool, t11_refusal
from role_set.generate.shortcuts import names_in, prove
from role_set.puzzle import PuzzleError, holds
from role_set.puzzle_check import skeleton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "role_set" / "generate" / "t11_catalogue.json"
VALUES = {2: ["v1", "v2"], 3: ["v1", "v2", "v3"]}
NAMES = ["e1", "e2", "s"]
FACTS = (2, 3)
# Rows kept per value count and side: every 2-fact skeleton, then 3-fact ones in skeleton-hash order. 480 covers four
# files of 20 seeds over the six three-value packs, each seed on its own skeleton.
CAP = 480
READING = "T11.consistent_so_supported"


def enumerate_side(nvalues: int, side: str) -> list[dict]:
    values = VALUES[nvalues]
    attrs = {"a": {"values": values, "distinct": False}}
    claim = ["is", "s", "a", "v1"]
    shape = Shape(tuple(NAMES), attrs, claim=claim, structure="falsifier_first")
    sols = assignments(shape)
    full = (1 << len(sols)) - 1
    pool = t11_pool(shape)
    bits = [truth(shape, sols, c) for c in pool]
    by_key = {_key(c): b for c, b in zip(pool, bits)}
    m = _norm(shape)
    claim_true = [holds(m, x, claim) for x in sols]
    true_mask = sum(1 << i for i, t in enumerate(claim_true) if t)
    want = "unsupported" if side == "find" else "supported"

    def verdict(mask: int) -> str | None:
        if not mask:
            return None
        return "supported" if mask & true_mask == mask else "unsupported" if not mask & true_mask else "cannot_tell"

    seen: dict[str, dict] = {}
    for k in FACTS:
        for combo in itertools.combinations(range(len(pool)), k):
            mask = full
            for i in combo:
                mask &= bits[i]
            if verdict(mask) != want or any(verdict(bits[i]) == want for i in combo):
                continue
            if any(verdict(_without(bits, combo, i, full)) == want for i in combo):
                continue  # a fact the verdict doesn't need
            model = {"entities": NAMES, "attributes": attrs, "claim": claim, "flavour": [1],
                     "facts": [{"line": j + 2, "rule": pool[i]} for j, i in enumerate(combo)]}
            key = skeleton(model)
            if key in seen:
                continue
            if t11_refusal(model, side) is not None:
                continue
            try:
                check_model(model, "falsifier_first")
            except PuzzleError:
                continue
            if side == "find" and not prove(model, "falsifier_first", [READING])[READING]["differs"]:
                continue
            kept = [pool[i] for i in combo]
            if side == "confirm" and t11_contrast(shape, sols, kept, pool, by_key, claim_true, random.Random(0)) is None:
                continue  # no off-subject one-slot change moves the verdict off supported
            seen[key] = {"facts": [pool[i] for i in combo],
                         "skeleton_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                         "subject_lines": sum(bool(names_in(pool[i], set(NAMES)) & {"s"}) for i in combo)}
    rows = sorted(seen.values(), key=lambda r: (len(r["facts"]), r["skeleton_sha256"]))
    return rows[:CAP]


def _without(bits: list[int], combo: tuple, drop: int, full: int) -> int:
    mask = full
    for j in combo:
        if j != drop:
            mask &= bits[j]
    return mask


def build_table() -> dict:
    table = {"names": NAMES, "claim": ["is", "s", "a", "v1"], "values": {str(k): v for k, v in VALUES.items()},
             "sides": {}}
    for side in T11_SIDES_BUILT:
        table["sides"][side] = {str(nv): enumerate_side(nv, side) for nv in VALUES}
    return table


def _dump(table: dict) -> str:
    """One row per line, so a diff of the table reads row by row."""

    head = {k: v for k, v in table.items() if k != "sides"}
    lines = ["{", f' "head": {json.dumps(head, sort_keys=True)},', ' "sides": {']
    sides = list(table["sides"].items())
    for si, (side, by_nv) in enumerate(sides):
        lines.append(f"  {json.dumps(side)}: {{")
        groups = list(by_nv.items())
        for gi, (nv, rows) in enumerate(groups):
            lines.append(f"   {json.dumps(nv)}: [")
            lines += [f"    {json.dumps(r, sort_keys=True)}{',' if ri < len(rows) - 1 else ''}" for ri, r in enumerate(rows)]
            lines.append(f"   ]{',' if gi < len(groups) - 1 else ''}")
        lines.append(f"  }}{',' if si < len(sides) - 1 else ''}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    table = build_table()
    text = _dump(table)
    if "--check" in sys.argv:
        if OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT.name} is stale; regenerate with: uv run python scripts/t11_catalogue.py")
        print(f"{OUT.name} is current")
        return
    OUT.write_text(text, encoding="utf-8", newline="\n")
    for side, by_nv in table["sides"].items():
        print(side, {nv: len(rows) for nv, rows in by_nv.items()})


if __name__ == "__main__":
    main()
