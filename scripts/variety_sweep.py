"""Report fact-skeleton variety per recipe against R&D's floor (role-set#52): at least 75% distinct, no skeleton on more
than 5% of builds (3 of 60), measured with `puzzle_check.skeleton`.

- T11, per side: the acceptance sweep (lamps, cups and doors, seeds 1-20, 60 builds a side), then all packs, with the
  split of contrast_verdict on the confirm side (R&D's Q3 ruling, point 4). Also the
  share of unsupported keys in a file the selector picks from both sides, which R&D asks to keep at 40% or more.
- T10: from the committed record (t10_record.json), per question, seeds 1-20 of every pack.
- T12: every pack at 3 and 4 spans, seeds 1-20.

    uv run python scripts/variety_sweep.py
    uv run python scripts/variety_sweep.py --check   # exit 1 when T11 misses the floor (T10 and T12 are reported only)
"""

from __future__ import annotations

import sys
from collections import Counter

from role_set.generate.build import receipt
from role_set.generate.recipes import PACKS, T12_PACKS, T11_SIDES, build_t11, build_t12
from role_set.generate.select import select
from role_set.generate.t10_variety import load_record
from role_set.puzzle_check import skeleton

SWEEP_PACKS = ("lamps", "cups", "doors")
SEEDS = range(1, 21)


def floor_line(label: str, keys: list[str]) -> tuple[str, bool]:
    tally = Counter(keys)
    n, distinct, top = len(keys), len(tally), max(tally.values()) if tally else 0
    ok = n > 0 and distinct >= 0.75 * n and top <= max(1, int(0.05 * n + 1e-9))
    return f"{label:<34} {n:>4} builds  {distinct:>4} distinct ({distinct / max(n, 1):.0%})  top {top}  {'PASS' if ok else 'BELOW'}", ok


def main() -> None:
    lines, t11_ok = [], True
    for side in T11_SIDES:
        sweep = [build_t11(p, s, side) for p in SWEEP_PACKS for s in SEEDS]
        line, ok = floor_line(f"T11 {side} (sweep)", [skeleton(b.model) for b in sweep])
        lines.append(line)
        t11_ok &= ok
        allp = [build_t11(p, s, side) for p in sorted(PACKS) for s in SEEDS]
        lines.append(floor_line(f"T11 {side} (all packs)", [skeleton(b.model) for b in allp])[0])
        # Where each confirm contrast moves the verdict (R&D's Q3 ruling, point 4); a find item has no contrast.
        for label, builds in (("sweep", sweep), ("all packs", allp)):
            split = Counter(b.contrast["contrast_verdict"] for b in builds if b.contrast)
            shown = ", ".join(f"{v} {split[v]}" for v in ("cannot_tell", "unsupported")) if split else "no contrasts on this side"
            lines.append(f"T11 {side} contrast_verdict ({label}): {shown}")
    pool = [receipt(build_t11(p, s, side), "T11", PACKS[p], "falsifier_first")
            for p in SWEEP_PACKS for side in T11_SIDES for s in range(1, 6)]
    picks = select(pool)
    unsupported = sum(pool[p.index]["answer"] == "unsupported" for p in picks) / len(picks)
    lines.append(f"T11 selected file: {len(picks)} items, unsupported share {unsupported:.0%} (R&D: at least 40%)")
    t11_ok &= unsupported >= 0.4
    record = load_record()
    for q in ("set", "claim"):
        keys = [b["canonical_skeleton"] for b in record["builds"] if b.get("question") == q and b.get("seed") in SEEDS
                and b.get("built", True)]
        lines.append(floor_line(f"T10 {q} (record, seeds 1-20)", keys)[0] + "  (canonical_skeleton)")
    for n in (3, 4):
        keys = [skeleton(build_t12(p, s, n=n).model) for p in sorted(T12_PACKS) for s in SEEDS]
        lines.append(floor_line(f"T12 {n} spans", keys)[0])
    print("\n".join(lines))
    if "--check" in sys.argv and not t11_ok:
        sys.exit("T11 misses R&D's variety floor")


if __name__ == "__main__":
    main()
