"""Write src/role_set/generate/t12_catalogue.json: one proven T12 clue set per fact skeleton, by span count and form.

Enumerates every set of 2 or 3 clues on T12's forms over n editions and n spans in start order (n = 3, 4). A set is kept
when every clue is needed for what it decides:
- ask: which edition holds the span at a position (the rows keep each such position);
- claim: whether an edition holds the span at a position, supported or unsupported (the rows keep each such
  [edition, position, verdict]; R&D's claim ruling on role-set#52's follow-up).
Each row is checked once by `check_model` on a real calendar. Builds draw distinct skeletons from this table, place a
calendar and names on the row, and keep the build only when a T12 reading names another answer (or flips the claim).

    uv run python scripts/t12_catalogue.py           # write
    uv run python scripts/t12_catalogue.py --check   # fail if the committed table is stale
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

from role_set.generate.build import Shape, assignments, candidates, check_model, truth
from role_set.generate.recipes import T12, t12_calendar
from role_set.puzzle import PuzzleError
from role_set.puzzle_check import skeleton

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "role_set" / "generate" / "t12_catalogue.json"
SPANS = (3, 4)
FACTS = (2, 3)
FORMS = ("ask", "claim")
CAP = 480  # rows per span count and form; a file of 20 seeds over the four packs takes 80


def _abstract(n: int):
    ents = [f"e{i}" for i in range(n)]
    vals = [f"s{i}" for i in range(n)]
    shape = Shape(tuple(ents), {"span": {"values": vals, "ordered": True, "distinct": True}},
                  asks={"entity_with": {"span": "s0"}})
    return ents, vals, shape


def _placed(n: int, facts: list[list], form: str, target: list) -> dict:
    """The row on a real calendar, for check_model: s_i becomes the i-th span, the asked date falls in span p."""

    spans, span, asked = t12_calendar(n, random.Random(f"catalogue:{n}:{target}"), target[1] if form == "claim" else target[0])
    names = [f"edition {i + 3}" for i in range(n)]
    ent = {f"e{i}": names[i] for i in range(n)}
    val = {f"s{i}": spans[i] for i in range(n)}

    def place(r):
        out = [r[0]]
        for x in r[1:]:
            out.append(ent.get(x, val.get(x, x)) if isinstance(x, str) else [val.get(v, v) for v in x] if isinstance(x, list) else x)
        return out

    shape = Shape(tuple(names), {"span": {"values": spans, "ordered": True}},
                  **({"asks": {"contains": asked}} if form == "ask" else {"claim": ["contains", ent[target[0]], asked]}),
                  span=span)
    return shape.model([place(r) for r in facts])


def enumerate_rows(n: int, form: str) -> list[dict]:
    ents, vals, shape = _abstract(n)
    sols = assignments(shape)
    full = (1 << len(sols)) - 1
    pool = candidates(shape, T12.forms)
    bits = [truth(shape, sols, c) for c in pool]
    holder = [[next(e for e in range(n) if s["span"][e] == f"s{p}") for s in sols] for p in range(n)]

    def ask_fixed(mask: int, p: int) -> bool:
        return len({holder[p][j] for j in range(len(sols)) if mask >> j & 1}) == 1

    def claim_verdict(mask: int, e: int, p: int) -> str | None:
        seen = {holder[p][j] == e for j in range(len(sols)) if mask >> j & 1}
        return None if len(seen) != 1 else ("supported" if True in seen else "unsupported")

    def holders(mask: int, p: int) -> set[int]:
        return {holder[p][j] for j in range(len(sols)) if mask >> j & 1}

    def acts(mask: int, target: list) -> bool:
        """Whether `newest` or `boundary_next` can give another answer here (in any pack, numbered names or not; the
        boundary reading once the calendar asks a boundary day). Both depend only on the clues, so the catalogue
        keeps only targets some reading can act on."""

        p = target[0] if form == "ask" else target[1]
        read = [holders(mask, n - 1)] + [holders(mask, q) for q in (p - 1, p + 1) if 0 <= q < n]
        if form == "ask":
            (a,) = holders(mask, p)
            return any(s != {a} if len(s) == 1 else a not in s for s in read)
        e, v = int(target[0][1:]), target[2]
        lazy = ["supported" if s == {e} else "unsupported" if e not in s else "cannot_tell" for s in read]
        return any(x != v for x in lazy)

    seen: dict[str, dict] = {}
    for k in FACTS:
        for combo in itertools.combinations(range(len(pool)), k):
            mask = full
            for i in combo:
                mask &= bits[i]
            if not mask:
                continue
            drops = [_without(bits, combo, i, full) for i in range(k)]
            if form == "ask":
                targets = [[p] for p in range(n) if ask_fixed(mask, p) and not any(ask_fixed(d, p) for d in drops)]
            else:
                targets = []
                for e, p in itertools.product(range(n), range(n)):
                    v = claim_verdict(mask, e, p)
                    if v and not any(claim_verdict(d, e, p) == v for d in drops):
                        targets.append([f"e{e}", p, v])
            targets = [t for t in targets if acts(mask, t)]
            if not targets:
                continue
            facts = [pool[i] for i in combo]
            model = {"entities": ents, "attributes": shape.attributes, "flavour": [1],
                     "facts": [{"line": j + 2, "rule": r} for j, r in enumerate(facts)], "asks": {"entity_with": {"span": "s0"}}}
            key = skeleton(model)
            if key in seen:
                continue
            good = []
            for t in targets:
                try:
                    check_model(_placed(n, facts, form, t), "constraint_grid")
                    good.append(t)
                except PuzzleError:
                    pass
            if good:
                seen[key] = {"facts": facts, "targets": good,
                             "skeleton_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]}
    rows = sorted(seen.values(), key=lambda r: (len(r["facts"]), r["skeleton_sha256"]))
    return rows[:CAP]


def _without(bits: list[int], combo: tuple, drop: int, full: int) -> int:
    mask = full
    for j, i in enumerate(combo):
        if j != drop:
            mask &= bits[i]
    return mask


def build_table() -> dict:
    return {"head": {"forms": list(FORMS), "spans": list(SPANS)},
            "rows": {form: {str(n): enumerate_rows(n, form) for n in SPANS} for form in FORMS}}


def _dump(table: dict) -> str:
    """One row per line, so a diff of the table reads row by row."""

    lines = ["{", f' "head": {json.dumps(table["head"], sort_keys=True)},', ' "rows": {']
    forms = list(table["rows"].items())
    for fi, (form, by_n) in enumerate(forms):
        lines.append(f"  {json.dumps(form)}: {{")
        groups = list(by_n.items())
        for gi, (n, rows) in enumerate(groups):
            lines.append(f"   {json.dumps(n)}: [")
            lines += [f"    {json.dumps(r, sort_keys=True)}{',' if ri < len(rows) - 1 else ''}" for ri, r in enumerate(rows)]
            lines.append(f"   ]{',' if gi < len(groups) - 1 else ''}")
        lines.append(f"  }}{',' if fi < len(forms) - 1 else ''}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    text = _dump(build_table())
    if "--check" in sys.argv:
        if OUT.read_text(encoding="utf-8") != text:
            sys.exit(f"{OUT.name} is stale; regenerate with: uv run python scripts/t12_catalogue.py")
        print(f"{OUT.name} is current")
        return
    OUT.write_text(text, encoding="utf-8", newline="\n")
    table = json.loads(text)
    for form, by_n in table["rows"].items():
        print(form, {n: len(rows) for n, rows in by_n.items()})


if __name__ == "__main__":
    main()
