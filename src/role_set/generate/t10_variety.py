"""T10's structural cells.

A cell is (how many entities, which count word, k, how many if clues). The usable cells are the ones that build, recorded
in t10_cells.json by scripts/t10_usable_cells.py. Set cells keep all three count words. Claim cells are exactly only.
Seeds 1-20 of one pack and question each take a different cell. Each pack starts one file further around that cycle, so
the files sit apart instead of one seed apart.

The builder refuses a finished puzzle only while it is still inside one seed's tries, and only when the reduced puzzle
did not keep its cell's if count. Nothing here throws a finished puzzle away because its skeleton has been seen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_DATA = Path(__file__).with_name("t10_cells.json")
_CELLS: dict | None = None
# One file is 20 seeds. The next pack starts this many places further along the cycle.
FILE = 20


def load_cells() -> dict:
    """The committed list: pack, question, then the cells that built within MAX_TRIES."""

    global _CELLS
    if _CELLS is None:
        _CELLS = json.loads(_DATA.read_text(encoding="utf-8"))
    return _CELLS


def _by_size_and_word(cells: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["op"], cell["n"]), []).append(cell)
    keys = sorted(groups)
    out: list[dict] = []
    while any(groups[key] for key in keys):
        for key in keys:
            if groups[key]:
                out.append(groups[key].pop(0))
    return out


def _spread(cells: list[dict]) -> list[dict]:
    """Deal if-counts in turn, and inside each if-count deal the count word and the size in turn. A stride then keeps
    both mixes in every file, including the locked seeds, instead of walking one block of the enumeration."""

    buckets = {ifs: [] for ifs in (2, 1, 0)}
    for cell in cells:
        buckets[cell["ifs"]].append(cell)
    dealt = {ifs: _by_size_and_word(bucket) for ifs, bucket in buckets.items()}
    cursors = {ifs: 0 for ifs in dealt}
    out: list[dict] = []
    moving = True
    while moving:
        moving = False
        for ifs in (2, 1, 0):
            cursor = cursors[ifs]
            if cursor < len(dealt[ifs]):
                out.append(dealt[ifs][cursor])
                cursors[ifs] = cursor + 1
                moving = True
    return out


def _stride(length: int) -> int:
    """A step coprime to the list, near a third of its length, so successive seeds jump rather than take a block."""

    target = max(1, length // 3)
    for step in range(target, length):
        if math.gcd(step, length) == 1:
            return step
    return 1


def cell_for(pack: str, question: str, seed: int) -> dict:
    """The cell for this pack, question and seed. Seeds 1-20 land on 20 different cells when at least 20 are usable."""

    table = load_cells()
    cells = _spread(table["cells"][pack][question])
    if len(cells) < 20:
        raise ValueError(f"{pack} {question} has {len(cells)} usable cells; a file of 20 seeds needs 20 different cells")
    order = table["packs"]
    step = _stride(len(cells))
    start = (order.index(pack) * FILE * step) % len(cells)
    return cells[(start + (seed - 1) * step) % len(cells)]


def canonical_skeleton(model: dict) -> str:
    """Entities and values renamed by first appearance in the clue list. Attribute shapes and the clue list stay, so two
    puzzles from different packs share a skeleton only when the clues have the same shape."""

    ents: dict[str, str] = {}
    vals: dict[tuple[str, str], str] = {}

    def ent(name: str) -> str:
        if name not in ents:
            ents[name] = f"e{len(ents) + 1}"
        return ents[name]

    def val(attr: str, value: str) -> str:
        key = (attr, value)
        if key not in vals:
            n = 1 + sum(1 for a, _v in vals if a == attr)
            vals[key] = f"v{n}"
        return vals[key]

    def rename(rule: list) -> list:
        form = rule[0]
        if form in ("is", "is_not"):
            return [form, ent(rule[1]), "a", val(rule[2], rule[3])]
        if form in ("in", "not_in"):
            return [form, ent(rule[1]), "a", [val(rule[2], v) for v in rule[3]]]
        if form == "count":
            return ["count", "a", val(rule[1], rule[2]), rule[3], rule[4]]
        if form == "if":
            return ["if", rename(rule[1]), rename(rule[2])]
        return rule

    clues = [rename(fact["rule"]) for fact in model["facts"]]
    shapes = [{"values": len(spec["values"]), "ordered": spec.get("ordered") is True,
               "distinct": spec.get("distinct", True) is not False}
              for spec in model["attributes"].values()]
    return json.dumps({"n": len(model["entities"]), "attributes": shapes, "clues": clues}, separators=(",", ":"))
