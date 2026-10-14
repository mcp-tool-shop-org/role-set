"""The selector: choose a lesson's review file from generated receipts under the per-file shortcut floor (design rule 3,
confirmed by R&D on 2026-10-11).

Each chosen item rests on one shortcut reading it proves (a reading that differs from the key on that item). The floor:
- at least `min_distinct` distinct readings rested on in the file (default 2);
- no reading rested on by more than `max_share` of the file (default 1/2, so at most 2 of 5);
- items resting on a reading that loosens `=` are at most `loosening_max_share` of the file (default 1/2; design: "at
  most half of a lesson's B items rely on loosening `=`").

A dated lesson (T12) also floors answer positions, read from each receipt's "position" (earliest, between, newest):
- no position holds more than `position_max_share` of the file (default 1/2);
- at least `min_newest` items have the newest edition as their answer (default 1).
The keep filter rejects more newest-edition answers than others, so without this floor "never the newest" was a working
shortcut (R&D on role-set#50, class filter_induced_shortcut). Receipts without a position skip this floor.

`count` sets what a reading's share counts. "rests_on" (the default) counts each item once, under the reading it rests on;
"differs" counts an item under every reading that differs on it, which is stricter. The floor is a configurable gate, and
R&D's ruling sets the default.

A count item rests on a reading of its trait clue's operator. A reading that differs only by rewriting some other count
clue does not test the trait (class trait_not_tested), so the item does not rest on it.

The search is deterministic: candidates are taken in the order given, and the first file that meets the floor is returned,
so the same receipts give the same file.
"""

from __future__ import annotations

from dataclasses import dataclass

from role_set.generate.shortcuts import REGISTRY

COUNTS = ("rests_on", "differs")


class SelectError(ValueError):
    """No file meets the floor. The message says what is short."""


@dataclass(frozen=True)
class Floor:
    size: int = 5
    min_distinct: int = 2
    max_share: float = 0.5
    loosening_max_share: float = 0.5
    count: str = "rests_on"
    position_max_share: float = 0.5
    min_newest: int = 1

    def cap(self, share: float) -> int:
        return int(share * self.size + 1e-9)


@dataclass(frozen=True)
class Pick:
    index: int  # the receipt's position in the pool
    rests_on: str


def differing(receipt: dict) -> list[str]:
    """The registered readings that differ from the key on this receipt, in registry order."""

    return [n for n in REGISTRY if receipt["shortcuts"].get(n, {}).get("differs")]


def resting_readings(receipt: dict) -> list[str]:
    """The readings an item may rest on. A count trait rests only on a reading of its own operator."""

    options = differing(receipt)
    trait = receipt.get("trait_clue") or []
    if not trait or trait[0] != "count":
        return options
    return [n for n in options if REGISTRY[n].get("from") == trait[3]]


def loosens_eq(name: str) -> bool:
    return REGISTRY[name].get("rewrite") == "count_op" and REGISTRY[name].get("from") == "="


def _lesson(pool: list[dict]) -> str:
    lessons = {REGISTRY[n]["lesson"] for r in pool for n in differing(r)}
    if len(lessons) > 1:
        raise SelectError(f"a review file is chosen per lesson; this pool mixes {sorted(lessons)}")
    return lessons.pop() if lessons else ""


def check_file(file: list[tuple[dict, str]], floor: Floor = Floor()) -> list[str]:
    """What a chosen file breaks, as messages (empty when it meets the floor). `file` pairs each receipt with the reading
    it rests on."""

    if floor.count not in COUNTS:
        raise SelectError(f"floor.count is one of {COUNTS}")
    problems = []
    if len(file) != floor.size:
        problems.append(f"the file has {len(file)} items; the floor is set for {floor.size}")
    tally: dict[str, int] = {}
    loose = 0
    for r, name in file:
        if name not in differing(r):
            problems.append(f"an item rests on {name!r}, which does not differ from its key")
        elif name not in resting_readings(r):
            problems.append(f"an item rests on {name!r}, which does not read its trait clue")
        for n in ([name] if floor.count == "rests_on" else differing(r)):
            tally[n] = tally.get(n, 0) + 1
        loose += loosens_eq(name)
    rested = {name for _, name in file}
    if len(rested) < floor.min_distinct:
        problems.append(f"{len(rested)} distinct reading(s) rested on; the floor is {floor.min_distinct}")
    cap = floor.cap(floor.max_share)
    for n, c in sorted(tally.items()):
        if c > cap:
            problems.append(f"{n} counts on {c} items; at most {cap} of {floor.size}")
    lcap = floor.cap(floor.loosening_max_share)
    if loose > lcap:
        problems.append(f"{loose} items rest on loosening '='; at most {lcap} of {floor.size}")
    positions = [r.get("position") for r, _ in file if r.get("position")]
    if positions:
        pcap = floor.cap(floor.position_max_share)
        for pos in sorted(set(positions)):
            if positions.count(pos) > pcap:
                problems.append(f"the answer is the {pos} edition on {positions.count(pos)} items; at most {pcap} of {floor.size}")
        if positions.count("newest") < floor.min_newest:
            problems.append(f"the answer is the newest edition on {positions.count('newest')} items; the floor is "
                            f"{floor.min_newest}")
    return problems


def select(pool: list[dict], floor: Floor = Floor()) -> list[Pick]:
    """The first file of `floor.size` receipts from `pool` (in pool order) that meets the floor, with the reading each
    rests on. Raises SelectError, saying which readings are short, when none does."""

    if floor.count not in COUNTS:
        raise SelectError(f"floor.count is one of {COUNTS}")
    _lesson(pool)
    options = [resting_readings(r) for r in pool]
    cap, lcap = floor.cap(floor.max_share), floor.cap(floor.loosening_max_share)
    supply: dict[str, int] = {}
    for opts in options:
        for n in opts:
            supply[n] = supply.get(n, 0) + 1
    usable = sum(min(c, cap) for c in supply.values())
    if usable < floor.size or len(supply) < floor.min_distinct:
        raise SelectError(f"the pool can't fill a file of {floor.size}: readings to rest on {dict(sorted(supply.items()))}, "
                          f"at most {cap} per reading and {floor.min_distinct} distinct; generate more seeds")
    pos = [r.get("position") for r in pool]
    dated = any(pos)
    pcap = floor.cap(floor.position_max_share)
    if dated and pos.count("newest") < floor.min_newest:
        raise SelectError(f"the pool has {pos.count('newest')} newest-edition answers and the floor is {floor.min_newest}; "
                          "build some with position=n - 1")

    chosen: list[Pick] = []
    tally: dict[str, int] = {}

    def fits(i: int, name: str) -> bool:
        counted = [name] if floor.count == "rests_on" else options[i]
        if any(tally.get(n, 0) >= cap for n in counted):
            return False
        if dated and pos[i] and sum(pos[p.index] == pos[i] for p in chosen) >= pcap:
            return False
        return not loosens_eq(name) or sum(loosens_eq(p.rests_on) for p in chosen) < lcap

    def search(start: int) -> bool:
        if len(chosen) == floor.size:
            newest = sum(pos[p.index] == "newest" for p in chosen)
            return len({p.rests_on for p in chosen}) >= floor.min_distinct and (not dated or newest >= floor.min_newest)
        left = floor.size - len(chosen)
        need = floor.min_newest - sum(pos[p.index] == "newest" for p in chosen) if dated else 0
        for i in range(start, len(pool) - left + 1):
            if need >= left and pos[i] != "newest":
                continue  # the slots left are owed to newest-edition answers
            for name in options[i]:
                if not fits(i, name):
                    continue
                counted = [name] if floor.count == "rests_on" else options[i]
                chosen.append(Pick(i, name))
                for n in counted:
                    tally[n] = tally.get(n, 0) + 1
                if search(i + 1):
                    return True
                chosen.pop()
                for n in counted:
                    tally[n] -= 1
        return False

    if not search(0):
        raise SelectError(f"no file of {floor.size} meets the floor (readings to rest on {dict(sorted(supply.items()))}, "
                          f"count={floor.count!r}); generate more seeds or loosen the floor in config")
    return list(chosen)
