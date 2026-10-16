"""The selector: choose a lesson's review file from generated receipts under the per-file shortcut floor (design rule 3,
confirmed by R&D on 2026-10-11).

Each chosen item rests on one shortcut reading it proves (a reading that differs from the key on that item). The floor:
- at least `min_distinct` distinct readings rested on in the file (default 2);
- no reading rested on by more than `max_share` of the file (default 1/2, so at most 2 of 5);
- items resting on a reading that loosens `=` are at most `loosening_max_share` of the file (default 1/2; design: "at
  most half of a lesson's B items rely on loosening `=`").

A dated lesson (T12) also floors answer positions, read from each receipt's "position" (earliest, between, newest):
- no position holds more than `position_max_share` of the file (default 1/2);
- at least `min_newest_share` of the file, rounded up, has the newest edition as its answer (default 0.2: one of five;
  R&D on role-set#51).
The keep filter rejects more newest-edition answers than others, so without this floor "never the newest" was a working
shortcut (R&D on role-set#50, class filter_induced_shortcut). Receipts without a position skip this floor.

A T11 confirm-side item has no reading that differs (the subject-only reading agrees with a supported key). It rests on
the contrast that proves it, named `T11.contrast.<form of the changed line>`, so lookups can't fill the floor (R&D,
2026-10-11).

`distinct_skeletons` (on by default) keeps two items of one file off the same fact skeleton, the batch cap's rule applied
at selection (R&D's variety floor on role-set#52).

T12 claims (receipts with form "claim") are between `min_unsupported_share` (0.4) and `max_unsupported_share` (0.6)
unsupported within a file, the same 40/60 band as T11 (R&D's T12 claim ruling, and the cap from Argon's review as settled by
R&D). Claims record position None, so the position floor reads only which-edition items.

`file_receipt` records each chosen item's rests-on reading with the floor it was chosen under (R&D on role-set#51).

`count` sets what a reading's share counts. "rests_on" (the default) counts each item once, under the reading it rests on;
"differs" counts an item under every reading that differs on it, which is stricter. The floor is a configurable gate, and
R&D's ruling sets the default.

A count item rests on a reading of its trait clue's operator. A reading that differs only by rewriting some other count
clue does not test the trait (class trait_not_tested), so the item does not rest on it.

The search is deterministic: candidates are taken in the order given, and the first file that meets the floor is returned,
so the same receipts give the same file.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
    min_newest_share: float = 0.2
    distinct_skeletons: bool = True  # no two items in a file share a fact skeleton (R&D's variety floor on #52)
    min_unsupported_share: float = 0.4  # within a file's T12 claims (receipts with form "claim"; R&D's claim ruling)
    max_unsupported_share: float = 0.6  # and at most this, so neither verdict can be guessed (the 40/60 band)

    def cap(self, share: float) -> int:
        return int(share * self.size + 1e-9)

    @property
    def min_newest(self) -> int:
        return math.ceil(self.min_newest_share * self.size - 1e-9)


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


def _contrast(receipt: dict) -> list[str]:
    contrast = receipt.get("contrast")
    return [f"T11.contrast.{contrast['form']}"] if contrast else []


def rests(receipt: dict) -> list[str]:
    """What an item may rest on: a reading of its trait clue (`resting_readings`), and a T11 confirm item's contrast."""

    return resting_readings(receipt) + _contrast(receipt)


def lesson_of(name: str) -> str:
    return REGISTRY[name]["lesson"] if name in REGISTRY else name.split(".")[0]


def loosens_eq(name: str) -> bool:
    spec = REGISTRY.get(name, {})
    return spec.get("rewrite") == "count_op" and spec.get("from") == "="


def _lesson(pool: list[dict]) -> str:
    lessons = {lesson_of(n) for r in pool for n in rests(r)}
    if len(lessons) > 1:
        raise SelectError(f"a review file is chosen per lesson; this pool mixes {sorted(lessons)}")
    return lessons.pop() if lessons else ""


def _claims_balanced(claims: list[dict], floor: Floor) -> bool:
    unsupported = sum(r.get("answer") == "unsupported" for r in claims)
    return floor.min_unsupported_share * len(claims) - 1e-9 <= unsupported <= floor.max_unsupported_share * len(claims) + 1e-9


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
        if name not in differing(r) + _contrast(r):
            problems.append(f"an item rests on {name!r}, which does not differ from its key")
        elif name not in rests(r):
            problems.append(f"an item rests on {name!r}, which does not read its trait clue")
        for n in ([name] if floor.count == "rests_on" else differing(r) + _contrast(r)):
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
    claims = [r for r, _ in file if r.get("form") == "claim"]
    if claims and not _claims_balanced(claims, floor):
        unsupported = sum(r.get("answer") == "unsupported" for r in claims)
        problems.append(f"{unsupported} of {len(claims)} claims are unsupported; the band is {floor.min_unsupported_share:.0%} to "
                        f"{floor.max_unsupported_share:.0%}")
    skeletons = [r.get("skeleton_sha256") for r, _ in file if r.get("skeleton_sha256")]
    if floor.distinct_skeletons and len(set(skeletons)) < len(skeletons):
        problems.append(f"{len(skeletons) - len(set(skeletons))} item(s) repeat another item's fact skeleton")
    return problems


def select(pool: list[dict], floor: Floor = Floor()) -> list[Pick]:
    """The first file of `floor.size` receipts from `pool` (in pool order) that meets the floor, with the reading each
    rests on. Raises SelectError, saying which readings are short, when none does."""

    if floor.count not in COUNTS:
        raise SelectError(f"floor.count is one of {COUNTS}")
    _lesson(pool)
    options = [rests(r) for r in pool]
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
        sk = pool[i].get("skeleton_sha256")
        if floor.distinct_skeletons and sk and any(pool[p.index].get("skeleton_sha256") == sk for p in chosen):
            return False
        return not loosens_eq(name) or sum(loosens_eq(p.rests_on) for p in chosen) < lcap

    def search(start: int) -> bool:
        if len(chosen) == floor.size:
            newest = sum(pos[p.index] == "newest" for p in chosen)
            claims = [pool[p.index] for p in chosen if pool[p.index].get("form") == "claim"]
            return len({p.rests_on for p in chosen}) >= floor.min_distinct and (not dated or newest >= floor.min_newest) \
                and (not claims or _claims_balanced(claims, floor))
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


def file_receipt(pool: list[dict], picks: list[Pick], floor: Floor = Floor()) -> dict:
    """The record of a chosen file: the floor it was chosen under and, per item, what it rests on, its answer position,
    and the receipt fields that find it again (R&D on role-set#51)."""

    items = [{"pool_index": p.index, "rests_on": p.rests_on, "recipe": pool[p.index].get("recipe"),
              "seed": pool[p.index].get("seed"), "skeleton_sha256": pool[p.index].get("skeleton_sha256"),
              "position": pool[p.index].get("position")} for p in picks]
    return {"floor": asdict(floor), "items": items,
            "problems": check_file([(pool[p.index], p.rests_on) for p in picks], floor)}
