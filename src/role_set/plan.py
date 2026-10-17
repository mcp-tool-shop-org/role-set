"""The plan: uniqueness before writing.

One slot per item, fixed before any model call, so the floors are met by
construction and the writer is never asked to keep shares right. The same seed
reproduces the same plan. Nothing here talks to a model.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass, field

KINDS = ("A", "B", "C", "D", "E")
PER_KIND = 8
VERDICTS = ("supported", "unsupported", "cannot_tell")
CONTENT_KINDS = ("notice", "policy", "label", "timetable", "changelog", "code", "record", "design_note")
ORDERED_KINDS = ("changelog", "timetable")  # line order carries meaning; T09 apply slots never draw these
# Content kinds whose material naturally spans lines (a table row, a heading plus a bullet).
MULTI_LINE_KINDS = ("changelog", "label", "record", "code")
REAL_SOURCE_KINDS = ("changelog", "code", "record", "design_note")

# The B menu. Each structure appears at most once per lesson and at most twice per batch.
PUZZLE_STRUCTURES = (
    ("constraint_grid", False),
    ("scheduling_with_exclusions", False),
    ("elimination_from_a_set", False),
    ("underdetermined_with_explicit_set", False),
    ("stopping_problem", False),
    ("tree_with_pruning", True),
    ("falsifier_first", True),
    ("when_to_stop", True),
)
MIN_NON_VERIFICATION = 3

FLOOR_SHARE = 0.25  # cannot_tell, supported, and multi-line, of the verdict-bearing items
MIN_PAIRS = 2
# The ESCALATE key format the gates accept (aspire-si-gates ESC): a yes names the domain that needs a qualified person.
ESCALATE_DOMAINS = ("medical", "legal", "safety", "money")
ESCALATE_KEY = re.compile(r"^(no|yes \((medical|legal|safety|money)[^)]*\))$")
MIN_TWO_TURN = 2


class PlanError(ValueError):
    """A plan that would not meet its floors is never written."""


@dataclass
class Slot:
    id: str
    lesson: str
    kind: str
    side: str
    tier: int
    content_kind: str | None = None
    verdict: str | None = None
    escalate: str = "no"
    multi_line: bool = False
    pair: str | None = None
    two_turn: str | None = None  # "stands" or "changes"
    structure: str | None = None
    verification: bool | None = None
    real_source: bool = False
    e_type: str | None = None  # "written" or "verdict"
    driven_by: list[str] = field(default_factory=list)
    split: str | None = None  # "apply": asked after each session, never trained (aspire-si#127)
    domain: dict | None = None  # {"code", "title"[, "escalate"]}: the NAICS 2022 domain the item is set in (BUILD-SPEC B2)
    frame: str | None = None  # the framing's opening frame (role_set.frames), for C, D and verdict E
    pack: str | None = None  # kind B: the puzzle pack (role_set.solvefirst), distinct within a lesson (BUILD-SPEC B3)
    b_seed: int | None = None  # kind B: the seed the locked model is built from
    b_path: str | None = None  # kind B: "solve_first" (a locked model) or "freehand" (the writer writes the model)

    def as_dict(self) -> dict:
        return asdict(self)


def _verdict_bearing(slots: list[Slot]) -> list[Slot]:
    return [s for s in slots if s.kind in ("C", "D") or (s.kind == "E" and s.e_type == "verdict")]


def plan_lesson(lesson: str, sides: tuple[str, str], *, seed: int) -> list[Slot]:
    """Forty slots for one lesson: eight per kind, both sides four and four."""

    rng = random.Random(f"{lesson}:{seed}")
    slots: list[Slot] = []
    content = list(CONTENT_KINDS)
    rng.shuffle(content)
    tiers = [1, 1, 3, 3, 3, 4, 4, 5]

    for kind in KINDS:
        for n in range(PER_KIND):
            side = sides[n % 2]
            s = Slot(id=f"{lesson}-{kind}-{n + 1:02d}", lesson=lesson, kind=kind, side=side, tier=tiers[n])
            if kind == "A":
                s.tier = 1
            elif kind == "B":
                name, is_verification = PUZZLE_STRUCTURES[n]
                s.structure, s.verification = name, is_verification
            elif kind in ("C", "D"):
                s.content_kind = content[n]
            elif kind == "E":
                s.e_type = "verdict" if n >= PER_KIND - 2 else "written"
                s.tier = 1 if s.e_type == "written" else 3
                if s.e_type == "verdict":
                    s.content_kind = content[n]
            slots.append(s)

    bearing = _verdict_bearing(slots)  # 8 C + 8 D + 2 verdict E = 18
    # Verdicts: six each, so both floors (25%) hold with room, then shuffled by seed.
    verdicts = [v for v in VERDICTS for _ in range(len(bearing) // 3)]
    verdicts += [VERDICTS[0]] * (len(bearing) - len(verdicts))
    rng.shuffle(verdicts)
    for s, v in zip(bearing, verdicts):
        s.verdict = v

    # Two contrast pairs: a cannot_tell and an unsupported C or D item on the same statement.
    cd = [s for s in bearing if s.kind in ("C", "D")]
    silent = [s for s in cd if s.verdict == "cannot_tell"]
    contradicted = [s for s in cd if s.verdict == "unsupported"]
    for i in range(MIN_PAIRS):
        pid = f"{lesson.lower()}-pair-{i + 1}"
        silent[i].pair = pid
        contradicted[i].pair = pid

    # Two two-turn D items: one stands under pressure, one changes on new material.
    d_items = [s for s in slots if s.kind == "D"]
    d_items[0].two_turn = "stands"
    d_items[1].two_turn = "changes"

    # One escalate-yes item (advice), and one everyday cannot_tell with escalate no.
    decided = [s for s in cd if s.verdict != "cannot_tell" and s.pair is None]
    decided[0].escalate = f"yes ({random.Random(f'{lesson}:{seed}:escalate').choice(ESCALATE_DOMAINS)})"
    silent[-1].escalate = "no"

    # Multi-line keys on at least a quarter of the supported/unsupported items, where the material allows. The gates
    # count each item's final verdict, so a "changes" item may add a decided key its first turn didn't have.
    decided_all = [s for s in bearing if s.verdict != "cannot_tell"]
    need = _multi_line_need(decided_all, slots)
    candidates = [s for s in decided_all if s.content_kind in MULTI_LINE_KINDS]
    for s in (candidates + [s for s in decided_all if s not in candidates])[:need]:
        s.multi_line = True

    # At least four real-source C or D items per lesson.
    real = [s for s in cd if s.content_kind in REAL_SOURCE_KINDS][:4]
    for s in real:
        s.real_source = True

    check_plan(slots)
    return slots


def _multi_line_need(decided: list[Slot], slots: list[Slot]) -> int:
    """A quarter of the decided keys the gates may count: the decided items plus one per 'changes' item."""
    changes = sum(1 for s in slots if s.two_turn == "changes")
    return max(1, math.ceil((len(decided) + changes) * FLOOR_SHARE))


def check_plan(slots: list[Slot]) -> dict:
    """Refuse a plan that would not meet its floors. Returns the figures."""

    lessons = {s.lesson for s in slots}
    if len(lessons) != 1:
        raise PlanError("a plan is checked one lesson at a time")
    counts = {k: sum(1 for s in slots if s.kind == k) for k in KINDS}
    if any(c != PER_KIND for c in counts.values()):
        raise PlanError(f"kinds are not {PER_KIND} each: {counts}")
    bearing = _verdict_bearing(slots)
    n = len(bearing)
    ct = sum(1 for s in bearing if s.verdict == "cannot_tell")
    sup = sum(1 for s in bearing if s.verdict == "supported")
    if ct < n * FLOOR_SHARE or sup < n * FLOOR_SHARE:
        raise PlanError(f"floors: cannot_tell {ct} of {n}, supported {sup} of {n}")
    decided = [s for s in bearing if s.verdict != "cannot_tell"]
    ml = sum(1 for s in decided if s.multi_line)
    if ml < _multi_line_need(decided, slots):
        raise PlanError(f"multi-line share {ml} of {len(decided)} decided, plus any 'changes' item's final turn")
    pairs: dict[str, list[str]] = {}
    for s in slots:
        if s.pair:
            pairs.setdefault(s.pair, []).append(s.verdict or "")
    if len(pairs) < MIN_PAIRS or any(sorted(v) != ["cannot_tell", "unsupported"] for v in pairs.values()):
        raise PlanError(f"contrast pairs: {pairs}")
    two_turn = {s.two_turn for s in slots if s.two_turn}
    if two_turn != {"stands", "changes"}:
        raise PlanError(f"two-turn D items: {two_turn}")
    bad = sorted({s.escalate for s in slots if not ESCALATE_KEY.match(s.escalate)})
    if bad:
        raise PlanError(f"escalate keys {bad}: only no, or yes (medical|legal|safety|money ...), as the gates read them")
    if not any(s.escalate.startswith("yes (") for s in bearing):
        raise PlanError("no ESCALATE yes item")
    if not any(s.verdict == "cannot_tell" and s.escalate == "no" for s in bearing):
        raise PlanError("no everyday cannot_tell with ESCALATE no")
    puzzles = [s for s in slots if s.kind == "B"]
    structures = [s.structure for s in puzzles]
    if len(set(structures)) != len(structures):
        raise PlanError("a puzzle structure repeats within the lesson")
    if sum(1 for s in puzzles if s.verification is False) < MIN_NON_VERIFICATION:
        raise PlanError("fewer than three non-verification puzzles")
    real = sum(1 for s in slots if s.real_source)
    if real < 4:
        raise PlanError(f"real-source items {real} of 4")
    return {
        "lesson": next(iter(lessons)),
        "items": len(slots),
        "verdict_bearing": n,
        "cannot_tell": ct,
        "supported": sup,
        "multi_line": ml,
        "pairs": len(pairs),
        "escalate_yes": sum(1 for s in bearing if s.escalate.startswith("yes (")),
        "real_source": real,
    }


def plan_batch(lessons: dict[str, tuple[str, str]], *, seed: int) -> list[Slot]:
    """Several lessons, then the batch-level rule: no puzzle structure more than twice."""

    slots: list[Slot] = []
    for lesson, sides in lessons.items():
        slots.extend(plan_lesson(lesson, sides, seed=seed))
    seen: dict[str, int] = {}
    for s in slots:
        if s.kind == "B":
            seen[s.structure or ""] = seen.get(s.structure or "", 0) + 1
    if len(lessons) > 2 and any(c > 2 for c in seen.values()):
        # Rotate the menu for the third lesson onward so no structure exceeds two per batch.
        names = [name for name, _ in PUZZLE_STRUCTURES]
        for i, lesson in enumerate(list(lessons)[2:], start=2):
            for s in (x for x in slots if x.lesson == lesson and x.kind == "B"):
                idx = (names.index(s.structure) + i) % len(names)
                s.structure, s.verification = PUZZLE_STRUCTURES[idx]
        seen = {}
        for s in slots:
            if s.kind == "B":
                seen[s.structure or ""] = seen.get(s.structure or "", 0) + 1
    # Eight structures, eight puzzles per lesson: every structure appears once per lesson, so
    # a batch of n lessons has n of each. The gate's "at most two per batch" holds up to two lessons;
    # beyond that the menu must grow. Report rather than hide it.
    over = {k: v for k, v in seen.items() if v > 2}
    if over:
        raise PlanError(f"puzzle structures over two per batch: {over}; the menu needs more structures")
    return slots


# ---- apply items (run sheet B6; the schema in aspire-si#127, R&D's rulings on #128) --------------------------------
APPLY_PER_KIND = 5
APPLY_TIERS = (1, 3, 3, 4, 5)


def plan_apply(lesson: str, sides: tuple[str, str], *, seed: int) -> list[Slot]:
    """Ten apply slots for one lesson: five C and five D, one turn each, ids <lesson>-<kind>-AP-NN, split "apply". The
    model answers them after each session and is never trained on them, so they carry no pairs and no two-turn items.
    Verdicts four supported, three unsupported, three cannot_tell; one ESCALATE-yes; multi-line keys on a quarter of
    the decided items, where the material allows. T09 draws only unordered kinds: a changelog or a timetable has a
    line order that carries meaning, and assembly refuses order_free on those."""

    rng = random.Random(f"{lesson}:apply:{seed}")
    pool = CONTENT_KINDS if lesson != "T09" else tuple(k for k in CONTENT_KINDS if k not in ORDERED_KINDS)
    content = list(pool)
    rng.shuffle(content)
    slots = [Slot(id=f"{lesson}-{kind}-AP-{n + 1:02d}", lesson=lesson, kind=kind, side=sides[n % 2], tier=APPLY_TIERS[n],
                  content_kind=content[(n + (0 if kind == "C" else APPLY_PER_KIND)) % len(content)], split="apply")
             for kind in ("C", "D") for n in range(APPLY_PER_KIND)]
    verdicts = ["supported"] * 4 + ["unsupported"] * 3 + ["cannot_tell"] * 3
    rng.shuffle(verdicts)
    for s, v in zip(slots, verdicts):
        s.verdict = v
    decided = [s for s in slots if s.verdict != "cannot_tell"]
    decided[0].escalate = f"yes ({random.Random(f'{lesson}:apply:{seed}:escalate').choice(ESCALATE_DOMAINS)})"
    need = math.ceil(FLOOR_SHARE * len(decided))
    candidates = [s for s in decided if s.content_kind in MULTI_LINE_KINDS]
    for s in (candidates + [s for s in decided if s not in candidates])[:need]:
        s.multi_line = True
    return slots


def plan_apply_batch(lessons: dict[str, tuple[str, str]], *, seed: int) -> list[Slot]:
    return [s for name, sides in lessons.items() for s in plan_apply(name, sides, seed=seed)]

