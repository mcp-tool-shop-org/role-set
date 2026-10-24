"""The plan: uniqueness before writing.

One slot per item, fixed before any model call, so the floors are met by
construction and the writer is never asked to keep shares right. The same seed
reproduces the same plan. Nothing here talks to a model.
"""

from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import asdict, dataclass, field

from role_set.gates import load_gates

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
SIDE_PAIRING_SETTINGS = ("default", "verdict_safe")
# Sides that assert the statement matches the material: never paired with a non-supported verdict under verdict_safe (R&D item 20).
MATCHING_ONLY_SIDES = ("an equivalent wording still matches",)


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
    b_skeleton: str | None = None  # kind B, solve-first: the locked model's fact skeleton (hashed), fixed at plan time
    b_reseeds: int = 0  # kind B, solve-first: how often the plan re-seeded the slot to avoid a repeated skeleton
    buffer: bool = False  # training or apply buffer slot (Argon round 2, F2)
    side_pairing: str = "default"  # "default" or "verdict_safe" (item 20)

    def as_dict(self) -> dict:
        return asdict(self)


def _verdict_bearing(slots: list[Slot]) -> list[Slot]:
    return [s for s in slots if s.kind in ("C", "D") or (s.kind == "E" and s.e_type == "verdict")]


def _align_verdict_sides(slots: list[Slot], primary_side: str, *, side_pairing: str = "default") -> None:
    """Under `verdict_safe`, never pair a non-supported verdict with a side that asserts the statement matches."""
    for s in slots:
        s.side_pairing = side_pairing
        if side_pairing == "verdict_safe" and s.verdict is not None and s.verdict != "supported" and s.side in MATCHING_ONLY_SIDES:
            s.side = primary_side


def _assign_escalate_and_multi_line(slots: list[Slot], esc_domain: str) -> None:
    base = [s for s in slots if not s.buffer and s.split != "apply"]
    bearing = _verdict_bearing(base)
    cd = [s for s in bearing if s.kind in ("C", "D")]
    for s in bearing:
        s.escalate = "no"
        s.multi_line = False
    silent = [s for s in cd if s.verdict == "cannot_tell"]
    decided = [s for s in cd if s.verdict != "cannot_tell" and s.pair is None]
    if decided:
        decided[0].escalate = f"yes ({esc_domain})"
    if silent:
        silent[-1].escalate = "no"
    decided_all = [s for s in bearing if s.verdict != "cannot_tell"]
    need = _multi_line_need(decided_all, base)
    candidates = [s for s in decided_all if s.content_kind in MULTI_LINE_KINDS]
    for s in (candidates + [s for s in decided_all if s not in candidates])[:need]:
        s.multi_line = True


def _resolve_all_silent_real_c(
    lesson: str,
    seed: int,
    slots: list[Slot],
    esc_domain: str,
    primary_side: str = "",
    *,
    side_pairing: str = "default",
) -> None:
    """Rare case (Argon round 3, F1): if all 4 real-source C slots drew cannot_tell, swap the first with the
    verdict of the first supported non-C verdict-bearing slot (D or verdict E) for which the floors still hold."""

    base = [s for s in slots if not s.buffer and s.split != "apply"]
    real_c = [s for s in base if s.kind == "C" and s.real_source]
    if not real_c or not all(s.verdict == "cannot_tell" for s in real_c):
        return
    bearing = _verdict_bearing(base)
    p_side = primary_side or (base[0].side if base else "")
    for cand in bearing:
        if cand.kind != "C" and cand.verdict == "supported":
            old_r_side, old_c_side = real_c[0].side, cand.side
            real_c[0].verdict, cand.verdict = cand.verdict, real_c[0].verdict
            _align_verdict_sides(base, p_side, side_pairing=side_pairing)
            _assign_escalate_and_multi_line(base, esc_domain)
            try:
                check_plan(base, side_pairing=side_pairing)
                return
            except PlanError:
                real_c[0].verdict, cand.verdict = cand.verdict, real_c[0].verdict
                real_c[0].side, cand.side = old_r_side, old_c_side
    raise PlanError(
        f"REFUSED: {lesson} (seed {seed}): all 4 real-source C slots drew cannot_tell and no supported non-C "
        "verdict-bearing slot preserves the floors"
    )


def plan_lesson(
    lesson: str,
    sides: tuple[str, str],
    *,
    seed: int,
    buffers: bool = False,
    side_pairing: str = "default",
) -> list[Slot]:
    """Forty slots for one lesson: eight per kind, both sides four and four. With `buffers=True`, adds the 8 training
    buffers and deals verdicts under the buffered rule (Argon round 2 F2, round 3 F1)."""

    if side_pairing not in SIDE_PAIRING_SETTINGS:
        raise PlanError(f"REFUSED: side_pairing must be one of {SIDE_PAIRING_SETTINGS}, got {side_pairing!r}")
    rng = random.Random(f"{lesson}:{seed}")
    slots: list[Slot] = []
    content = list(CONTENT_KINDS)
    rng.shuffle(content)
    tiers = [1, 1, 3, 3, 3, 4, 4, 5]

    for kind in KINDS:
        for n in range(PER_KIND):
            side = sides[n % 2]
            s = Slot(
                id=f"{lesson}-{kind}-{n + 1:02d}",
                lesson=lesson,
                kind=kind,
                side=side,
                tier=tiers[n],
                side_pairing=side_pairing,
            )
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
    cd = [s for s in bearing if s.kind in ("C", "D")]
    esc_domain = random.Random(f"{lesson}:{seed}:escalate").choice(ESCALATE_DOMAINS)

    if buffers:
        non_real_c = [s for s in slots if s.kind == "C" and s.content_kind not in REAL_SOURCE_KINDS]
        real_c = [s for s in slots if s.kind == "C" and s.content_kind in REAL_SOURCE_KINDS]
        c_verdicts = ["cannot_tell", "cannot_tell", "unsupported", "unsupported"]
        rng.shuffle(c_verdicts)
        for s, v in zip(non_real_c, c_verdicts):
            s.verdict = v
        other_bearing = [s for s in bearing if s not in non_real_c]
        rest_verdicts = ["supported"] * 6 + ["unsupported"] * 4 + ["cannot_tell"] * 4
        rng.shuffle(rest_verdicts)
        for s, v in zip(other_bearing, rest_verdicts):
            s.verdict = v

        silent = [s for s in non_real_c if s.verdict == "cannot_tell"]
        contradicted = [s for s in non_real_c if s.verdict == "unsupported"]
        for i in range(MIN_PAIRS):
            pid = f"{lesson.lower()}-pair-{i + 1}"
            silent[i].pair = pid
            contradicted[i].pair = pid

        d_items = [s for s in slots if s.kind == "D" and s.pair is None]
        d_items[0].two_turn = "stands"
        d_items[1].two_turn = "changes"

        for s in real_c:
            s.real_source = True

        _align_verdict_sides(slots, sides[0], side_pairing=side_pairing)
        _assign_escalate_and_multi_line(slots, esc_domain)
        _resolve_all_silent_real_c(lesson, seed, slots, esc_domain, sides[0], side_pairing=side_pairing)

        p3 = f"{lesson.lower()}-pair-3"
        slots.extend([
            Slot(id=f"{lesson}-A-09", lesson=lesson, kind="A", side=sides[0], tier=1, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-E-09", lesson=lesson, kind="E", e_type="written", side=sides[0], tier=1, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-C-09", lesson=lesson, kind="C", side=sides[0], tier=3, content_kind="notice",
                 verdict="cannot_tell", escalate="no", multi_line=False, pair=p3, real_source=False, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-C-10", lesson=lesson, kind="C", side=sides[1], tier=3, content_kind="label",
                 verdict="unsupported", escalate="no", multi_line=True, pair=p3, real_source=False, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-C-11", lesson=lesson, kind="C", side=sides[0], tier=4, content_kind="changelog",
                 verdict="supported", escalate="yes (safety)", multi_line=True, real_source=True, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-C-12", lesson=lesson, kind="C", side=sides[1], tier=4, content_kind="code",
                 verdict="supported", escalate="no", multi_line=True, real_source=True, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-D-09", lesson=lesson, kind="D", side=sides[0], tier=3, content_kind="label",
                 two_turn="stands", verdict="supported", escalate="yes (medical)", multi_line=True, real_source=False, buffer=True, side_pairing=side_pairing),
            Slot(id=f"{lesson}-D-10", lesson=lesson, kind="D", side=sides[1], tier=3, content_kind="notice",
                 two_turn="changes", verdict="cannot_tell", escalate="no", multi_line=False, real_source=False, buffer=True, side_pairing=side_pairing),
        ])
        _align_verdict_sides(slots, sides[0], side_pairing=side_pairing)
    else:
        # Verdicts: six each, so both floors (25%) hold with room, then shuffled by seed.
        verdicts = [v for v in VERDICTS for _ in range(len(bearing) // 3)]
        verdicts += [VERDICTS[0]] * (len(bearing) - len(verdicts))
        rng.shuffle(verdicts)
        for s, v in zip(bearing, verdicts):
            s.verdict = v

        # Two contrast pairs: a cannot_tell and an unsupported C or D item on the same statement.
        silent = [s for s in cd if s.verdict == "cannot_tell"]
        contradicted = [s for s in cd if s.verdict == "unsupported"]
        for i in range(MIN_PAIRS):
            pid = f"{lesson.lower()}-pair-{i + 1}"
            silent[i].pair = pid
            contradicted[i].pair = pid

        # At least four real-source C or D items per lesson, skipping contrast pairs.
        real = [s for s in cd if s.content_kind in REAL_SOURCE_KINDS and s.pair is None][:4]
        for s in real:
            s.real_source = True

        # Two two-turn D items: one stands under pressure, one changes on new material (skipping contrast pairs and real-source items).
        d_items = [s for s in slots if s.kind == "D" and s.pair is None and not s.real_source]
        d_items[0].two_turn = "stands"
        d_items[1].two_turn = "changes"

        _align_verdict_sides(slots, sides[0], side_pairing=side_pairing)
        _assign_escalate_and_multi_line(slots, esc_domain)

    check_plan(slots, side_pairing=side_pairing)
    return slots


def _multi_line_need(decided: list[Slot], slots: list[Slot]) -> int:
    """A quarter of the decided keys the gates may count: the decided items plus one per 'changes' item."""
    changes = sum(1 for s in slots if s.two_turn == "changes")
    return max(1, math.ceil((len(decided) + changes) * FLOOR_SHARE))


def _check_apply_plan(apply_slots: list[Slot], *, side_pairing: str | None = None) -> None:
    base = [s for s in apply_slots if not s.buffer]
    buf = [s for s in apply_slots if s.buffer]
    if buf and any(s.kind not in ("C", "D") for s in buf):
        raise PlanError("apply buffers may only be kind C or D")
    counts = {k: sum(1 for s in base if s.kind == k) for k in ("C", "D")}
    if any(c != APPLY_PER_KIND for c in counts.values()):
        raise PlanError(f"apply kinds are not {APPLY_PER_KIND} each: {counts}")
    if any(s.pair or s.two_turn for s in base):
        raise PlanError("apply items carry no pairs and no two-turn items")
    n = len(base)
    ct = sum(1 for s in base if s.verdict == "cannot_tell")
    sup = sum(1 for s in base if s.verdict == "supported")
    if ct < n * FLOOR_SHARE or sup < n * FLOOR_SHARE:
        raise PlanError(f"apply floors: cannot_tell {ct} of {n}, supported {sup} of {n}")
    decided = [s for s in base if s.verdict != "cannot_tell"]
    ml = sum(1 for s in decided if s.multi_line)
    if ml < math.ceil(FLOOR_SHARE * len(decided)):
        raise PlanError(f"apply multi-line share {ml} of {len(decided)} decided")
    bad = sorted({s.escalate for s in apply_slots if not ESCALATE_KEY.match(s.escalate)})
    if bad:
        raise PlanError(f"escalate keys {bad}: only no, or yes (medical|legal|safety|money ...), as the gates read them")
    if not any(s.escalate.startswith("yes (") for s in base):
        raise PlanError("apply: no ESCALATE yes item")
    if not any(s.verdict == "cannot_tell" and s.escalate == "no" for s in base):
        raise PlanError("apply: no everyday cannot_tell with ESCALATE no")
    bad_sides = [
        s.id
        for s in apply_slots
        if (side_pairing == "verdict_safe" or s.side_pairing == "verdict_safe")
        and s.verdict is not None
        and s.verdict != "supported"
        and s.side in MATCHING_ONLY_SIDES
    ]
    if bad_sides:
        raise PlanError(f"non-supported verdict paired with matching-only side {MATCHING_ONLY_SIDES}: {', '.join(bad_sides)}")


def check_plan(slots: list[Slot], *, side_pairing: str | None = None) -> dict:
    """Refuse a plan that would not meet its floors. Accepts buffer slots only alongside the exact base counts.
    Returns the figures."""

    if side_pairing is not None and side_pairing not in SIDE_PAIRING_SETTINGS:
        raise PlanError(f"REFUSED: side_pairing must be one of {SIDE_PAIRING_SETTINGS}, got {side_pairing!r}")
    lessons = {s.lesson for s in slots}
    if len(lessons) != 1:
        raise PlanError("a plan is checked one lesson at a time")
    train = [s for s in slots if s.split != "apply"]
    apply_slots = [s for s in slots if s.split == "apply"]
    if apply_slots:
        _check_apply_plan(apply_slots, side_pairing=side_pairing)
    if not train:
        return {"lesson": next(iter(lessons)), "items": len(apply_slots)}
    base = [s for s in train if not s.buffer]
    buf = [s for s in train if s.buffer]
    if buf:
        if any(s.kind == "B" or (s.kind == "E" and s.e_type == "verdict") for s in buf):
            raise PlanError("kind B and verdict E have no buffers")
    counts = {k: sum(1 for s in base if s.kind == k) for k in KINDS}
    if any(c != PER_KIND for c in counts.values()):
        raise PlanError(f"kinds are not {PER_KIND} each: {counts}")
    e_written = sum(1 for s in base if s.kind == "E" and s.e_type == "written")
    e_verdict = sum(1 for s in base if s.kind == "E" and s.e_type == "verdict")
    if e_written != 6 or e_verdict != 2:
        raise PlanError(f"kind E must be 6 written and 2 verdict, got {e_written} written and {e_verdict} verdict")
    bearing = _verdict_bearing(base)
    n = len(bearing)
    ct = sum(1 for s in bearing if s.verdict == "cannot_tell")
    sup = sum(1 for s in bearing if s.verdict == "supported")
    if ct < n * FLOOR_SHARE or sup < n * FLOOR_SHARE:
        raise PlanError(f"floors: cannot_tell {ct} of {n}, supported {sup} of {n}")
    decided = [s for s in bearing if s.verdict != "cannot_tell"]
    ml = sum(1 for s in decided if s.multi_line)
    if ml < _multi_line_need(decided, base):
        raise PlanError(f"multi-line share {ml} of {len(decided)} decided, plus any 'changes' item's final turn")
    pairs: dict[str, list[str]] = {}
    for s in base:
        if s.pair:
            pairs.setdefault(s.pair, []).append(s.verdict or "")
    if len(pairs) < MIN_PAIRS or any(sorted(v) != ["cannot_tell", "unsupported"] for v in pairs.values()):
        raise PlanError(f"contrast pairs: {pairs}")
    if any(s.pair and s.two_turn for s in base):
        raise PlanError("a two-turn D item cannot be part of a contrast pair")
    two_turn = {s.two_turn for s in base if s.two_turn}
    if two_turn != {"stands", "changes"}:
        raise PlanError(f"two-turn D items: {two_turn}")
    bad = sorted({s.escalate for s in slots if not ESCALATE_KEY.match(s.escalate)})
    if bad:
        raise PlanError(f"escalate keys {bad}: only no, or yes (medical|legal|safety|money ...), as the gates read them")
    if not any(s.escalate.startswith("yes (") for s in bearing):
        raise PlanError("no ESCALATE yes item")
    if not any(s.verdict == "cannot_tell" and s.escalate == "no" for s in bearing):
        raise PlanError("no everyday cannot_tell with ESCALATE no")
    puzzles = [s for s in base if s.kind == "B"]
    structures = [s.structure for s in puzzles]
    if len(set(structures)) != len(structures):
        raise PlanError("a puzzle structure repeats within the lesson")
    if sum(1 for s in puzzles if s.verification is False) < MIN_NON_VERIFICATION:
        raise PlanError("fewer than three non-verification puzzles")
    bad_real = [s.id for s in slots if s.real_source and (s.pair or s.two_turn or s.content_kind not in REAL_SOURCE_KINDS)]
    if bad_real:
        raise PlanError(
            f"a real-source item cannot be paired, two-turn, or outside REAL_SOURCE_KINDS: {', '.join(bad_real)}"
        )
    real = sum(1 for s in base if s.real_source)
    if real < 4:
        raise PlanError(f"real-source items {real} of 4")
    bad_sides = [
        s.id
        for s in slots
        if (side_pairing == "verdict_safe" or s.side_pairing == "verdict_safe")
        and s.verdict is not None
        and s.verdict != "supported"
        and s.side in MATCHING_ONLY_SIDES
    ]
    if bad_sides:
        raise PlanError(f"non-supported verdict paired with matching-only side {MATCHING_ONLY_SIDES}: {', '.join(bad_sides)}")
    return {
        "lesson": next(iter(lessons)),
        "items": len(base),
        "verdict_bearing": n,
        "cannot_tell": ct,
        "supported": sup,
        "multi_line": ml,
        "pairs": len(pairs),
        "escalate_yes": sum(1 for s in bearing if s.escalate.startswith("yes (")),
        "real_source": real,
    }


def plan_batch(
    lessons: dict[str, tuple[str, str]],
    *,
    seed: int,
    buffers: bool = False,
    side_pairing: str = "default",
) -> list[Slot]:
    """Several lessons, then the batch-level rule: no puzzle structure more than twice."""

    slots: list[Slot] = []
    for lesson, sides in lessons.items():
        slots.extend(plan_lesson(lesson, sides, seed=seed, buffers=buffers, side_pairing=side_pairing))
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


def plan_apply(
    lesson: str,
    sides: tuple[str, str],
    *,
    seed: int,
    buffers: bool = False,
    side_pairing: str = "default",
) -> list[Slot]:
    """Ten apply slots for one lesson: five C and five D, one turn each, ids <lesson>-<kind>-AP-NN, split "apply". The
    model answers them after each session and is never trained on them, so they carry no pairs and no two-turn items.
    Verdicts four supported, three unsupported, three cannot_tell; one ESCALATE-yes; multi-line keys on a quarter of
    the decided items, where the material allows. T09 draws only unordered kinds: a changelog or a timetable has a
    line order that carries meaning, and assembly refuses order_free on those. With `buffers=True`, deals 3 supported +
    2 unsupported to C and 1 supported + 1 unsupported + 3 cannot_tell to D, sets multi_line on 2 decided C slots, and
    adds the 2 apply buffers."""

    if side_pairing not in SIDE_PAIRING_SETTINGS:
        raise PlanError(f"REFUSED: side_pairing must be one of {SIDE_PAIRING_SETTINGS}, got {side_pairing!r}")
    rng = random.Random(f"{lesson}:apply:{seed}")
    pool = CONTENT_KINDS if lesson != "T09" else tuple(k for k in CONTENT_KINDS if k not in ORDERED_KINDS)
    content = list(pool)
    rng.shuffle(content)
    slots = [Slot(id=f"{lesson}-{kind}-AP-{n + 1:02d}", lesson=lesson, kind=kind, side=sides[n % 2], tier=APPLY_TIERS[n],
                  content_kind=content[(n + (0 if kind == "C" else APPLY_PER_KIND)) % len(content)], split="apply",
                  side_pairing=side_pairing)
             for kind in ("C", "D") for n in range(APPLY_PER_KIND)]
    esc_domain = random.Random(f"{lesson}:apply:{seed}:escalate").choice(ESCALATE_DOMAINS)
    if buffers:
        c_slots = [s for s in slots if s.kind == "C"]
        d_slots = [s for s in slots if s.kind == "D"]
        c_verdicts = ["supported"] * 3 + ["unsupported"] * 2
        d_verdicts = ["supported", "unsupported"] + ["cannot_tell"] * 3
        rng.shuffle(c_verdicts)
        rng.shuffle(d_verdicts)
        for s, v in zip(c_slots, c_verdicts):
            s.verdict = v
        for s, v in zip(d_slots, d_verdicts):
            s.verdict = v
        decided = [s for s in slots if s.verdict != "cannot_tell"]
        decided[0].escalate = f"yes ({esc_domain})"
        decided_c = [s for s in c_slots if s.verdict != "cannot_tell"]
        c_ml = [s for s in decided_c if s.content_kind in MULTI_LINE_KINDS]
        for s in (c_ml + [s for s in decided_c if s not in c_ml])[:2]:
            s.multi_line = True
        slots.extend([
            Slot(id=f"{lesson}-C-AP-06", lesson=lesson, kind="C", side=sides[0], tier=3, content_kind="label",
                 verdict="supported", escalate="yes (safety)", multi_line=True, split="apply", buffer=True,
                 side_pairing=side_pairing),
            Slot(id=f"{lesson}-D-AP-06", lesson=lesson, kind="D", side=sides[1], tier=3, content_kind="notice",
                 verdict="cannot_tell", escalate="no", multi_line=False, split="apply", buffer=True,
                 side_pairing=side_pairing),
        ])
    else:
        verdicts = ["supported"] * 4 + ["unsupported"] * 3 + ["cannot_tell"] * 3
        rng.shuffle(verdicts)
        for s, v in zip(slots, verdicts):
            s.verdict = v
        decided = [s for s in slots if s.verdict != "cannot_tell"]
        decided[0].escalate = f"yes ({esc_domain})"
        need = math.ceil(FLOOR_SHARE * len(decided))
        candidates = [s for s in decided if s.content_kind in MULTI_LINE_KINDS]
        for s in (candidates + [s for s in decided if s not in candidates])[:need]:
            s.multi_line = True
    _align_verdict_sides(slots, sides[0], side_pairing=side_pairing)
    return slots


def plan_apply_batch(
    lessons: dict[str, tuple[str, str]],
    *,
    seed: int,
    buffers: bool = False,
    side_pairing: str = "default",
) -> list[Slot]:
    return [
        s
        for name, sides in lessons.items()
        for s in plan_apply(name, sides, seed=seed, buffers=buffers, side_pairing=side_pairing)
    ]


def select_and_trim(
    rows: list[dict],
    plan: list[Slot] | list[dict] | dict,
    excluded_ids: set[str] | list[str] | tuple[str, ...],
    *,
    gates=None,
    models: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Deterministic post-gold-check trim of a buffered batch back to the base plan's counts (40 training, 10 apply).
    Returns `(trimmed_rows, substitutions)`."""

    raw_slots = plan["slots"] if isinstance(plan, dict) else plan
    slots: list[Slot] = [s if isinstance(s, Slot) else Slot(**s) for s in raw_slots]
    excluded = set(excluded_ids or ())
    slot_by_id = {s.id: s for s in slots}
    unknown = sorted(excluded - set(slot_by_id))
    if unknown:
        raise PlanError(f"REFUSED: unknown slot id(s) in excluded_ids: {', '.join(unknown)}")
    row_by_id = {r["id"]: r for r in rows}

    base = [copy.copy(s) for s in slots if not s.buffer]
    for s in base:
        if s.id in excluded and (s.kind == "B" or (s.kind == "E" and s.e_type == "verdict")):
            label = f"{s.kind}:{s.e_type}" if s.e_type else s.kind
            raise PlanError(f"REFUSED: {s.id} ({label}) has no buffer; an unbuffered slot cannot be excluded")

    surviving_bufs = [copy.copy(s) for s in slots if s.buffer and s.id not in excluded and s.id in row_by_id]
    plan_idx = {s.id: i for i, s in enumerate(slots)}
    used_bufs: set[str] = set()
    replaced_slots: dict[str, Slot] = {}
    substitutions: list[dict] = []

    # 1. A broken base pair is replaced whole by the buffer pair (t10-pair-3).
    base_pairs: dict[str, list[Slot]] = {}
    for s in base:
        if s.pair:
            base_pairs.setdefault(s.pair, []).append(s)
    for pid, members in base_pairs.items():
        if any(m.id in excluded for m in members):
            pair_bufs = [b for b in surviving_bufs if b.pair and b.id not in used_bufs]
            b_ct = next((b for b in pair_bufs if b.verdict == "cannot_tell"), None)
            b_un = next((b for b in pair_bufs if b.verdict == "unsupported"), None)
            if b_ct is None or b_un is None:
                raise PlanError(f"REFUSED: broken base pair {pid!r} cannot be replaced: buffer pair is unavailable")
            for m in members:
                chosen = copy.copy(b_ct if m.verdict == "cannot_tell" else b_un)
                chosen.buffer = False
                replaced_slots[m.id] = chosen
                used_bufs.add(chosen.id)
                substitutions.append({
                    "excluded_id": m.id,
                    "buffer_id": chosen.id,
                    "kind": m.kind,
                    "split": m.split or "train",
                    "reason": "broken_pair" if m.id in excluded else "broken_pair_partner",
                    "pair_dropped": False,
                })

    # 2. Every other excluded base slot, in plan order.
    others = [s for s in base if s.id in excluded and s.id not in replaced_slots]
    has_turns = any("turns" in r for r in rows)
    resolved_gates = gates
    if resolved_gates is None and has_turns:
        resolved_gates = load_gates()

    def _build_rows(rep_map: dict[str, tuple[Slot, bool]]) -> list[dict]:
        out_rows: list[dict] = []
        for s in base:
            if s.id in rep_map:
                chosen_slot, drop_pair = rep_map[s.id]
                r = copy.deepcopy(row_by_id[chosen_slot.id])
                if drop_pair:
                    r.pop("pair", None)
                out_rows.append(r)
            else:
                out_rows.append(copy.deepcopy(row_by_id[s.id]))
        return out_rows

    def _valid_batch(rep_map: dict[str, tuple[Slot, bool]]) -> bool:
        trial_slots = [rep_map[s.id][0] if s.id in rep_map else s for s in base]
        try:
            check_plan(trial_slots)
        except PlanError:
            return False
        if resolved_gates and has_turns:
            trial_rows = [r for r in _build_rows(rep_map) if r.get("split") != "apply"]
            if trial_rows:
                row_models = dict(models) if models else {}
                for r in trial_rows:
                    if r.get("kind") == "B" and r["id"] not in row_models and "model" in r:
                        row_models[r["id"]] = r["model"]
                try:
                    if all(r["id"] in row_models for r in trial_rows if r.get("kind") == "B"):
                        resolved_gates.check_batch(trial_rows, models=row_models)
                    else:
                        resolved_gates.module.check_batch(trial_rows, models=row_models or None)
                except resolved_gates.error_type():
                    return False
        return True

    initial_map: dict[str, tuple[Slot, bool]] = {k: (v, False) for k, v in replaced_slots.items()}

    def _search(idx: int, current_map: dict[str, tuple[Slot, bool]], current_used: set[str],
                current_subs: list[dict]) -> tuple[dict[str, tuple[Slot, bool]], list[dict]] | None:
        if idx == len(others):
            return (current_map, current_subs) if _valid_batch(current_map) else None
        target = others[idx]
        cands = [
            b for b in surviving_bufs
            if b.id not in current_used and b.split == target.split and b.kind == target.kind and b.e_type == target.e_type
        ]
        cands.sort(key=lambda b: (
            b.two_turn != target.two_turn,
            b.verdict != target.verdict,
            b.side != target.side,
            plan_idx[b.id],
        ))
        for b in cands:
            chosen = copy.copy(b)
            chosen.buffer = False
            drop_pair = bool(chosen.pair)
            if drop_pair:
                chosen.pair = None
            next_map = {**current_map, target.id: (chosen, drop_pair)}
            next_used = current_used | {chosen.id}
            next_subs = [*current_subs, {
                "excluded_id": target.id,
                "buffer_id": chosen.id,
                "kind": target.kind,
                "split": target.split or "train",
                "reason": "excluded",
                "pair_dropped": drop_pair,
            }]
            res = _search(idx + 1, next_map, next_used, next_subs)
            if res is not None:
                return res
        return None

    missing_base = [s.id for s in base if s.id not in excluded and s.id not in replaced_slots and s.id not in row_by_id]
    if missing_base:
        raise PlanError(f"REFUSED: missing row(s) for non-excluded base slot(s): {', '.join(missing_base)}")

    found = _search(0, initial_map, set(used_bufs), list(substitutions))
    if found is None:
        raise PlanError(
            f"REFUSED: no surviving buffer assignment for excluded_ids={sorted(excluded)} satisfies check_plan and check_batch"
        )
    final_map, final_subs = found
    return _build_rows(final_map), final_subs


# ---- the wave pilot (BUILD-SPEC §9.2, PLAN.md §7; composition ruled by R&D, 2026-10-17) ----------------------------
PILOT_LESSON = "T02"
PILOT_TWIN_LESSON = "T04"  # its apply rule is silent_without_deciding, the twin Argon's review exercised
PILOT_COUNTS = {"A": 4, "B": 5, "C": 4, "D": 4, "E": 3}
PILOT_TWINS = {"C": 2, "D": 2}


def plan_pilot(
    sides: dict[str, tuple[str, str]],
    *,
    seed: int,
    solve_first: tuple[str, ...],
    side_pairing: str = "default",
) -> list[Slot]:
    """The 20-slot wave pilot, from one full T02 lesson plan: A 4; B 5, one per solve-first structure; C 4 and D 4 (one
    two-turn D, no half of a contrast pair); E 3 (2 written, 1 verdict). A pilot's lesson floors are reported, not
    enforced: 20 slots can't hold a 40-slot lesson's shares. Its apply twins are a batch of their own
    (plan_pilot_twins), written after it with its accepted rows as --training (R&D, 2026-10-17)."""

    full = plan_lesson(PILOT_LESSON, sides[PILOT_LESSON], seed=seed, side_pairing=side_pairing)
    by_kind = {k: [s for s in full if s.kind == k] for k in KINDS}
    picked: list[Slot] = by_kind["A"][:PILOT_COUNTS["A"]]
    b = [s for s in by_kind["B"] if s.structure in solve_first[:PILOT_COUNTS["B"]]]
    if len(b) != PILOT_COUNTS["B"]:
        raise PlanError(f"the pilot needs one B slot per solve-first structure; the lesson plan has {len(b)}")
    picked += b
    for kind in ("C", "D"):
        pool = [s for s in by_kind[kind] if s.pair is None]
        if kind == "D":
            two = next((s for s in pool if s.two_turn), None)
            if two is None:
                raise PlanError("the pilot needs a two-turn D slot; the lesson plan has none outside a pair")
            pool = [two] + [s for s in pool if s is not two and not s.two_turn]
        # Spread the verdicts: one of each first, then the rest in plan order.
        mix: list[Slot] = []
        for v in VERDICTS:
            mix += [s for s in pool if s.verdict == v][:1]
        mix += [s for s in pool if s not in mix]
        if kind == "D":
            mix = [two] + [s for s in mix if s is not two]
        picked += mix[:PILOT_COUNTS[kind]]
    e = by_kind["E"]
    picked += [s for s in e if s.e_type == "written"][:2] + [s for s in e if s.e_type == "verdict"][:1]
    return picked


def plan_pilot_twins(
    sides: dict[str, tuple[str, str]],
    *,
    seed: int,
    side_pairing: str = "default",
) -> list[Slot]:
    """The pilot's apply twins: 2 C and 2 D decided T04 apply items, whose twin drops the deciding lines. Written after
    the pilot, with the pilot's own accepted T02 rows as --training, so the leakage check runs for real."""

    twins = [
        s
        for s in plan_apply(PILOT_TWIN_LESSON, sides[PILOT_TWIN_LESSON], seed=seed, side_pairing=side_pairing)
        if s.verdict != "cannot_tell"
    ]
    picked: list[Slot] = []
    for kind, n in PILOT_TWINS.items():
        got = [s for s in twins if s.kind == kind][:n]
        if len(got) != n:
            raise PlanError(f"the pilot needs {n} decided {PILOT_TWIN_LESSON} {kind} apply items")
        picked += got
    return picked

