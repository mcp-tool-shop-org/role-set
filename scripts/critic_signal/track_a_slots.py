"""Track A Rung 0 slot census and frozen 48-slot list builder (PLAN-ROUND4 §3 and §6).

Selects 48 kind C slots across non-T10 verifier lessons:
- 24 odd decided slots (`slot_flaw != "none"`, verdict in `supported`/`unsupported`) -> `f_eligible`
- 24 even decided slots (`slot_flaw == "none"`, verdict in `supported`/`unsupported`) -> `controls`
- Zero `T10` slots.

Priority order within each 24-slot half (sorted in lesson order, then slot ID order):
1. Lessons Phase 4 did not use (`T04`, `T12`).
2. Unused slots in Phase 4's non-T10 lessons (`T02, T03, T06, T07, T08, T09, T11, T15, T18`).
3. Re-write candidates (previously used non-T10 slots) with fresh domain draws Recorded alongside seed-0 domains.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = ROOT / "roles" / "domains" / "naics-2022.json"

ALL_LESSONS: tuple[str, ...] = (
    "T02",
    "T03",
    "T04",
    "T06",
    "T07",
    "T08",
    "T09",
    "T10",
    "T11",
    "T12",
    "T15",
    "T18",
)
NON_T10_LESSONS: tuple[str, ...] = tuple(l for l in ALL_LESSONS if l != "T10")
PHASE4_LESSONS: tuple[str, ...] = (
    "T02",
    "T03",
    "T06",
    "T07",
    "T08",
    "T09",
    "T10",
    "T11",
    "T15",
    "T18",
)
PHASE4_NON_T10_LESSONS: tuple[str, ...] = tuple(l for l in PHASE4_LESSONS if l != "T10")
UNUSED_LESSONS: tuple[str, ...] = tuple(l for l in NON_T10_LESSONS if l not in PHASE4_NON_T10_LESSONS)

PILOT_USED_C_SLOTS: tuple[str, ...] = ("T02-C-01", "T02-C-08")
PHASE4_F_ELIGIBLE: tuple[str, ...] = (
    "T02-C-03",
    "T02-C-05",
    "T02-C-07",
    "T03-C-01",
    "T03-C-05",
    "T03-C-07",
    "T06-C-05",
    "T06-C-07",
    "T07-C-01",
    "T07-C-03",
    "T08-C-01",
    "T08-C-07",
    "T09-C-01",
    "T09-C-03",
    "T09-C-05",
    "T09-C-07",
    "T10-C-01",
    "T10-C-03",
    "T10-C-07",
    "T11-C-03",
    "T11-C-05",
    "T11-C-07",
    "T15-C-03",
    "T15-C-05",
    "T18-C-01",
    "T18-C-03",
    "T18-C-05",
)
PHASE4_CONTROLS: tuple[str, ...] = (
    "T02-C-06",
    "T03-C-02",
    "T03-C-04",
    "T03-C-08",
    "T06-C-02",
    "T06-C-06",
    "T06-C-08",
    "T07-C-02",
    "T07-C-04",
    "T07-C-08",
    "T08-C-04",
)
USED_BEFORE_IDS: frozenset[str] = frozenset(PHASE4_F_ELIGIBLE + PHASE4_CONTROLS)
DECIDED_VERDICTS: frozenset[str] = frozenset(("supported", "unsupported"))


@dataclass(frozen=True)
class SlotCandidate:
    slot_id: str
    lesson: str
    number: int
    parity: str
    verdict: str
    flaw: str
    content_kind: str
    side: str
    domain_seed0: dict | None
    domain_fresh: dict | None
    frame_seed0: str | None
    frame_fresh: str | None
    source_bucket: str

    def as_dict(self) -> dict:
        return {
            "id": self.slot_id,
            "lesson": self.lesson,
            "number": self.number,
            "parity": self.parity,
            "verdict": self.verdict,
            "flaw": self.flaw,
            "content_kind": self.content_kind,
            "side": self.side,
            "domain_seed0": self.domain_seed0,
            "domain_fresh": self.domain_fresh,
            "frame_seed0": self.frame_seed0,
            "frame_fresh": self.frame_fresh,
            "domain_changed": self.domain_seed0 != self.domain_fresh,
            "source_bucket": self.source_bucket,
            "needs_novelty_check": self.source_bucket == "rewrite_candidate",
        }


def _slot_order_key(slot_id: str) -> tuple[int, int]:
    lesson, _, num = slot_id.split("-")
    lesson_idx = NON_T10_LESSONS.index(lesson) if lesson in NON_T10_LESSONS else 999
    return (lesson_idx, int(num))


def plan_c_slots_for_lesson(
    lesson: str,
    *,
    plan_seed: int = 1,
    fresh_domain_seed: int = 1,
    domains_path: Path = DEFAULT_DOMAINS,
) -> list[SlotCandidate]:
    """Plan one lesson at `plan_seed`, compute `domain_seed0`/`frame_seed0` at seed 0 always, compute fresh domain/frame at `fresh_domain_seed` if `plan_seed == 0` else `plan_seed`, and check `slot_flaw`."""
    from role_set import domains as dom_mod
    from role_set.assemble import FLAWS, slot_flaw
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import plan_batch

    if lesson == "T10":
        raise ValueError("REFUSED: T10 is excluded from Track A (pending R&D T10 redesign)")

    table = dom_mod.load(domains_path)

    slots_seed0 = plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=0)
    dom_mod.assign(slots_seed0, table, seed=0)
    seed0_by_id = {s.id: s for s in slots_seed0 if s.kind == "C"}

    fresh_seed = fresh_domain_seed if plan_seed == 0 else plan_seed
    slots_plan = plan_batch({lesson: VERIFIER_SIDES[lesson]}, seed=plan_seed)
    dom_mod.assign(slots_plan, table, seed=fresh_seed)

    out: list[SlotCandidate] = []
    for s in slots_plan:
        if s.kind != "C" or s.id in PILOT_USED_C_SLOTS:
            continue
        flaw = slot_flaw(s)
        num = int(s.id.rsplit("-", 1)[1])
        parity = "even" if num % 2 == 0 else "odd"
        if parity == "even" and flaw != "none":
            raise RuntimeError(f"REFUSED: even slot {s.id} has slot_flaw={flaw!r}, expected 'none'")
        if parity == "odd" and flaw not in FLAWS:
            raise RuntimeError(f"REFUSED: odd slot {s.id} has slot_flaw={flaw!r}, expected one of {FLAWS}")

        if lesson in UNUSED_LESSONS:
            bucket = "unused_lesson"
        elif s.id not in USED_BEFORE_IDS:
            bucket = "phase4_unused_slot"
        else:
            bucket = "rewrite_candidate"

        s0 = seed0_by_id[s.id]
        out.append(
            SlotCandidate(
                slot_id=s.id,
                lesson=lesson,
                number=num,
                parity=parity,
                verdict=s.verdict,
                flaw=flaw,
                content_kind=s.content_kind or "",
                side=s.side,
                domain_seed0=s0.domain,
                domain_fresh=s.domain,
                frame_seed0=s0.frame,
                frame_fresh=s.frame,
                source_bucket=bucket,
            )
        )
    return out


def census_for_seed(
    *,
    plan_seed: int = 1,
    fresh_domain_seed: int = 1,
    domains_path: Path = DEFAULT_DOMAINS,
    target_per_half: int = 24,
    no_rewrites: bool = False,
) -> dict:
    """Compute the full non-T10 C-slot census and select `target_per_half` odd + `target_per_half` even decided slots."""
    from role_set.assemble import FLAWS, slot_flaw
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import plan_batch

    all_c: list[SlotCandidate] = []
    for lesson in NON_T10_LESSONS:
        all_c.extend(
            plan_c_slots_for_lesson(
                lesson,
                plan_seed=plan_seed,
                fresh_domain_seed=fresh_domain_seed,
                domains_path=domains_path,
            )
        )

    decided_odd = [c for c in all_c if c.parity == "odd" and c.verdict in DECIDED_VERDICTS]
    decided_even = [c for c in all_c if c.parity == "even" and c.verdict in DECIDED_VERDICTS]

    def _by_bucket(items: list[SlotCandidate], bucket: str) -> list[SlotCandidate]:
        return sorted((c for c in items if c.source_bucket == bucket), key=lambda x: _slot_order_key(x.slot_id))

    odd_unused_lesson = _by_bucket(decided_odd, "unused_lesson")
    odd_phase4_unused = _by_bucket(decided_odd, "phase4_unused_slot")
    odd_rewrite = _by_bucket(decided_odd, "rewrite_candidate")

    even_unused_lesson = _by_bucket(decided_even, "unused_lesson")
    even_phase4_unused = _by_bucket(decided_even, "phase4_unused_slot")
    even_rewrite = _by_bucket(decided_even, "rewrite_candidate")

    selected_odd = (odd_unused_lesson + odd_phase4_unused + odd_rewrite)[:target_per_half]
    selected_even = (even_unused_lesson + even_phase4_unused + even_rewrite)[:target_per_half]

    if len(selected_odd) != target_per_half:
        raise RuntimeError(f"REFUSED: found {len(selected_odd)} odd decided slots, need {target_per_half}")
    if len(selected_even) != target_per_half:
        raise RuntimeError(f"REFUSED: found {len(selected_even)} even decided slots, need {target_per_half}")

    odd_rewrites_taken = sum(1 for c in selected_odd if c.source_bucket == "rewrite_candidate")
    even_rewrites_taken = sum(1 for c in selected_even if c.source_bucket == "rewrite_candidate")
    needs_novelty_check_count = odd_rewrites_taken + even_rewrites_taken
    if no_rewrites and needs_novelty_check_count > 0:
        raise RuntimeError(
            f"REFUSED: --no-rewrites forbids rewrite candidates, but seed {plan_seed} requires "
            f"{needs_novelty_check_count} rewrite slot(s) ({odd_rewrites_taken} odd, {even_rewrites_taken} even)"
        )

    # Re-verify every selected slot directly against role-set's plan_batch and slot_flaw
    by_lesson_plan = {
        l: {s.id: s for s in plan_batch({l: VERIFIER_SIDES[l]}, seed=plan_seed)}
        for l in NON_T10_LESSONS
    }
    for c in selected_odd:
        if c.lesson == "T10" or c.slot_id.startswith("T10-"):
            raise RuntimeError(f"REFUSED: T10 slot {c.slot_id} in selected_odd")
        if c.slot_id in PILOT_USED_C_SLOTS:
            raise RuntimeError(f"REFUSED: pilot-used slot {c.slot_id} in selected_odd")
        raw_slot = by_lesson_plan[c.lesson][c.slot_id]
        flaw = slot_flaw(raw_slot)
        if flaw == "none" or flaw not in FLAWS:
            raise RuntimeError(f"REFUSED: selected odd slot {c.slot_id} has slot_flaw={flaw!r}")
        if raw_slot.verdict not in DECIDED_VERDICTS:
            raise RuntimeError(f"REFUSED: selected odd slot {c.slot_id} has undecided verdict {raw_slot.verdict!r}")

    for c in selected_even:
        if c.lesson == "T10" or c.slot_id.startswith("T10-"):
            raise RuntimeError(f"REFUSED: T10 slot {c.slot_id} in selected_even")
        if c.slot_id in PILOT_USED_C_SLOTS:
            raise RuntimeError(f"REFUSED: pilot-used slot {c.slot_id} in selected_even")
        raw_slot = by_lesson_plan[c.lesson][c.slot_id]
        flaw = slot_flaw(raw_slot)
        if flaw != "none":
            raise RuntimeError(f"REFUSED: selected even slot {c.slot_id} has slot_flaw={flaw!r}, expected 'none'")
        if raw_slot.verdict not in DECIDED_VERDICTS:
            raise RuntimeError(f"REFUSED: selected even slot {c.slot_id} has undecided verdict {raw_slot.verdict!r}")

    selected_odd_sorted = sorted(selected_odd, key=lambda x: _slot_order_key(x.slot_id))
    selected_even_sorted = sorted(selected_even, key=lambda x: _slot_order_key(x.slot_id))
    all_selected_sorted = sorted(selected_odd_sorted + selected_even_sorted, key=lambda x: _slot_order_key(x.slot_id))
    lessons_used = [l for l in NON_T10_LESSONS if any(c.lesson == l for c in all_selected_sorted)]

    return {
        "plan_seed": plan_seed,
        "fresh_domain_seed": fresh_domain_seed if plan_seed == 0 else plan_seed,
        "lessons": lessons_used,
        "excluded_lessons": ["T10"],
        "excluded_pilot_slots": list(PILOT_USED_C_SLOTS),
        "slots_per_lesson": 8,
        "f_eligible": [c.slot_id for c in selected_odd_sorted],
        "controls": [c.slot_id for c in selected_even_sorted],
        "slot_ids": [c.slot_id for c in all_selected_sorted],
        "draft_slot_count": len(all_selected_sorted),
        "needs_novelty_check_count": needs_novelty_check_count,
        "counts": {
            "total_non_t10_c_slots": len(all_c),
            "odd_decided_available": len(decided_odd),
            "even_decided_available": len(decided_even),
            "odd_by_bucket": {
                "unused_lessons": len(odd_unused_lesson),
                "phase4_unused_slots": len(odd_phase4_unused),
                "rewrite_candidates_available": len(odd_rewrite),
                "rewrite_candidates_taken": odd_rewrites_taken,
            },
            "even_by_bucket": {
                "unused_lessons": len(even_unused_lesson),
                "phase4_unused_slots": len(even_phase4_unused),
                "rewrite_candidates_available": len(even_rewrite),
                "rewrite_candidates_taken": even_rewrites_taken,
            },
        },
        "buckets": {
            "odd": {
                "unused_lessons": [c.slot_id for c in odd_unused_lesson],
                "phase4_unused_slots": [c.slot_id for c in odd_phase4_unused],
                "rewrite_candidates_taken": [
                    c.slot_id for c in selected_odd_sorted if c.source_bucket == "rewrite_candidate"
                ],
                "rewrite_candidates_available": [c.slot_id for c in odd_rewrite],
            },
            "even": {
                "unused_lessons": [c.slot_id for c in even_unused_lesson],
                "phase4_unused_slots": [c.slot_id for c in even_phase4_unused],
                "rewrite_candidates_taken": [
                    c.slot_id for c in selected_even_sorted if c.source_bucket == "rewrite_candidate"
                ],
                "rewrite_candidates_available": [c.slot_id for c in even_rewrite],
            },
        },
        "slot_details": {c.slot_id: c.as_dict() for c in all_selected_sorted},
    }


def build_track_a_slots(
    *,
    plan_seed: int = 1,
    fresh_domain_seed: int = 1,
    domains_path: Path = DEFAULT_DOMAINS,
    out_path: Path | None = None,
    no_rewrites: bool = False,
) -> dict:
    """Build the Track A Rung 0 frozen slot JSON (default seed 1, plus alternate seed census for comparison)."""
    primary = census_for_seed(
        plan_seed=plan_seed,
        fresh_domain_seed=fresh_domain_seed,
        domains_path=domains_path,
        no_rewrites=no_rewrites,
    )
    alt_seed = 0 if plan_seed != 0 else 1
    alt = census_for_seed(
        plan_seed=alt_seed,
        fresh_domain_seed=fresh_domain_seed,
        domains_path=domains_path,
        no_rewrites=False,
    )

    payload = {
        "source": (
            "PLAN-ROUND4.md §3 and §6 (Track A Rung 0); computed with role-set's plan_batch, "
            "domains.assign, and assemble.slot_flaw (T10 and pilot C slots excluded)"
        ),
        **primary,
        f"alternate_seed_{alt_seed}_census": {
            "plan_seed": alt["plan_seed"],
            "lessons": alt["lessons"],
            "f_eligible": alt["f_eligible"],
            "controls": alt["controls"],
            "needs_novelty_check_count": alt["needs_novelty_check_count"],
            "counts": alt["counts"],
            "buckets": alt["buckets"],
        },
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1, help="plan_batch seed (default: 1)")
    p.add_argument("--fresh-domain-seed", type=int, default=1, help="domain draw seed when --seed is 0 (default: 1)")
    p.add_argument("--domains", type=Path, default=DEFAULT_DOMAINS, help="NAICS domain table path")
    p.add_argument("--no-rewrites", action="store_true", help="refuse if any rewrite candidate would be taken")
    p.add_argument("--out", type=Path, default=None, help="write frozen JSON to this file")
    args = p.parse_args(argv)

    try:
        payload = build_track_a_slots(
            plan_seed=args.seed,
            fresh_domain_seed=args.fresh_domain_seed,
            domains_path=args.domains,
            out_path=args.out,
            no_rewrites=args.no_rewrites,
        )
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
