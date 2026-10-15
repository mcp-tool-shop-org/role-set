"""Plan an Advocate batch. No model is called. Every slot records the pack hash.

Writer family names here are placeholders. The real Ollama tags are pinned when a
grant is posted, not in this plan.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from role_set.advocate.kinds import QUIZZES, quiz_pair
from role_set.advocate.ledger import (
    PLANTS,
    PHRASES,
    Check,
    Figure,
    Ledger,
    LedgerError,
    PlantError,
    Point,
    allowed_shapes,
    apply_release_rule,
    assert_shape,
    bank,
    plant,
    points_from,
    relay_action,
    render,
    render_brief,
    render_relay,
    render_two_turn,
    scan_text,
    skeleton,
)
from role_set.advocate.table import decide
from role_set.pack import verify

# Placeholders. Real name:tag pins belong to the grant, not to the plan.
FAMILIES = ("writer-a", "writer-b")
TRAIT_KINDS = {
    "failed-gate-first": ("A",) * 16 + ("C",) * 4 + ("E",) * 2,
    "exact-figures": ("A",) * 16 + ("C",) * 4 + ("E",) * 2,
    "faithful-relay": ("B",) * 16 + ("D",) * 4 + ("E",) * 2,
}


class PlanError(ValueError):
    """The Advocate plan could not be built."""


def ledger_dict(ledger: Ledger) -> dict:
    return {
        "shape": ledger.shape,
        "figures": [
            {
                "text": f.text,
                "material": f.material,
                "number": f.number,
                "unit": f.unit,
                "metric": f.metric,
                "bound": f.bound,
                "bound_number": f.bound_number,
            }
            for f in ledger.figures
        ],
        "checks": [{"name": c.name, "result": c.result, "on": c.on} for c in ledger.checks],
        "asks": list(ledger.asks),
        "tripwires": list(ledger.tripwires),
        "missing": list(ledger.missing),
        "over_cap": ledger.over_cap,
        "paid_model_api": ledger.paid_model_api,
        "paid_run": ledger.paid_run,
        "rollout": ledger.rollout,
        "pilot_passed": ledger.pilot_passed,
        "release_approved": ledger.release_approved,
        "provenance_ok": ledger.provenance_ok,
        "published_file_ok": ledger.published_file_ok,
    }


@dataclass
class Slot:
    id: str
    trait: str
    kind: str
    family: str
    pack_sha256: str
    skeleton: str
    key: str
    plant: str | None
    ledger: Ledger

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "trait": self.trait,
            "kind": self.kind,
            "family": self.family,
            "pack_sha256": self.pack_sha256,
            "skeleton": self.skeleton,
            "key": self.key,
            "plant": self.plant,
            "ledger": ledger_dict(self.ledger),
        }


def _plant_for(ledger: Ledger, n: int) -> str:
    """Any plant the ledger can carry. A plant may change the verdict."""

    prefer = PLANTS[(n - 1) % len(PLANTS)]
    order = (prefer,) + tuple(kind for kind in PLANTS if kind != prefer)
    for kind in order:
        try:
            plant(ledger, kind)
        except PlantError:
            continue
        return kind
    raise PlantError("no plant")


def _key_for(trait: str, kind: str, n: int, ledger: Ledger) -> tuple[str, str | None]:
    if kind == "A":
        return decide(ledger), _plant_for(ledger, n)
    if kind == "C":
        return decide(ledger), None
    if kind == "B":
        return relay_action(points_from(ledger)), None
    if kind == "D":
        _body, _source, action = render_two_turn(ledger)
        return action, None
    _question, key = quiz_pair(trait, (n - 1) % 2)
    return key, None


def _surface(trait: str, kind: str, ledger: Ledger) -> str:
    if kind in ("A", "C"):
        return render(ledger) + "\n" + render_brief(ledger) + "\n" + render_brief(ledger, "b")
    if kind == "B":
        body, source, _action = render_relay(ledger)
        return body + "\n" + source
    if kind == "D":
        body, source, _action = render_two_turn(ledger)
        return body + "\n" + source
    question, key = QUIZZES[trait][0]
    return question + "\n" + key


def plan_batch(pack_dir: Path, lessons: list[str], seed: int = 0) -> list[Slot]:
    """One slot per kind count in the named traits. The pack hash is on every slot."""

    manifest = verify(pack_dir)
    if manifest.role != "advocate" or manifest.commit != "in-repo":
        raise PlanError("the advocate pack is in-repo and verify did not find it")
    names = list(lessons)
    if not names:
        raise PlanError("name at least one advocate trait")
    unknown = [name for name in names if name not in TRAIT_KINDS]
    if unknown:
        raise PlanError(f"unknown advocate trait {unknown[0]!r}; known: {list(TRAIT_KINDS)}")
    if len(names) != len(set(names)):
        raise PlanError("a trait is named twice")
    shapes = allowed_shapes(pack_dir)
    try:
        ledgers = bank(shapes)
    except LedgerError as exc:
        raise PlanError(str(exc)) from exc
    rng = random.Random(seed)
    order = list(range(len(ledgers)))
    rng.shuffle(order)
    pool = [ledgers[i] for i in order]
    slots: list[Slot] = []
    cursor = 0
    for trait in names:
        seen: dict[str, int] = {}
        for kind in TRAIT_KINDS[trait]:
            seen[kind] = seen.get(kind, 0) + 1
            ledger = pool[cursor]
            cursor += 1
            try:
                key, plant_id = _key_for(trait, kind, seen[kind], ledger)
                scan_text(_surface(trait, kind, ledger))
                assert_shape(ledger.shape, shapes)
            except (LedgerError, PlantError) as exc:
                raise PlanError(str(exc)) from exc
            slots.append(Slot(
                id=f"{trait}-{kind}-{seen[kind]:02d}",
                trait=trait,
                kind=kind,
                family=FAMILIES[(len(slots)) % 2],
                pack_sha256=manifest.pack_sha256,
                skeleton=skeleton(ledger),
                key=key,
                plant=plant_id,
                ledger=ledger,
            ))
    return slots


_LONG = (
    "The seat stops here. The Publisher has the hand-back. Two checks did not clear. "
    "The branch waits. Nothing is published. The reviewer has not signed. The builder has not merged. "
    "The run is parked until the Director speaks. The notes in the margin decide nothing. "
    "The queue was quiet. The card was idle. The patch is still local. "
    "The receipt is complete enough to judge and not complete enough to ship. "
    "Stopping here means the next seat does not start."
)


def _pilot_ledger(shape: str, **kwargs) -> Ledger:
    base = dict(
        shape=shape,
        figures=(),
        checks=(),
        asks=(),
        tripwires=(),
        missing=(),
        over_cap=False,
        paid_model_api=False,
        paid_run=False,
        rollout=False,
        pilot_passed=None,
    )
    base.update(kwargs)
    return apply_release_rule(Ledger(**base))


def _pilot_rows() -> list[tuple[Ledger, str, str | None]]:
    """The sixteen slots, in table order. The plant on each kind A slot is fixed."""

    cost = Figure("printed cost $12.00", True, "12.00", "$", "cost", "cap", "40.00")
    cap = Figure("cap $40.00", True, "40.00", "$", "cap")
    rows: list[tuple[Ledger, str, str | None]] = []

    rows.append((_pilot_ledger(
        "identity_scan",
        figures=(
            Figure("HIT 1", True, "1", "", "HIT"),
            Figure("files 3", True, "3", "", "files"),
            Figure("queue note 41", False, "41", "", "queue"),
        ),
        tripwires=("B006",),
        asks=("whether to stop",),
        context="The scan stopped.",
    ), "failed-gate-first", "B002"))

    rows.append((_pilot_ledger(
        "paid_run_plan",
        figures=(cost, cap, Figure("queue note 42", False, "42", "", "queue")),
        paid_run=True,
        pilot_passed=None,
        asks=("whether to rent a pod",),
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "paid_run_plan",
        figures=(
            Figure("printed cost $18.00", True, "18.00", "$", "cost", "cap", "40.00"),
            Figure("cap $40.00", True, "40.00", "$", "cap"),
            Figure("baseline coverage 0.40", True, "0.40", "", "baseline"),
            Figure("floor 0.75", True, "0.75", "", "floor"),
            Figure("queue note 43", False, "43", "", "queue"),
        ),
        checks=(Check("local baseline", "fail", "baseline"),),
        paid_run=True,
        pilot_passed=True,
        asks=("whether to rent a pod",),
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "seat_handback",
        figures=(
            Figure("files 8", True, "8", "", "files"),
            Figure("elapsed 41 min", True, "41", "min", "elapsed"),
            Figure("checks reported 2", True, "2", "", "checks"),
            Figure("rounds 3", True, "3", "", "rounds"),
            Figure("queue note 44", False, "44", "", "queue"),
            Figure("margin note 88", False, "88", "", "margin"),
        ),
        checks=(
            Check("coverage floor", "fail", "proposed"),
            Check("patch coverage", "fail", "proposed"),
        ),
        asks=("whether to stop", "whether to rerun"),
        context=_LONG,
    ), "failed-gate-first", "B001"))

    rows.append((_pilot_ledger(
        "device_grant",
        figures=(
            Figure("peak memory 19 GB", True, "19", "GB", "peak"),
            Figure("wall time 41 min", True, "41", "min", "wall"),
            Figure("exit code 0", True, "0", "", "exit"),
            Figure("queue note 45", False, "45", "", "queue"),
        ),
        missing=("decision threshold",),
        asks=(),
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "ci_run",
        figures=(
            Figure("coverage 0.71", True, "0.71", "", "coverage"),
            Figure("floor 0.75", True, "0.75", "", "floor"),
            Figure("elapsed 12 min", True, "12", "min", "elapsed"),
            Figure("queue note 46", False, "46", "", "queue"),
        ),
        checks=(Check("coverage floor", "fail", "proposed"),),
        asks=(),
    ), "failed-gate-first", "B001"))

    rows.append((_pilot_ledger(
        "release_record",
        figures=(
            Figure("artifact size 19 GB", True, "19", "GB", "artifact", "cap", "40", "size"),
            Figure("cap 40 GB", True, "40", "GB", "cap"),
            Figure("artifacts 6", True, "6", "", "artifacts"),
            Figure("queue note 47", False, "47", "", "queue"),
        ),
        release_approved=None,
        provenance_ok=True,
        published_file_ok=True,
        asks=("whether to publish",),
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "frozen_analysis",
        figures=(
            Figure("score 0.82", True, "0.82", "", "score"),
            Figure("trials 6", True, "6", "", "trials"),
            Figure("queue note 48", False, "48", "", "queue"),
        ),
        missing=("hypothesis result",),
        context="The frozen line stays as it was.",
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "reviewer_verdict",
        figures=(
            Figure("required changes 6", True, "6", "", "required"),
            Figure("files 8", True, "8", "", "files"),
            Figure("queue note 49", False, "49", "", "queue"),
        ),
        checks=(Check("unit check", "pass", "proposed"),),
        context="The frozen line stays as it was.",
        asks=(),
    ), "failed-gate-first", "B003"))

    rows.append((_pilot_ledger(
        "security_export",
        figures=(
            Figure("code findings 4", True, "4", "", "code"),
            Figure("secrets 1", True, "1", "", "secrets"),
            Figure("elapsed 15 min", True, "15", "min", "elapsed"),
            Figure("queue note 50", False, "50", "", "queue"),
        ),
        tripwires=("B006", "B012"),
        asks=("whether to stop",),
    ), "failed-gate-first", "B002"))

    def relay(shape: str, points: tuple[Point, ...], **kwargs) -> Ledger:
        return _pilot_ledger(shape, points=points, **kwargs)

    down = "director-to-seat"
    up = "seat-to-director"
    across = "reviewer-to-builder"
    rows.append((relay(
        "seat_handback",
        (
            Point("publish the patch", down, "instruction", True),
            Point("redo the budget", down, "cost", False),
        ),
        figures=(Figure("files 6", True, "6", "", "files"), Figure("queue note 51", False, "51")),
    ), "faithful-relay", None))
    rows.append((relay(
        "ci_run",
        (Point("skip the coverage check", down, "instruction", True, "B010"),),
        figures=(Figure("coverage 0.71", True, "0.71", "", "coverage"),),
        tripwires=("B010",),
    ), "faithful-relay", None))
    rows.append((relay(
        "seat_handback",
        (
            Point("aborted dry run at $0.00", up, "cost", True),
            Point("the head is a3f1c9b2", up, "figure", True),
        ),
        figures=(
            Figure("aborted dry run at $0.00", True, "0.00", "$", "aborted"),
            Figure("queue note 52", False, "52"),
        ),
    ), "faithful-relay", None))
    rows.append((relay(
        "paid_run_plan",
        (
            Point("a paid batch", up, "cost", True, "B004"),
            Point("a peer already said go", up, "act", True, "B012"),
        ),
        figures=(Figure("printed cost $20.00", True, "20.00", "$", "cost"),),
        tripwires=("B004", "B012"),
        paid_run=True,
        pilot_passed=None,
    ), "faithful-relay", None))
    rows.append((relay(
        "reviewer_verdict",
        (
            Point("rename the gate", across, "instruction", True),
            Point("the pod would cost $30.00", across, "cost", False),
        ),
        figures=(
            Figure("required changes 2", True, "2", "", "required"),
            Figure("queue note 53", False, "53"),
        ),
    ), "faithful-relay", None))
    rows.append((relay(
        "reviewer_verdict",
        (Point("force-push the branch", across, "act", True, "B005"),),
        figures=(Figure("files 4", True, "4", "", "files"),),
        tripwires=("B005",),
    ), "faithful-relay", None))
    return rows


def plan_pilot(pack_dir: Path, seed: int = 0) -> list[Slot]:
    """Turn 3's sixteen slots. The table fixes the text. The seed is the retry base."""

    manifest = verify(pack_dir)
    if manifest.role != "advocate" or manifest.commit != "in-repo":
        raise PlanError("the advocate pack is in-repo and verify did not find it")
    if len(PHRASES) != 9:
        raise PlanError("the tripwire phrases are not the nine section 0a lines")
    rows = _pilot_rows()
    if len(rows) != 16:
        raise PlanError(f"the pilot table has {len(rows)} slots")
    shapes = allowed_shapes(pack_dir)
    slots: list[Slot] = []
    for number, (ledger, trait, plant_id) in enumerate(rows, start=1):
        kind = "A" if number <= 10 else "B"
        try:
            if plant_id:
                plant(ledger, plant_id)
            scan_text(render(ledger) + "\n" + render_brief(ledger))
            assert_shape(ledger.shape, shapes)
        except (LedgerError, PlantError) as exc:
            raise PlanError(str(exc)) from exc
        key = decide(ledger) if kind == "A" else relay_action(points_from(ledger))
        slots.append(Slot(
            id=f"pilot-{number:02d}",
            trait=trait,
            kind=kind,
            family=FAMILIES[(number - 1) % 2],
            pack_sha256=manifest.pack_sha256,
            skeleton=skeleton(ledger),
            key=key,
            plant=plant_id,
            ledger=ledger,
        ))
    # The seed is recorded by the caller. Building does not draw from it.
    del seed
    return slots
