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
    Ledger,
    LedgerError,
    allowed_shapes,
    assert_shape,
    bank,
    points_from,
    relay_action,
    render_brief,
    render_record,
    render_relay,
    render_two_turn,
    scan_text,
    skeleton,
    stable_plants,
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
        "figures": [{"text": f.text, "material": f.material, "number": f.number} for f in ledger.figures],
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
    options = stable_plants(ledger)
    prefer = PLANTS[(n - 1) % len(PLANTS)]
    return prefer if prefer in options else options[0]


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
        return render_record(ledger) + "\n" + render_brief(ledger) + "\n" + render_brief(ledger, "b")
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
            key, plant_id = _key_for(trait, kind, seen[kind], ledger)
            try:
                scan_text(_surface(trait, kind, ledger))
                assert_shape(ledger.shape, shapes)
            except LedgerError as exc:
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
