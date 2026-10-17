"""The cumulative exclusion state across batches (BUILD-SPEC B4).

`exclusion-state.json` holds every row accepted in earlier batches of the regeneration (its prompt, openers and
framing are in the row), each kind B row's model, the solve-first skeleton counts per structure, the domains and
frames used, and the ids of the batches it includes, with a sha256 over its contents. Every write and critique wave
loads it: the gates' `check_against` and the reply-opening check run against those rows plus this run's, and the
writer's avoid list is seeded with their framings. The plan reads its skeleton counts, so the skeleton floor holds
across batches. A batch's run writes the state it leaves behind, its own accepted rows added, beside its rows.

Standards: PIN_PER_STEP (the state's sha256 is a batch pin, the same in every wave of the batch), ANDON_AUTHORITY (a
state whose hash doesn't match its contents is refused). The undo is to keep using the earlier state file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "role-set exclusion state 1"
FILE = "exclusion-state.json"


class ExclusionError(ValueError):
    """A state file that can't be trusted: wrong format, a hash that doesn't match, or a batch added twice."""


def _digest(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class State:
    batches: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    models: dict = field(default_factory=dict)  # row id -> kind B model
    skeletons: dict = field(default_factory=dict)  # structure -> {skeleton: uses}
    domains: dict = field(default_factory=dict)  # NAICS code -> uses
    frames: dict = field(default_factory=dict)  # frame -> uses
    packs: dict = field(default_factory=dict)  # kind B pack id -> uses, so PACK_USES_MAX holds across batches (Argon §9.4 #10)

    def body(self) -> dict:
        return {"format": FORMAT, "batches": self.batches, "rows": self.rows, "models": self.models,
                "skeletons": self.skeletons, "domains": self.domains, "frames": self.frames, "packs": self.packs}

    @property
    def sha256(self) -> str:
        return _digest(self.body())

    def framings(self) -> list[str]:
        out = []
        for r in self.rows:
            first = (r.get("turns") or [{}])[0].get("user", "").split("\n", 1)[0]
            if first and r.get("kind") in ("C", "D", "E"):
                out.append(first)
        return out


def load(path: Path | None) -> State:
    """The state, its hash checked. None (no file named) is the empty state of a first batch."""

    if path is None:
        return State()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ExclusionError(f"REFUSED: cannot read the exclusion state ({type(e).__name__})") from e
    body = {k: data.get(k) for k in ("format", "batches", "rows", "models", "skeletons", "domains", "frames", "packs")}
    if body["format"] != FORMAT:
        raise ExclusionError(f"REFUSED: {path.name} is not an exclusion state ({FORMAT})")
    if data.get("sha256") != _digest(body):
        raise ExclusionError(f"REFUSED: {path.name}'s sha256 doesn't match its contents; it was edited or damaged")
    return State(batches=list(body["batches"]), rows=list(body["rows"]), models=dict(body["models"]),
                 skeletons={k: dict(v) for k, v in body["skeletons"].items()}, domains=dict(body["domains"]),
                 frames=dict(body["frames"]), packs=dict(body["packs"] or {}))


def save(state: State, path: Path) -> str:
    body = state.body()
    sha = _digest(body)
    path.write_text(json.dumps({**body, "sha256": sha}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return sha


def with_batch(state: State, batch_id: str, rows: list[dict], models: dict, slots: list) -> State:
    """The state after one batch: its accepted rows and models added, and its plan's skeletons, domains and frames
    counted. A batch already in the state is refused, so a rerun can't count its rows twice."""

    if batch_id in state.batches:
        raise ExclusionError(f"REFUSED: batch {batch_id} is already in the exclusion state")
    ids = {r.get("id") for r in rows}
    new = State(batches=[*state.batches, batch_id], rows=[*state.rows, *rows],
                models={**state.models, **{k: v for k, v in models.items() if k in ids}},
                skeletons={k: dict(v) for k, v in state.skeletons.items()}, domains=dict(state.domains), frames=dict(state.frames),
                packs=dict(state.packs))
    accepted = ids
    for s in slots:
        if s.id not in accepted:
            continue
        if s.kind == "B" and getattr(s, "b_skeleton", None):
            c = new.skeletons.setdefault(s.structure, {})
            c[s.b_skeleton] = c.get(s.b_skeleton, 0) + 1
        if s.domain:
            new.domains[s.domain["code"]] = new.domains.get(s.domain["code"], 0) + 1
        if s.frame:
            new.frames[s.frame] = new.frames.get(s.frame, 0) + 1
        if s.kind == "B" and getattr(s, "pack", None):
            new.packs[s.pack] = new.packs.get(s.pack, 0) + 1
    return new
