"""Generate-then-reduce: sample a hidden solution, add clues true in it until the question has its answer, delete every
clue the answer doesn't need, then prove the trait's shortcut readings answer differently (design pins 1-5).

The model is the data contract: `build` returns an ordinary role-set model (entities, attributes, facts, asks or claim,
span), so the solver, puzzle-check, the sidecar and the readout take it unchanged. Fact lines are numbered from 2; line 1
is the setting line the renderer writes.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

from role_set.puzzle import Model, PuzzleError, holds, parse_model, solve
from role_set.span import parse_spans

# The forms the builder draws clues from. `contains` is never a clue (it is asked or claimed), and `same`/`different`
# need references the builder doesn't sample in slice 1.
CLUE_FORMS = ("is", "is_not", "in", "not_in", "before", "after", "adjacent", "not_adjacent", "offset", "count", "if")
MAX_TRIES = 400


class BuildError(ValueError):
    """No puzzle met the recipe within the tries allowed. The message says which proof kept failing."""


@dataclass(frozen=True)
class Shape:
    """What a puzzle is about, before any clue: its entities and attributes, what it asks, and its spans if dated."""

    entities: tuple[str, ...]
    attributes: dict  # name -> {"values": [...], "ordered": bool, "distinct": bool}
    asks: dict | None = None
    claim: list | None = None
    span: dict | None = None
    structure: str = "constraint_grid"

    def model(self, facts: list[list]) -> dict:
        m = {"entities": list(self.entities), "attributes": self.attributes,
             "facts": [{"line": i + 2, "rule": r} for i, r in enumerate(facts)], "flavour": [1]}
        if self.span is not None:
            m["span"] = self.span
        if self.claim is not None:
            m["claim"] = self.claim
        else:
            m["asks"] = self.asks
        return m


@dataclass
class Built:
    model: dict
    answer: str
    hidden: int  # index of the hidden solution among the shape's assignments
    trait_clue: list
    shortcuts: dict  # name -> {"answer": str, "differs": bool}
    metrics: dict  # assignments explored, load-bearing facts, lookup distance (design pin 1)
    seed: int
    tries: int
    notes: list[str] = field(default_factory=list)


def _norm(shape: Shape) -> Model:
    attrs = {a: {"values": list(s["values"]), "ordered": s.get("ordered") is True, "distinct": s.get("distinct", True) is not False}
             for a, s in shape.attributes.items()}
    span = parse_spans(shape.span, attrs) if shape.span is not None else None
    return Model(list(shape.entities), attrs, [], [], shape.asks, shape.claim, span)


def assignments(shape: Shape) -> list[dict]:
    m = _norm(shape)
    spaces = []
    for a in m.attrs.values():
        n = len(m.entities)
        spaces.append(list(itertools.permutations(a["values"])) if a["distinct"] else list(itertools.product(a["values"], repeat=n)))
    names = list(m.attrs)
    return [dict(zip(names, combo)) for combo in itertools.product(*spaces)]


def candidates(shape: Shape, forms=CLUE_FORMS) -> list[list]:
    """Every clue the builder may use, true or false: the rule forms over this shape's entities and values."""

    m = _norm(shape)
    out: list[list] = []
    ents = m.entities
    for a, spec in m.attrs.items():
        vals = spec["values"]
        if "is" in forms:
            out += [["is", e, a, v] for e in ents for v in vals]
        if "is_not" in forms:
            out += [["is_not", e, a, v] for e in ents for v in vals]
        if len(vals) >= 3:
            pairs = [list(p) for p in itertools.combinations(vals, 2)]
            if "in" in forms:
                out += [["in", e, a, p] for e in ents for p in pairs]
            if "not_in" in forms:
                out += [["not_in", e, a, p] for e in ents for p in pairs]
        if spec["ordered"]:
            for x, y in itertools.permutations(ents, 2):
                if "before" in forms:
                    out.append(["before", x, y, a])
                if "after" in forms:
                    out.append(["after", x, y, a])
                if "offset" in forms:
                    out += [["offset", x, y, a, k] for k in (1, 2) if k < len(vals)]
            for x, y in itertools.combinations(ents, 2):
                if "adjacent" in forms:
                    out.append(["adjacent", x, y, a])
                if "not_adjacent" in forms:
                    out.append(["not_adjacent", x, y, a])
        if not spec["distinct"] and "count" in forms:
            n = len(ents)
            out += [["count", a, v, op, k] for v in vals for op in ("=", ">=", "<=") for k in range(1, n)]
    if "if" in forms:
        simple = [r for r in out if r[0] in ("is", "is_not")]
        out += [["if", p, q] for p in simple for q in simple if p[0] == "is" and p[1] != q[1]]
    return out


def truth(shape: Shape, sols: list[dict], rule: list) -> int:
    m = _norm(shape)
    bits = 0
    for i, sol in enumerate(sols):
        if holds(m, sol, rule):
            bits |= 1 << i
    return bits


def solver_version() -> dict:
    """Pins the solver on every receipt, since "assignments explored" depends on its search (R&D, 2026-10-11)."""

    src = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for name in ("puzzle.py", "span.py", "generate/build.py", "generate/shortcuts.py"):
        digest.update((src / name).read_bytes())
    return {"sha256": digest.hexdigest()}


def metrics(model: dict) -> dict:
    """Design pin 1: assignments explored (the solver enumerates the whole space), load-bearing facts (every fact, once
    reduced), and lookup distance (the spread, in lines, of the facts the answer needs together)."""

    m = parse_model(model)
    size = 1
    for a in m.attrs.values():
        n = len(m.entities)
        size *= len(list(itertools.permutations(a["values"]))) if a["distinct"] else len(a["values"]) ** n
    lines = [n for n, _ in m.facts]
    return {"assignments_explored": size, "load_bearing_facts": len(lines), "lookup_distance": max(lines) - min(lines)}


def reduce(shape: Shape, sols: list[dict], chosen: list[list], bits: dict, decided, rng: random.Random, keep=None) -> list[list]:
    """Delete each clue the answer doesn't need, in a seeded order; `keep` is the trait clue, which the recipe needs on
    the deciding path and so is tried last (R&D: after reducing, check it is still there)."""

    order = [c for c in chosen if c is not keep]
    rng.shuffle(order)
    if keep is not None:
        order.append(keep)
    out = list(chosen)
    for c in order:
        trial = [x for x in out if x is not c]
        mask = _mask(trial, bits, len(sols))
        if decided(mask):
            out = trial
    return out


def _mask(rules: list[list], bits: dict, n: int) -> int:
    mask = (1 << n) - 1
    for r in rules:
        mask &= bits[_key(r)]
    return mask


def _key(rule) -> str:
    return repr(rule)


def receipt(built: Built, recipe: str, pack: dict, structure: str) -> dict:
    """What a generated puzzle carries (design step 10): the seed and tries, the recipe, the trait clue, every reading
    with its answer, the difficulty metrics, the skeleton, and the solver and pack hashes it was built with."""

    import json

    from role_set.puzzle_check import skeleton

    return {"recipe": recipe, "seed": built.seed, "tries": built.tries, "structure": structure, "answer": built.answer,
            "trait_clue": built.trait_clue, "shortcuts": built.shortcuts, "metrics": built.metrics,
            "skeleton_sha256": hashlib.sha256(skeleton(built.model).encode("utf-8")).hexdigest(),
            "pack_sha256": hashlib.sha256(json.dumps(pack, sort_keys=True).encode("utf-8")).hexdigest(),
            "solver": solver_version()}


def check_model(model: dict, structure: str) -> str:
    """The solver's own verdict on a built model: its answer, refusing a lookup or an idle fact like any reviewed item."""

    got = solve(parse_model(model), structure, "")
    if got.lookup:
        raise PuzzleError("a lookup: one fact settles the answer")
    return got.answer
