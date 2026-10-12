"""Trait recipes: the shape a lesson's puzzles take, the clue forms they draw on, the trait clue that must stay on the
deciding path, and the shortcut readings that must answer differently.

Slice 1, first recipe: T10 (takes statements literally). The trait clue is a count ("exactly / at least / at most k of
them are v"), and the readings are R&D's four size loosenings. A puzzle is accepted only when the count survives the
reduction and at least one applicable reading gives a different answer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from role_set.generate.build import (MAX_TRIES, BuildError, Built, Shape, _key, _mask, assignments, candidates, check_model,
                                     metrics, reduce, truth)
from role_set.generate.shortcuts import REGISTRY, prove
from role_set.puzzle import PuzzleError

# Slice-1 domain packs: names, a plural noun and a two- or three-value state. Seat C's packs replace these (with sense
# rules) once the clue-form API is settled; each pack here keeps every name a whole word no other name contains.
PACKS = {
    "lamps": {"plural": "lamps", "names": ["the brass lamp", "the copper lamp", "the tin lamp", "the iron lamp", "the lead lamp"],
              "attribute": "light", "values": ["lit", "dark"]},
    "cups": {"plural": "cups", "names": ["the blue cup", "the green cup", "the grey cup", "the white cup", "the yellow cup"],
             "attribute": "heat", "values": ["hot", "warm", "cold"]},
    "doors": {"plural": "doors", "names": ["the oak door", "the pine door", "the ash door", "the elm door", "the yew door"],
              "attribute": "state", "values": ["open", "shut"]},
}


@dataclass(frozen=True)
class Recipe:
    lesson: str
    trait_form: str
    shortcuts: tuple[str, ...]
    forms: tuple[str, ...]


MAX_IF = 2  # conditionals per puzzle; a wall of if-lines reads as a logic drill, not a page

T10 = Recipe("T10", "count", tuple(n for n, s in REGISTRY.items() if s["lesson"] == "T10"),
             ("is", "is_not", "count", "if"))


def t10_shape(pack: dict, n: int, rng: random.Random, question: str) -> Shape:
    names = tuple(rng.sample(pack["names"], n))
    attrs = {pack["attribute"]: {"values": list(pack["values"]), "distinct": False}}
    if question == "claim":
        return Shape(names, attrs, claim=["is", names[-1], pack["attribute"], pack["values"][0]], structure="falsifier_first")
    if question == "set":
        # "Which of them could be lit?": the answer is every possible group, so a loosened size word changes the set. A
        # single-value question can't carry a T10 reading: loosening only widens the possibilities, so the key stays
        # among them, which R&D's divergence rule doesn't count.
        return Shape(names, attrs, asks={"entity_with": {pack["attribute"]: pack["values"][0]}},
                     structure="underdetermined_with_explicit_set")
    return Shape(names, attrs, asks={"attribute": pack["attribute"], "of": names[-1]})


SET_MAX = 4  # possible groups in an open-set answer; more reads as a listing exercise


def _groups(shape: Shape, sols: list[dict]):
    from role_set.generate.build import _norm
    from role_set.puzzle import _project

    m = _norm(shape)
    projs = [_project(m, s) for s in sols]
    return lambda mask: frozenset(projs[i] for i in range(len(sols)) if mask >> i & 1)


def build_t10(pack_name: str, seed: int, n: int = 4, question: str = "value", tries: int = MAX_TRIES) -> Built:
    """One T10 puzzle from a seed: a count clue on the deciding path, every fact needed, no lookup, and at least one size
    loosening that answers differently."""

    pack = PACKS[pack_name]
    rng = random.Random(seed)
    last = "no attempt"
    for attempt in range(1, tries + 1):
        shape = t10_shape(pack, n, rng, question)
        sols = assignments(shape)
        pool = candidates(shape, T10.forms)
        bits = {_key(c): truth(shape, sols, c) for c in pool}
        hidden = rng.randrange(len(sols))
        true_pool = [c for c in pool if bits[_key(c)] >> hidden & 1]
        traits = [c for c in true_pool if c[0] == T10.trait_form]
        if not traits:
            last = "no true count clue"
            continue
        trait = rng.choice(traits)
        others = [c for c in true_pool if c is not trait]
        rng.shuffle(others)
        if shape.structure == "underdetermined_with_explicit_set":
            kept = _open_set(shape, sols, bits, trait, others, rng)
            if kept is None:
                last = "no clue set left a small open set that needs the count"
                continue
            built = _finish(shape, kept, trait, hidden, seed, attempt)
            if isinstance(built, str):
                last = built
                continue
            return built
        decided = _decider(shape, sols, hidden)
        chosen = [trait]
        rest: list[list] = []
        for c in others:
            if decided(_mask(chosen, bits, len(sols))):
                break
            if c[0] == "if" and sum(1 for x in chosen if x[0] == "if") >= MAX_IF:
                continue
            # The trait clue stays on the deciding path by construction: a clue joins only while the clues without the
            # count still leave the answer open.
            if decided(_mask(rest + [c], bits, len(sols))):
                continue
            if _mask(chosen + [c], bits, len(sols)) != _mask(chosen, bits, len(sols)):
                chosen.append(c)
                rest.append(c)
        if not decided(_mask(chosen, bits, len(sols))):
            last = "the clues never settled the answer"
            continue
        kept = reduce(shape, sols, chosen, bits, decided, rng, keep=trait)
        if trait not in kept:
            last = "the count dropped out in reduction"
            continue
        built = _finish(shape, kept, trait, hidden, seed, attempt)
        if isinstance(built, str):
            last = built
            continue
        return built
    raise BuildError(f"T10 ({pack_name}, seed {seed}): no puzzle in {tries} tries; last refusal: {last}")


def _finish(shape: Shape, kept: list[list], trait: list, hidden: int, seed: int, attempt: int) -> Built | str:
    """The solver's own checks on the reduced model, then the shortcut proof; a refusal comes back as its reason."""

    model = shape.model(kept)
    try:
        answer = check_model(model, shape.structure)
    except PuzzleError as e:
        return str(e)
    proofs = prove(model, shape.structure, list(T10.shortcuts))
    if not any(p["differs"] for p in proofs.values()):
        return "no size loosening changed the answer"
    return Built(model=model, answer=answer, hidden=hidden, trait_clue=trait, shortcuts=proofs, metrics=metrics(model),
                 seed=seed, tries=attempt)


def _open_set(shape: Shape, sols: list[dict], bits: dict, trait: list, others: list[list], rng: random.Random):
    """Clues for an open-set question: add until 2 to SET_MAX groups stay possible, drop every clue the set doesn't need,
    and keep the count only if the set changes without it."""

    groups = _groups(shape, sols)
    n = len(sols)
    chosen = [trait]
    for c in others:
        g = groups(_mask(chosen, bits, n))
        if 2 <= len(g) <= SET_MAX:
            break
        if c[0] == "if" and sum(1 for x in chosen if x[0] == "if") >= MAX_IF:
            continue
        if len(groups(_mask(chosen + [c], bits, n))) >= 2 and _mask(chosen + [c], bits, n) != _mask(chosen, bits, n):
            chosen.append(c)
    target = groups(_mask(chosen, bits, n))
    if not 2 <= len(target) <= SET_MAX:
        return None
    order = [c for c in chosen if c is not trait]
    rng.shuffle(order)
    for c in order:
        trial = [x for x in chosen if x is not c]
        if groups(_mask(trial, bits, n)) == target:
            chosen = trial
    without = [x for x in chosen if x is not trait]
    if not without or groups(_mask(without, bits, n)) == target:
        return None
    return chosen


def _decider(shape: Shape, sols: list[dict], hidden: int):
    """The stop test for adding clues: the question's answer is settled (one value, or a decided claim verdict)."""

    from role_set.generate.build import _norm
    from role_set.puzzle import _project, holds

    m = _norm(shape)
    if shape.claim is not None:
        truths = [holds(m, s, shape.claim) for s in sols]

        def decided(mask: int) -> bool:
            seen = {truths[i] for i in range(len(sols)) if mask >> i & 1}
            return len(seen) == 1
        return decided
    projs = [_project(m, s) for s in sols]

    def decided(mask: int) -> bool:
        seen = {projs[i] for i in range(len(sols)) if mask >> i & 1}
        return len(seen) == 1
    return decided
