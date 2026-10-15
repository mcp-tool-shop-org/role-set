"""Trait recipes: the shape a lesson's puzzles take, the clue forms they draw on, the trait clue that must stay on the
deciding path, and the shortcut readings that must answer differently.

Slice 1, first recipe: T10 (takes statements literally). The trait clue is a count ("exactly / at least / at most k of
them are v"), and the readings are R&D's four size loosenings. A puzzle is accepted only when the count survives the
reduction and the trait clue's own size reading, applied to that clue alone, gives a different answer.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from role_set.generate.build import (MAX_TRIES, BuildError, Built, Shape, _key, _mask, assignments, candidates, check_model,
                                     deletion_order, metrics, reduce, truth)
from role_set.generate.shortcuts import REGISTRY, answer, differs, prove
from role_set.generate.t10_variety import cell_for, clues_if_shape
from role_set.puzzle import PuzzleError

# Domain packs: plural, names, one attribute, values. Each name is a whole word no other name in the pack contains.
# lamps, cups and doors are the slice-1 baseline (the tests use them). The rest are everyday domains in that shape.
# A new pack carries a sense block: the clues, counts and wording a generator may say. The stated facts still decide
# which assignments are possible. The builder leaves the block unread.
PACKS = {
    "lamps": {"plural": "lamps", "names": ["the brass lamp", "the copper lamp", "the tin lamp", "the iron lamp", "the lead lamp"],
              "attribute": "light", "values": ["lit", "dark"]},
    "cups": {"plural": "cups", "names": ["the blue cup", "the green cup", "the grey cup", "the white cup", "the yellow cup"],
             "attribute": "heat", "values": ["hot", "warm", "cold"]},
    "doors": {"plural": "doors", "names": ["the oak door", "the pine door", "the ash door", "the elm door", "the yew door"],
              "attribute": "state", "values": ["open", "shut"]},
    "pans": {"plural": "pans", "names": ["the stock pan", "the sauce pan", "the milk pan", "the stew pan", "the jam pan"],
             "attribute": "heat", "values": ["boiling", "simmering", "cold"],
             "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                       "counts": {"min": 1, "max_share": 0.75}}},
    "machines": {"plural": "machines", "names": ["the drill press", "the band saw", "the wood lathe", "the bench grinder",
                                                 "the thickness planer"],
                 "attribute": "motor", "values": ["running", "idle", "jammed"],
                 "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                           "counts": {"min": 1, "max_share": 0.75}}},
    "trains": {"plural": "trains", "names": ["the coastal train", "the valley train", "the harbour train", "the meadow train",
                                             "the summit train"],
               "attribute": "punctuality", "values": ["on time", "delayed", "cancelled"],
               "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                         "counts": {"min": 1, "max_share": 0.75}}},
    "beds": {"plural": "beds", "names": ["the rose bed", "the herb bed", "the lily bed", "the fern bed", "the moss bed"],
             "attribute": "moisture", "values": ["damp", "dry", "waterlogged"],
             "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                       "counts": {"min": 1, "max_share": 0.75}}},
    "clinics": {"plural": "clinics", "names": ["the asthma clinic", "the dental clinic", "the eye clinic", "the skin clinic",
                                               "the fracture clinic"],
                "attribute": "session", "values": ["morning", "afternoon", "evening"],
                "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                          "counts": {"min": 1, "max_share": 0.75}}},
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
    # "Which of them could be lit?": the answer is every possible group, so a loosened size word changes the set.
    return Shape(names, attrs, asks={"entity_with": {pack["attribute"]: pack["values"][0]}},
                 structure="underdetermined_with_explicit_set")


T10_QUESTIONS = ("set", "claim")
# R&D on #48, ruling 2: a single-value question never carries a T10 reading. A loosened reading keeps the key among its
# answers, and a tightened one gives the key or a contradiction, so the builder refuses it at once rather than search.
T10_NO_VALUE = ("T10 builds no single-value questions: a loosened size word keeps the key among the possible answers, and a "
                "tightened one gives the key or a contradiction, so no reading can decide; ask 'set' or 'claim'")


SET_MAX = 4  # possible groups in an open-set answer; more reads as a listing exercise


def _groups(shape: Shape, sols: list[dict]):
    from role_set.generate.build import _norm
    from role_set.puzzle import _project

    m = _norm(shape)
    projs = [_project(m, s) for s in sols]
    return lambda mask: frozenset(projs[i] for i in range(len(sols)) if mask >> i & 1)


def _if_count(clues: list[list]) -> int:
    return sum(1 for c in clues if c[0] == "if")


def _addition_order(others: list[list], rng: random.Random, if_target: int, if_shape: str | None = None) -> list[list]:
    """If clues are offered before unit clues, so the cell's if count and if_shape can be in the set before the answer
    settles. Reduction may still delete one; that attempt is refused and the seed tries again."""

    ifs = [c for c in others if c[0] == "if"]
    units = [c for c in others if c[0] != "if"]
    rng.shuffle(ifs)
    rng.shuffle(units)
    if not if_target or if_shape == "none":
        return units
    if if_shape is not None:
        if if_target == 1:
            ifs = [c for c in ifs if clues_if_shape([c]) == if_shape]
        elif if_target == 2:
            chosen_ifs: list[list] = []
            for c1 in ifs:
                partners = [c2 for c2 in ifs if c2 is not c1 and clues_if_shape([c1, c2]) == if_shape]
                if partners:
                    chosen_ifs = [c1, *partners]
                    break
            ifs = chosen_ifs
    return ifs + units


def build_t10(pack_name: str, seed: int, question: str, n: int = 4, tries: int = MAX_TRIES,
              cell: dict | None = None) -> Built:
    """One T10 puzzle from a seed: a count clue on the deciding path, every fact needed, no lookup, and the trait clue's
    own size reading answers differently. `question` is "set" (which could be ...?) or "claim".

    The cell fixes how many entities, the count word, k, and how many if clues (plus if_shape on claims) the reduced
    puzzle must keep. `n` stays in the signature for callers that pass it; the cell is what assigns the entity count.
    Pass `cell` only to probe one cell. A claim cell is "exactly": "at least" and "at most" keep a claim's verdict. The
    hidden solution is drawn from the assignments where that count clue is true, and not from the ones where the true
    count is exactly k: an at-least or at-most puzzle whose count is always exactly k would teach the misreading T10
    exists to train out."""

    if question not in T10_QUESTIONS:
        raise BuildError(T10_NO_VALUE if question == "value" else f"T10 question is one of {', '.join(T10_QUESTIONS)}")
    pack = PACKS[pack_name]
    spec = cell if cell is not None else cell_for(pack_name, question, seed)
    _ = n
    rng = random.Random(f"T10:{pack_name}:{question}:{seed}")  # each pack and question its own stream
    last = "no attempt"
    attr = pack["attribute"]
    op, k, if_target, size = spec["op"], spec["k"], spec["ifs"], spec["n"]
    if_shape = spec.get("if_shape")
    if question == "claim" and op != "=":
        raise BuildError(f"T10 ({pack_name}): a claim tests exactly; at least and at most do not change its verdict")
    for attempt in range(1, tries + 1):
        shape = t10_shape(pack, size, rng, question)
        sols = assignments(shape)
        pool = candidates(shape, T10.forms)
        bits = {_key(c): truth(shape, sols, c) for c in pool}
        value = rng.choice(pack["values"])
        trait = next(c for c in pool if c == ["count", attr, value, op, k])
        satisfying = [i for i in range(len(sols)) if bits[_key(trait)] >> i & 1]
        if not satisfying:
            last = "the count is true of no assignment"
            continue
        hidden = rng.choice(satisfying)  # the count is true, and a larger or smaller count may be too
        true_pool = [c for c in pool if bits[_key(c)] >> hidden & 1]
        # The trait is the only count clue. A second one can pass the shortcut check without the trait's own reading.
        others = _addition_order([c for c in true_pool if c is not trait and c[0] != "count"], rng, if_target, if_shape)
        if shape.structure == "underdetermined_with_explicit_set":
            kept = _open_set(shape, sols, bits, trait, others, rng, if_target)
            if kept is None:
                last = "no clue set left a small open set that keeps the count and the if count"
                continue
            built = _finish(shape, kept, trait, hidden, seed, attempt)
            if isinstance(built, str):
                last = built
                continue
            return built
        decided = _decider(shape, sols, hidden)
        chosen = [trait]
        rest: list[list] = []
        width = len(sols)
        for c in others:
            if decided(_mask(chosen, bits, width)) and _if_count(chosen) >= if_target:
                break
            if c[0] == "if" and _if_count(chosen) >= min(if_target, MAX_IF):
                continue
            # The trait clue stays on the deciding path by construction: a clue joins only while the clues without the
            # count still leave the answer open.
            if decided(_mask(rest + [c], bits, width)):
                continue
            if _mask(chosen + [c], bits, width) != _mask(chosen, bits, width):
                chosen.append(c)
                rest.append(c)
        if _if_count(chosen) != if_target:
            last = "the clues never held the cell's if count"
            continue
        if not decided(_mask(chosen, bits, width)):
            last = "the clues never settled the answer"
            continue
        kept = reduce(shape, sols, chosen, bits, decided, rng, keep=trait)
        if trait not in kept:
            last = "the count dropped out in reduction"
            continue
        if _if_count(kept) != if_target:
            last = "reduction did not keep the cell's if count"
            continue
        if if_shape is not None and clues_if_shape(kept) != if_shape:
            last = "reduction did not keep the cell's if shape"
            continue
        built = _finish(shape, kept, trait, hidden, seed, attempt)
        if isinstance(built, str):
            last = built
            continue
        return built
    raise BuildError(f"T10 ({pack_name}, seed {seed}): no puzzle in {tries} tries; last refusal: {last}")


def _trait_reading_differs(model: dict, structure: str, trait: list) -> bool:
    """A registered reading whose operator is the trait's, rewritten on the trait clue alone, changes the answer.

    `prove` still rewrites every count clue of that operator and is what the receipt stores. This check is the one the
    build has to pass: another count clue's misreading is not the trait (class trait_not_tested)."""

    op = trait[3]
    key = None
    for name, spec in REGISTRY.items():
        if spec.get("rewrite") != "count_op" or spec.get("from") != op:
            continue
        lazy_model = copy.deepcopy(model)
        hit = False
        for fact in lazy_model["facts"]:
            rule = fact["rule"]
            if rule == trait:
                fact["rule"] = [*rule[:3], spec["to"], rule[4]]
                hit = True
                break
        if not hit:
            continue
        if key is None:
            key = answer(model, structure)
        try:
            lazy = answer(lazy_model, structure)
        except PuzzleError as exc:
            lazy = {"kind": "none", "value": str(exc)}
        if differs(key, lazy):
            return True
    return False


def _finish(shape: Shape, kept: list[list], trait: list, hidden: int, seed: int, attempt: int) -> Built | str:
    """The solver's own checks on the reduced model, then the shortcut proof; a refusal comes back as its reason."""

    model = shape.model(kept)
    try:
        solved = check_model(model, shape.structure)
    except PuzzleError as e:
        return str(e)
    proofs = prove(model, shape.structure, list(T10.shortcuts))
    if not any(p["differs"] for p in proofs.values()):
        return "no size loosening changed the answer"
    if not _trait_reading_differs(model, shape.structure, trait):
        return "the trait clue's own size word did not change the answer"
    return Built(model=model, answer=solved, hidden=hidden, trait_clue=trait, shortcuts=proofs, metrics=metrics(model),
                 seed=seed, tries=attempt)


def _open_set(shape: Shape, sols: list[dict], bits: dict, trait: list, others: list[list], rng: random.Random,
              if_target: int):
    """Clues for an open-set question: add until 2 to SET_MAX groups stay possible, drop every clue the set doesn't need,
    and keep the count only if the set changes without it. The reduced set must still hold the cell's if count."""

    groups = _groups(shape, sols)
    width = len(sols)
    chosen = [trait]
    for c in others:
        g = groups(_mask(chosen, bits, width))
        if 2 <= len(g) <= SET_MAX and _if_count(chosen) >= if_target:
            break
        if c[0] == "if" and _if_count(chosen) >= min(if_target, MAX_IF):
            continue
        if len(groups(_mask(chosen + [c], bits, width))) >= 2 and _mask(chosen + [c], bits, width) != _mask(chosen, bits, width):
            chosen.append(c)
    target = groups(_mask(chosen, bits, width))
    if not 2 <= len(target) <= SET_MAX or _if_count(chosen) < if_target:
        return None
    for c in deletion_order(chosen, rng, keep=trait):
        trial = [x for x in chosen if x is not c]
        if groups(_mask(trial, bits, width)) == target:
            chosen = trial
    without = [x for x in chosen if x is not trait]
    if not without or groups(_mask(without, bits, width)) == target:
        return None
    if _if_count(chosen) != if_target:
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


# ---- T11: what a mistake costs; look for the falsifier before calling a claim supported (claim-keyed) --------------
# R&D, 2026-10-11. The lazy reading is "consistent, so supported" on the lines that name the claim's subject.
# - Find side: the key is unsupported, the lazy reading says supported, and no single line refutes the claim alone.
# - Confirm side: the key is supported, proved by a recorded contrast: a one-slot change to a line that doesn't name the
#   claim's subject, into one that doesn't, moves the verdict off supported (R&D's Q3 as amended on role-set#52; a flip
#   to unsupported is impossible there, since only subject lines can exclude the claim's value).
# No count clue: a count is T10's trait, and an item it decides tests T10 (round 3's T11-B-03 and B-05, the second a
# repeat of T10-B-01's skeleton).
# Both sides draw on a catalogue of proven skeletons (scripts/t11_catalogue.py), so a file's seeds take distinct
# skeletons where the puzzle is made (R&D's variety floor on role-set#52).
T11 = Recipe("T11", "", tuple(n for n, s in REGISTRY.items() if s["lesson"] == "T11"),
             ("is", "is_not", "in", "not_in", "if"))
T11_SIDES = ("find", "confirm")
T11_SIDES_BUILT = T11_SIDES  # the sides the catalogue covers
T11_FILE = 20  # seeds per file; within a file, each pack and seed of a value-count stratum takes its own skeleton


def t11_refusal(model: dict, side: str) -> str | None:
    """Why a claim model can't stand as a T11 item on this side (None when it can). The builder's checks, also run on
    hand-written items: the regression fixtures go through here."""

    from role_set.generate.shortcuts import subject_reading
    from role_set.puzzle import Table, parse_model

    m = parse_model(model)
    if m.claim is None:
        return "T11 items are claims"
    if any(r[0] == "count" for _, r in m.facts):
        return "a count clue decides it, and a count is T10's trait; build the chain on is, is_not, in, not_in and if"
    t = Table(m)
    v = t.verdict(t.all)
    want = "unsupported" if side == "find" else "supported"
    if v != want:
        return f"the {side} side needs a claim the facts make {want}; this one is {v}"
    for line, r in m.facts:
        alone = Table(parse_model({**model, "facts": [{"line": line, "rule": r}]}))
        if alone.verdict(alone.all) == v:
            return f"line {line} decides the claim on its own; T11 needs the search across lines"
    if side == "find" and subject_reading(model)["value"] == v:
        return "the lines that name the claim's subject already refute it, so the falsifier is not across lines"
    return None


def _leaves(rule: list) -> list:
    out = []
    for part in rule:
        if isinstance(part, list) and part and part[0] in T11.forms:
            out += _leaves(part)  # the two clauses of an `if`
        else:
            out.append(tuple(part) if isinstance(part, list) else part)  # a value pair is one leaf
    return out


def _one_slot(a: list, b: list) -> bool:
    """Whether two rules differ in exactly one leaf: a form word, an entity, a value or value pair, in either clause of
    an `if`. "If the oak door is open, the elm door is open" against "..., the elm door is shut" is one slot."""

    la, lb = _leaves(a), _leaves(b)
    return len(la) == len(lb) and sum(x != y for x, y in zip(la, lb)) == 1


def t11_pool(shape: Shape) -> list[list]:
    """T11's clues over a shape: the builder's forms, with `if` antecedents in `is` or `is_not`."""

    pool = candidates(shape, T11.forms)
    simple = [r for r in pool if r[0] in ("is", "is_not")]
    return pool + [["if", p, q] for p in simple for q in simple if p[0] == "is_not" and p[1] != q[1]]


def t11_contrast(shape: Shape, sols: list[dict], kept: list[list], pool: list[list], bits: dict, claim_true: list[bool],
                 rng: random.Random) -> dict | None:
    """The contrast a confirm item is proved by and counts under in the selector: a one-slot change to one kept clue
    that doesn't name the claim's subject, into a clue that doesn't either, which leaves an assignment and moves the
    verdict off supported. `contrast_verdict` records where it moves (R&D's Q3 as amended on role-set#52). None when no
    such change exists."""

    from role_set.generate.shortcuts import names_in

    n = len(sols)
    ents = set(shape.entities)
    subject = names_in(shape.claim, ents)
    found = []
    for i, fact in enumerate(kept):
        if names_in(fact, ents) & subject:
            continue
        rest = _mask(kept[:i] + kept[i + 1:], bits, n)
        for c in pool:
            if not _one_slot(fact, c) or names_in(c, ents) & subject:
                continue
            mask = rest & bits[_key(c)]
            truths = {claim_true[j] for j in range(n) if mask >> j & 1}
            if truths and truths != {True}:
                found.append({"line": i + 2, "form": fact[0], "from": fact, "to": c, "contrast_names_subject": False,
                              "contrast_verdict": "unsupported" if truths == {False} else "cannot_tell"})
    return rng.choice(found) if found else None


_T11_CAT: dict | None = None


def _t11_catalogue() -> dict:
    global _T11_CAT
    if _T11_CAT is None:
        import json
        from pathlib import Path

        _T11_CAT = json.loads(Path(__file__).with_name("t11_catalogue.json").read_text(encoding="utf-8"))
    return _T11_CAT


def t11_row(pack_name: str, side: str, seed: int) -> dict:
    """The catalogue row a T11 seed builds on. Packs with the same value count share one table, and in each file of
    T11_FILE seeds every pack and seed takes a different row; the next file draws a fresh order."""

    nv = len(PACKS[pack_name]["values"])
    rows = _t11_catalogue()["sides"][side][str(nv)]
    stratum = sorted(p for p in PACKS if len(PACKS[p]["values"]) == nv)
    file_index, offset = divmod(seed - 1, T11_FILE)
    order = list(range(len(rows)))
    random.Random(f"T11:{side}:{nv}:file{file_index}").shuffle(order)
    return rows[order[(stratum.index(pack_name) * T11_FILE + offset) % len(rows)]]


def _t11_from_row(pack_name: str, side: str, seed: int, row: dict | None = None) -> Built:
    """Instantiate a catalogue row with the pack's names and values, in a shuffled line order, and prove it again."""

    from role_set.generate.build import _norm
    from role_set.puzzle import holds

    pack = PACKS[pack_name]
    cat = _t11_catalogue()
    row = row or t11_row(pack_name, side, seed)
    rng = random.Random(f"T11:{pack_name}:{side}:{seed}")
    names = rng.sample(pack["names"], rng.choice((3, 4, 5)))
    ents = dict(zip(cat["head"]["names"], names[:3]))
    vals = dict(zip(cat["head"]["values"][str(len(pack["values"]))], rng.sample(pack["values"], len(pack["values"]))))
    attr = pack["attribute"]

    def place(r):
        if isinstance(r, list) and r and r[0] == "if":
            return ["if", place(r[1]), place(r[2])]
        out = [r[0], ents[r[1]], attr]
        if isinstance(r[3], list):  # a value pair, written in the pack's value order as the clue pool writes it
            return out + [sorted((vals[v] for v in r[3]), key=pack["values"].index)]
        return out + [vals[r[3]]]

    facts = [place(r) for r in row["facts"]]
    rng.shuffle(facts)
    claim = place(cat["head"]["claim"])
    shape = Shape(tuple(names), {attr: {"values": list(pack["values"]), "distinct": False}}, claim=claim,
                  structure="falsifier_first")
    model = shape.model(facts)
    where = f"T11 ({pack_name}, {side}, seed {seed}): catalogue row {row['skeleton_sha256']}"
    why = t11_refusal(model, side)
    if why:
        raise BuildError(f"{where} refused: {why}")
    answer = check_model(model, shape.structure)
    proofs = prove(model, shape.structure, list(T11.shortcuts))
    sols = assignments(shape)
    m = _norm(shape)
    hidden = next(i for i, s in enumerate(sols) if all(holds(m, s, f) for f in facts))
    contrast = None
    if side == "confirm":
        pool = t11_pool(shape)
        bits = {_key(c): truth(shape, sols, c) for c in pool}
        contrast = t11_contrast(shape, sols, facts, pool, bits, [holds(m, s, claim) for s in sols], rng)
        if contrast is None:
            raise BuildError(f"{where} has no off-subject contrast")
    return Built(model=model, answer=answer, hidden=hidden, trait_clue=[], shortcuts=proofs, metrics=metrics(model),
                 seed=seed, tries=1, notes=[f"t11 catalogue row {row['skeleton_sha256']}"], contrast=contrast)


def build_t11(pack_name: str, seed: int, side: str) -> Built:
    """One T11 puzzle: a claim decided only by reading across lines. `side` is "find" (the claim is refuted, and the
    lines naming its subject alone would pass it) or "confirm" (the claim holds, with a recorded contrast). The seed
    picks a catalogue row (`t11_row`), so a file's seeds take distinct skeletons by construction."""

    if side not in T11_SIDES:
        raise BuildError(f"T11 side is one of {', '.join(T11_SIDES)}")
    return _t11_from_row(pack_name, side, seed)


# ---- T12: which edition was in force on a date (R&D's shape 1, solved on `contains`) -------------------------------
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
T12_PACKS = {
    "issues": {"plural": "issues", "singular": "issue", "names": ["issue 3", "issue 4", "issue 5", "issue 6", "issue 7", "issue 8"]},
    "cards": {"plural": "cards", "singular": "card", "names": ["the red card", "the white card", "the black card", "the green card"]},
    "sheets": {"plural": "sheets", "singular": "sheet", "names": ["sheet A", "sheet B", "sheet C", "sheet D"]},
    "editions": {"plural": "editions", "singular": "edition",
                 "names": ["the 8th edition", "the 9th edition", "the 10th edition", "the 11th edition", "the 12th edition"]},
}
# Span wording: each sentence names exactly its rule's names, and direction words appear only on before/after.
T12_TEMPLATES = {
    "is_not": ["{X} was not in force from {v}."],
    "in": ["{X} was in force from {v1} or from {v2}."],
    "not_in": ["{X} was in force neither from {v1} nor from {v2}."],
    "before": ["{X} came into force before {y}."],
    "after": ["{X} came into force after {y}."],
    "adjacent": ["{X} and {y} were in force back to back."],
    "not_adjacent": ["{X} and {y} were never in force back to back."],
    "offset1": ["{Y} came into force next on from {x}."],
    "offset2": ["{Y} came into force two spans on from {x}."],
}
T12 = Recipe("T12", "", tuple(n for n, s in REGISTRY.items() if s["lesson"] == "T12"),
             ("is_not", "in", "not_in", "before", "after", "adjacent", "not_adjacent", "offset"))


def _label(day: int, month: int) -> str:
    return f"{day} {MONTHS[month - 1]}"


def t12_calendar(n: int, rng: random.Random, position: int | None = None) -> tuple[list[str], dict, str]:
    """n contiguous spans of 2 to 4 whole months from a random start month, an asked date inside one of them (now and
    then a boundary day, so the boundary reading has something to act on), and the typed day records. `position` names
    the span that holds the asked date (0 the earliest, n - 1 the newest); by default it is drawn at random."""

    lengths = [rng.randint(2, 4) for _ in range(n)]
    while sum(lengths) > 12:
        lengths[lengths.index(max(lengths))] -= 1
    start = rng.randint(1, 12)
    spans, cursor = [], 0  # cursor counts months from the start month; year = cursor // 12 on the wrap
    records: dict[str, tuple] = {}
    for ln in lengths:
        m0 = (start - 1 + cursor) % 12 + 1
        m1 = (start - 1 + cursor + ln - 1) % 12 + 1
        y0, y1 = (start - 1 + cursor) // 12, (start - 1 + cursor + ln - 1) // 12
        lo, hi = _label(1, m0), _label(MONTH_DAYS[m1 - 1], m1)
        records[lo], records[hi] = (y0, m0, 1), (y1, m1, MONTH_DAYS[m1 - 1])
        spans.append(f"{lo} to {hi}")
        cursor += ln
    pick = rng.randrange(n) if position is None else position
    lo_label, hi_label = spans[pick].split(" to ")
    if rng.random() < (0.6 if pick == n - 1 else 0.3):
        # A boundary day: a closing day goes to the next span under the boundary reading, an opening day to the one
        # before. The last span has no next, so it is asked on its opening day, and more often: in a pack without
        # numbered names that is the only way a last-span answer has a reading to differ, and without it the newest
        # edition was almost never the answer, which would make "never the newest" a shortcut of its own.
        asked = rng.choice(([hi_label] if pick < n - 1 else []) + ([lo_label] if pick > 0 else []))
    else:
        y, mth, _ = records[lo_label]
        month_off = rng.randrange(lengths[pick])
        mm = (mth - 1 + month_off) % 12 + 1
        yy = y + (mth - 1 + month_off) // 12
        dd = rng.choice([d for d in (6, 9, 14, 17, 20, 23) if d <= MONTH_DAYS[mm - 1]])
        asked = _label(dd, mm)
        records.setdefault(asked, (yy, mm, dd))
    wrap = any(r[0] for r in records.values())
    ordered = sorted(records.items(), key=lambda kv: kv[1])
    days = [{"label": lab, "day": d, "month": mm, "ordinal": i, **({"year": y} if wrap else {})}
            for i, (lab, (y, mm, d)) in enumerate(ordered)]
    return spans, {"attribute": "span", "days": days, "bounds": {s: s.split(" to ") for s in spans}}, asked


def build_t12(pack_name: str, seed: int, n: int = 3, tries: int = MAX_TRIES, position: int | None = None) -> Built:
    """One T12 puzzle: which edition was in force on a date. The clues map editions to spans (never stating one outright),
    the solver matches the date, and at least one of R&D's four T12 readings names a different edition.

    `position` asks for the answer's span (n - 1 is the newest edition). The keep filter rejects a puzzle no reading
    differs on, and that rejects more newest-edition answers than others, so a batch that leaves the position to chance
    under-supplies them (R&D's filter_induced_shortcut, on role-set#50); the selector floors positions per file, and this
    is how a batch asks for the ones it is short of."""

    if position is not None and not 0 <= position < n:
        raise BuildError(f"position is a span index from 0 to {n - 1}")
    pack = T12_PACKS[pack_name]
    rng = random.Random(f"T12:{pack_name}:{n}:{seed}" + ("" if position is None else f":p{position}"))
    last = "no attempt"
    for attempt in range(1, tries + 1):
        spans, span, asked = t12_calendar(n, rng, position)
        names = tuple(rng.sample(pack["names"], n))
        shape = Shape(names, {"span": {"values": spans, "ordered": True}}, asks={"contains": asked}, span=span)
        sols = assignments(shape)
        pool = candidates(shape, T12.forms)
        bits = {_key(c): truth(shape, sols, c) for c in pool}
        hidden = rng.randrange(len(sols))
        others = [c for c in pool if bits[_key(c)] >> hidden & 1]
        rng.shuffle(others)
        decided = _decider(shape, sols, hidden)
        chosen: list[list] = []
        for c in others:
            if chosen and decided(_mask(chosen, bits, len(sols))):
                break
            if _mask(chosen + [c], bits, len(sols)) != _mask(chosen, bits, len(sols)):
                chosen.append(c)
        if not chosen or not decided(_mask(chosen, bits, len(sols))):
            last = "the clues never settled the edition"
            continue
        kept = reduce(shape, sols, chosen, bits, decided, rng)
        model = shape.model(kept)
        try:
            answer = check_model(model, shape.structure)
        except PuzzleError as e:
            last = str(e)
            continue
        proofs = prove(model, shape.structure, list(T12.shortcuts))
        if not any(p["differs"] for p in proofs.values()):
            last = "no T12 reading named a different edition"
            continue
        return Built(model=model, answer=answer, hidden=hidden, trait_clue=[], shortcuts=proofs, metrics=metrics(model),
                     seed=seed, tries=attempt)
    raise BuildError(f"T12 ({pack_name}, seed {seed}): no puzzle in {tries} tries; last refusal: {last}")
