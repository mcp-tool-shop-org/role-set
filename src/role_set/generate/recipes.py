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
from role_set.generate.t10_variety import cell_for
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
             "attribute": "moisture", "values": ["damp", "dry", "frozen"],
             "sense": {"forms": ["is", "is_not", "in", "not_in", "count", "if"],
                       "counts": {"min": 1, "max_share": 0.75}}},
    "clinics": {"plural": "clinics", "names": ["the asthma clinic", "the dental clinic", "the eye clinic", "the skin clinic",
                                               "the fracture clinic"],
                "attribute": "session", "values": ["open", "closed", "on standby"],
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


def _addition_order(others: list[list], rng: random.Random, if_target: int) -> list[list]:
    """If clues are offered before unit clues, so the cell's if count can be in the set before the answer settles.
    Reduction may still delete one; that attempt is refused and the seed tries again."""

    ifs = [c for c in others if c[0] == "if"]
    units = [c for c in others if c[0] != "if"]
    rng.shuffle(ifs)
    rng.shuffle(units)
    return ifs + units if if_target else units


def build_t10(pack_name: str, seed: int, question: str, n: int = 4, tries: int = MAX_TRIES,
              cell: dict | None = None) -> Built:
    """One T10 puzzle from a seed: a count clue on the deciding path, every fact needed, no lookup, and the trait clue's
    own size reading answers differently. `question` is "set" (which could be ...?) or "claim".

    The cell fixes how many entities, the count word, k, and how many if clues the reduced puzzle must keep. `n` stays in
    the signature for callers that pass it; the cell is what assigns the entity count. Pass `cell` only to probe one
    cell. A claim cell is "exactly": "at least" and "at most" keep a claim's verdict. The hidden solution is drawn from
    the assignments where that count clue is true, and not from the ones where the true count is exactly k: an at-least
    or at-most puzzle whose count is always exactly k would teach the misreading T10 exists to train out."""

    if question not in T10_QUESTIONS:
        raise BuildError(T10_NO_VALUE if question == "value" else f"T10 question is one of {', '.join(T10_QUESTIONS)}")
    pack = PACKS[pack_name]
    spec = cell if cell is not None else cell_for(pack_name, question, seed)
    _ = n
    rng = random.Random(f"T10:{pack_name}:{question}:{seed}")  # each pack and question its own stream
    last = "no attempt"
    attr = pack["attribute"]
    op, k, if_target, size = spec["op"], spec["k"], spec["ifs"], spec["n"]
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
        others = _addition_order([c for c in true_pool if c is not trait and c[0] != "count"], rng, if_target)
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
