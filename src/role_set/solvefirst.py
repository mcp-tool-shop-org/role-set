"""Solve-first kind B (BUILD-SPEC B3a): the puzzle's model and answer are made in code, before the writer sees them.

For each of the plan's eight puzzle structures, a proving generator builds the locked model from the slot's seed and
pack. It picks a hidden assignment, draws facts true of it, adds facts until the solver leaves exactly the intended
answer (or, for underdetermined, two or three possibilities), and prunes every fact that changes nothing. `solve` is
the final oracle; a lookup, or any refusal, means another try. The writer then renders the prose from the locked model
and writes the thinking and the reply. It never writes the model or the answer: they are injected, and `check_puzzle`
runs unchanged (the text states exactly the model's facts, the key is the solver's, the reply ends on it).

The families (R&D, 2026-10-16): order and interval -> scheduling_with_exclusions; resource allocation ->
constraint_grid; dependency DAG -> tree_with_pruning; elimination -> elimination_from_a_set; underdetermined ->
underdetermined_with_explicit_set; stopping -> stopping_problem; falsifier_first -> falsifier_first; when_to_stop ->
when_to_stop.

The answer arrives late (R&D's ANSWER_EARLY gate): the thinking may not pair the asked subject with the keyed answer
before its final third, and it cites every fact line of the locked model ("line N"). The packs (b_packs.json) are row
inputs written by Claude, never targets.

Standards: PIN_PER_STEP (the slot's seed, the pack and the model's sha256 go on the receipt), ANDON_AUTHORITY (a
structure the generator can't prove within its tries refuses the slot). No model call here.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from role_set.puzzle import NEGATION, NEGATIVE, PuzzleError, Table, parse_model, rule_names, solve

PACKS_FILE = Path(__file__).with_name("b_packs.json")
FAMILIES = {
    "scheduling_with_exclusions": "order_and_interval",
    "constraint_grid": "resource_allocation",
    "tree_with_pruning": "dependency_dag",
    "elimination_from_a_set": "elimination",
    "underdetermined_with_explicit_set": "underdetermined",
    "stopping_problem": "stopping",
    "falsifier_first": "falsifier_first",
    "when_to_stop": "when_to_stop",
}
SOLVE_FIRST = tuple(FAMILIES)
MAX_TRIES = 300
# A puzzle is never one fact. An underdetermined puzzle narrows through at least three facts before it leaves its set,
# and a dependency claim is decided by a chain of at least three.
MIN_FACTS = {"underdetermined_with_explicit_set": 3, "tree_with_pruning": 3}
MIN_FACTS_DEFAULT = 2
DAG_FORMS = ("before", "after", "adjacent", "not_adjacent")
PACK_USES_MAX = 4  # across a batch's plan (R&D, 2026-10-16)
# The plan-time skeleton floor (R&D on #73): per structure, across the batch and any earlier batches' state, at least
# 75% of solve-first puzzles have a distinct fact skeleton and none is used more than 3 times. A slot whose skeleton
# collides is re-seeded, up to RESEED_MAX times, and the count is recorded on the slot.
SKELETON_DISTINCT_SHARE = 0.75
SKELETON_USES_MAX = 3
RESEED_MAX = 30
LATE_SHARE = 2 / 3  # the answer may appear only in the thinking's final third
LINE_REF = re.compile(r"\blines?\s+(\d+)(?:\s*(?:,|and|&)\s*(\d+))*", re.IGNORECASE)


class SolveFirstError(ValueError):
    """The generator could not prove a puzzle for this slot within its tries."""


@dataclass(frozen=True)
class Locked:
    """A proven puzzle: the model as the writer must state it, the solver's key, and how it was made."""

    model: dict
    key: str
    question: str  # what the Question line asks, in words
    glosses: dict  # line number -> the rule in plain words
    structure: str
    family: str
    pack: str
    seed: int
    tries: int
    subject: str | None  # the asked entity (or None for a claim)
    keyed: tuple  # the keyed value(s) the thinking may not reach early

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.model, sort_keys=True).encode("utf-8")).hexdigest()

    def receipt(self) -> dict:
        return {"b_path": "solve_first", "family": self.family, "pack": self.pack, "seed": self.seed, "tries": self.tries,
                "model_sha256": self.sha256}


def load_packs(path: Path = PACKS_FILE) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    packs = {p["id"]: p for p in data["packs"]}
    codes = [p["naics"] for p in packs.values()]
    if len(set(codes)) != len(codes):
        raise SolveFirstError("REFUSED: two packs share a NAICS code; each pack has its own")
    return packs


# ---- plain words for each rule ------------------------------------------------------------------------------
def _ref(r) -> str:
    if isinstance(r, dict):
        (a, v), = r.items()
        return f"the one with {a} {v}"
    return str(r)


def gloss(rule) -> str:
    """The rule in words the solver's text check accepts: a negative rule says 'not', a before/after rule names its
    first argument first, and nothing else uses before, after, earlier or later."""

    f = rule[0]
    if f == "is_not":
        return f"{_ref(rule[1])}'s {rule[2]} is not {rule[3]}"
    if f == "not_in":
        return f"{_ref(rule[1])}'s {rule[2]} is neither " + " nor ".join(rule[3])
    if f == "in":
        return f"{_ref(rule[1])}'s {rule[2]} is " + " or ".join(rule[3])
    if f == "same":
        return f"{_ref(rule[1])} is {_ref(rule[2])}"
    if f == "different":
        return f"{_ref(rule[1])} is not {_ref(rule[2])}"
    if f == "before":
        return f"{_ref(rule[1])}'s {rule[3]} is earlier than {_ref(rule[2])}'s"
    if f == "after":
        return f"{_ref(rule[1])}'s {rule[3]} is later than {_ref(rule[2])}'s"
    if f == "adjacent":
        return f"{_ref(rule[1])} and {_ref(rule[2])} have neighbouring {rule[3]}s"
    if f == "not_adjacent":
        return f"{_ref(rule[1])} and {_ref(rule[2])} do not have neighbouring {rule[3]}s"
    raise SolveFirstError(f"no gloss for rule form {f!r}")


def declare(attr: str, values: list[str], ents: list[str], ordered: bool) -> str:
    """A setting line that names every value of one attribute (and, on the first, every entity)."""

    vals = ", ".join(values[:-1]) + " and " + values[-1]
    who = ", ".join(ents[:-1]) + " and " + ents[-1]
    return f"{who} each have one {attr}: {vals}" + (", in that order" if ordered else "")


# ---- the generators ------------------------------------------------------------------------------------------
def _ordered_pool(sol, ents, attr, order) -> list:
    out = []
    pos = {e: order.index(sol[e]) for e in ents}
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            pa, pb = pos[a], pos[b]
            out.append(["before", a, b, attr] if pa < pb else ["after", a, b, attr])
            out.append(["adjacent", a, b, attr] if abs(pa - pb) == 1 else ["not_adjacent", a, b, attr])
        for v in order:
            if v != sol[a]:
                out.append(["is_not", a, attr, v])
    return out


def _grid_pool(sols: dict, ents, attrs: dict) -> list:
    out = []
    names = list(attrs)
    for a in ents:
        for name in names:
            for v in attrs[name]:
                if v != sols[name][a]:
                    out.append(["is_not", a, name, v])
            wrong = [v for v in attrs[name] if v != sols[name][a]]
            if len(wrong) >= 2:
                out.append(["not_in", a, name, sorted(wrong[:2])])
    if len(names) == 2:
        x, y = names
        for e in ents:
            for f in ents:
                ref1, ref2 = {x: sols[x][e]}, {y: sols[y][f]}
                out.append(["same", ref1, ref2] if e == f else ["different", ref1, ref2])
    return out


def _raw(ents, attrs: dict, ordered: str | None, facts: list, asks=None, claim=None) -> dict:
    flavour = []
    lines: dict[int, str] = {}
    n = 0
    for name, vals in attrs.items():
        n += 1
        flavour.append(n)
        lines[n] = declare(name, vals, ents, name == ordered)
    model = {"entities": list(ents),
             "attributes": {name: {"values": list(vals), **({"ordered": True} if name == ordered else {})} for name, vals in attrs.items()},
             "facts": [{"line": n + 1 + i, "rule": r} for i, r in enumerate(facts)], "flavour": flavour}
    if claim is not None:
        model["claim"] = claim
    else:
        model["asks"] = asks
    return model


def _grow_and_prune(rng, pool: list, build, goal) -> list | None:
    """Add pool facts in a shuffled order until goal(projections) holds, then drop every fact whose removal keeps
    exactly the same projections. None when the pool never reaches the goal."""

    rng.shuffle(pool)
    if not pool:
        return None
    try:
        m = parse_model(build(pool))
    except PuzzleError:
        return None
    t = Table(m)
    projs_of = t.projections if m.claim is None else (lambda mask: {t.verdict(mask)})
    mask = 0
    chosen: list[int] = []
    for i in range(len(pool)):
        mask |= 1 << i
        chosen.append(i)
        projs = projs_of(mask)
        if goal(projs):
            break
    else:
        return None
    target = projs_of(mask)
    for i in reversed(chosen):
        trial = mask & ~(1 << i)
        if trial and projs_of(trial) == target:
            mask = trial
    return [pool[i] for i in chosen if mask & (1 << i)]


def _scheduling(rng, pack, underdetermined: bool = False):
    n = rng.choice([4, 5])
    ents = rng.sample(pack["entities"], n)
    start = rng.randint(0, len(pack["order"]["values"]) - n)
    order = pack["order"]["values"][start:start + n]
    attr = pack["order"]["name"]
    sol = dict(zip(ents, rng.sample(order, n)))
    target = rng.choice(ents)
    asks = {"attribute": attr, "of": target}

    def build(facts):
        return _raw(ents, {attr: order}, attr, facts, asks=asks)

    goal = (lambda p: 2 <= len(p) <= 3) if underdetermined else (lambda p: len(p) == 1)
    facts = _grow_and_prune(rng, _ordered_pool(sol, ents, attr, order), build, goal)
    if facts is None:
        return None
    return build(facts), target, attr


def _grid(rng, pack):
    n = rng.choice([3, 4])
    ents = rng.sample(pack["entities"], n)
    kinds = pack["kinds"]
    attrs = {k["name"]: rng.sample(k["values"], n) for k in kinds}
    sols = {name: dict(zip(ents, rng.sample(vals, n))) for name, vals in attrs.items()}
    target = rng.choice(ents)
    asked = rng.choice(list(attrs))
    asks = {"attribute": asked, "of": target}

    def build(facts):
        return _raw(ents, attrs, None, facts, asks=asks)

    facts = _grow_and_prune(rng, _grid_pool(sols, ents, attrs), build, lambda p: len(p) == 1)
    return (build(facts), target, asked) if facts is not None else None


def _elimination(rng, pack):
    n = rng.choice([4, 5])
    ents = rng.sample(pack["entities"], n)
    kind = rng.choice(pack["kinds"])
    vals = rng.sample(kind["values"], n)
    sol = {kind["name"]: dict(zip(ents, rng.sample(vals, n)))}
    asked_value = rng.choice(vals)
    asks = {"entity_with": {kind["name"]: asked_value}}

    def build(facts):
        return _raw(ents, {kind["name"]: vals}, None, facts, asks=asks)

    facts = _grow_and_prune(rng, _grid_pool(sol, ents, {kind["name"]: vals}), build, lambda p: len(p) == 1)
    return (build(facts), asked_value, kind["name"]) if facts is not None else None


def _dag(rng, pack):
    n = rng.choice([5, 6])
    ents = rng.sample(pack["entities"], n)
    order = pack["order"]["values"][:n]
    attr = pack["order"]["name"]
    sol = dict(zip(ents, rng.sample(order, n)))
    pos = {e: order.index(sol[e]) for e in ents}
    a, b = rng.sample(ents, 2)
    form = rng.choice(["before", "after"])
    claim = [form, a, b, attr]
    holds = pos[a] < pos[b] if form == "before" else pos[a] > pos[b]
    want = "supported" if holds else "unsupported"
    # Order and neighbour facts (R&D, 2026-10-17): before/after alone fold to a handful of skeletons (3 in 84 seeds).
    # No fact of any form names both claim entities, so no single fact decides the claim.
    pool = [r for r in _ordered_pool(sol, ents, attr, order) if r[0] in DAG_FORMS and {r[1], r[2]} != {a, b}]

    def build(facts):
        return _raw(ents, {attr: order}, attr, facts, claim=claim)

    facts = _grow_and_prune(rng, pool, build, lambda p: p == {want})
    return (build(facts), None, attr) if facts is not None else None


def _stopping(rng, pack):
    n = rng.choice([4, 5])
    ents = rng.sample(pack["entities"], n)
    start = rng.randint(0, len(pack["order"]["values"]) - n)
    order = pack["order"]["values"][start:start + n]
    attr = pack["order"]["name"]
    sol = dict(zip(ents, rng.sample(order, n)))
    target = rng.choice(ents)
    asks = {"attribute": attr, "of": target}
    pool = [r for r in _ordered_pool(sol, ents, attr, order) if r[0] != "is_not" or r[1] != target]
    for a in ents:
        for w in order:
            if w != sol[a]:
                pool.append(["in", a, attr, sorted([sol[a], w])])

    def build(facts):
        return _raw(ents, {attr: order}, attr, facts, asks=asks)

    facts = _grow_and_prune(rng, pool, build, lambda p: len(p) == 1)
    if facts is None or not any(r[0] == "in" for r in facts):
        return None
    # A stopping puzzle's key is its final fact's line number, which the thinking cites as '(line N)'. Putting a
    # negative fact or one that doesn't name the asked entity last keeps that step from pairing subject and key early.
    safe = [i for i, r in enumerate(facts) if r[0] in NEGATIVE or target not in rule_names(r)]
    if not safe:
        return None
    last = safe[-1]
    if last != len(facts) - 1:
        facts = [r for i, r in enumerate(facts) if i != last] + [facts[last]]
    extra = rng.choice([0, 1, 2])
    rest = [r for r in pool if r not in facts]
    facts += rng.sample(rest, min(extra, len(rest)))
    return build(facts), target, attr


def _falsifier(rng, pack):
    n = rng.choice([4, 5])
    ents = rng.sample(pack["entities"], n)
    start = rng.randint(0, len(pack["order"]["values"]) - n)
    order = pack["order"]["values"][start:start + n]
    attr = pack["order"]["name"]
    sol = dict(zip(ents, rng.sample(order, n)))
    pos = {e: order.index(sol[e]) for e in ents}
    a, b = rng.sample(ents, 2)
    form = rng.choice(["before", "after"])
    if (form == "before" and pos[a] < pos[b]) or (form == "after" and pos[a] > pos[b]):
        a, b = b, a
    claim = [form, a, b, attr]
    asks = {"attribute": attr, "of": a}
    pool = [r for r in _ordered_pool(sol, ents, attr, order) if r[0] != "is_not" and {r[1], r[2]} != {a, b}]
    for e in ents:
        wrong = [v for v in order if v != sol[e]]
        for w in wrong:
            if not (e in (a, b) and order[0] not in (sol[e], w) and order[-1] not in (sol[e], w)):
                pool.append(["in", e, attr, sorted([sol[e], w])])
        if e in (a, b):
            wrong = [v for v in wrong if v not in (order[0], order[-1])]
        for i in range(len(wrong) - 1):
            pool.append(["not_in", e, attr, sorted([wrong[i], wrong[i + 1]])])

    def build(facts):
        m = _raw(ents, {attr: order}, attr, facts, asks=asks, claim=claim)
        m["asks"] = asks
        return m

    facts = _grow_and_prune(rng, pool, build, lambda p: p == {"unsupported"})
    if facts is None or not ({"in", "not_in"} <= {r[0] for r in facts}):
        return None
    return build(facts), None, attr


def _when_to_stop(rng, pack):
    n = rng.choice([4, 5])
    ents = rng.sample(pack["entities"], n)
    start = rng.randint(0, len(pack["order"]["values"]) - n)
    order = pack["order"]["values"][start:start + n]
    attr = pack["order"]["name"]
    sol = dict(zip(ents, rng.sample(order, n)))
    pos = {e: order.index(sol[e]) for e in ents}
    a, b = rng.sample(ents, 2)
    form = rng.choice(["before", "after"])
    claim = [form, a, b, attr]
    holds = pos[a] < pos[b] if form == "before" else pos[a] > pos[b]
    want = "supported" if holds else "unsupported"
    pool = [r for r in _ordered_pool(sol, ents, attr, order) if r[0] != "is_not" and {r[1], r[2]} != {a, b}]
    for e in ents:
        wrong = [v for v in order if v != sol[e]]
        if e in (a, b):
            wrong = [v for v in wrong if v not in (order[0], order[-1])]
        for i in range(len(wrong) - 1):
            pool.append(["not_in", e, attr, sorted([wrong[i], wrong[i + 1]])])

    def build(facts):
        return _raw(ents, {attr: order}, attr, facts, claim=claim)

    facts = _grow_and_prune(rng, pool, build, lambda p: p == {want})
    if facts is None or "not_in" not in {r[0] for r in facts}:
        return None
    return build(facts), None, attr


GENERATORS = {
    "scheduling_with_exclusions": lambda rng, pack: _scheduling(rng, pack),
    "underdetermined_with_explicit_set": lambda rng, pack: _scheduling(rng, pack, underdetermined=True),
    "constraint_grid": _grid,
    "elimination_from_a_set": _elimination,
    "tree_with_pruning": _dag,
    "stopping_problem": _stopping,
    "falsifier_first": _falsifier,
    "when_to_stop": _when_to_stop,
}


def _question(model: dict, structure: str, subject, attr: str) -> str:
    if "claim" in model:
        c = model["claim"]
        word = "earlier" if c[0] == "before" else "later"
        tail = " Name the lines that settle it, or say it can't be told." if structure == "tree_with_pruning" else ""
        return f"Is {c[1]}'s {attr} {word} than {c[2]}'s?{tail}"
    if structure == "stopping_problem":
        return f"After which numbered line is {subject}'s {attr} settled?"
    asks = model["asks"]
    if "entity_with" in asks:
        return f"Who has {attr} {subject}?"
    verb = "could" if structure == "underdetermined_with_explicit_set" else "does"
    return f"Which {attr} {verb} {subject} have?"


def generate(structure: str, pack: dict, seed: int) -> Locked:
    """The locked puzzle for one slot. Deterministic in (structure, pack, seed)."""

    if structure not in GENERATORS:
        raise SolveFirstError(f"no generator for {structure!r}; it stays freehand")
    rng = random.Random(f"b3:{structure}:{pack['id']}:{seed}")
    for tries in range(1, MAX_TRIES + 1):
        made = GENERATORS[structure](rng, pack)
        if made is None:
            continue
        model, subject, attr = made
        if len(model["facts"]) < MIN_FACTS.get(structure, MIN_FACTS_DEFAULT):
            continue
        try:
            got = solve(parse_model(model), structure, "")
        except PuzzleError:
            continue
        if got.lookup:
            continue
        lines = {f["line"]: gloss(f["rule"]) for f in model["facts"]}
        for n in model["flavour"]:
            name = list(model["attributes"])[n - 1]
            spec = model["attributes"][name]
            lines[n] = declare(name, spec["values"], model["entities"], spec.get("ordered", False))
        keyed = tuple(x.strip() for x in re.split(r",|;", got.answer) if x.strip())
        return Locked(model=model, key=got.answer, question=_question(model, structure, subject, attr), glosses=dict(sorted(lines.items())),
                      structure=structure, family=FAMILIES[structure], pack=pack["id"], seed=seed, tries=tries,
                      subject=subject, keyed=keyed)
    raise SolveFirstError(f"REFUSED: no provable {structure} puzzle from pack {pack['id']} at seed {seed} in {MAX_TRIES} tries")


# ---- the plan: a pack and a seed for every solve-first B slot ----------------------------------------------------
def assign(slots: list, packs: dict[str, dict], *, seed: int, table=None, prior: dict | None = None) -> None:
    """Every B slot gets a pack (distinct within a lesson, dealt evenly across the batch) and a seed; a solve-first
    structure also gets `b_path`. The pack's NAICS code is the slot's domain, titled from the table when one is given.
    A pack is used at most PACK_USES_MAX times across the regeneration: `prior` is B4's {pack: uses} from earlier
    batches, and the least-used packs are dealt first."""

    from collections import Counter

    rng = random.Random(f"packs:{seed}")
    earlier, dealt = dict(prior or {}), Counter()
    deck: list[str] = []
    titles = {e["code"]: e["title"] for e in table.subsectors} if table is not None else {}
    lessons: list[str] = []
    for s in slots:
        if s.lesson not in lessons:
            lessons.append(s.lesson)
    for lesson in lessons:
        used: set[str] = set()
        # A pack whose code another slot of the lesson already has is skipped: no two slots share a domain (B2).
        taken = {x.domain["code"] for x in slots if x.lesson == lesson and x.kind != "B" and x.domain}

        def free(p):
            # Cumulative across batches (Argon §9.4 #10): earlier batches' uses (B4's state) plus this plan's.
            return p not in used and packs[p]["naics"] not in taken and earlier.get(p, 0) + dealt[p] < PACK_USES_MAX

        for s in (x for x in slots if x.lesson == lesson and x.kind == "B"):
            pick = next((p for p in deck if free(p)), None)
            if pick is None:
                # Least-used packs first, so uses stay level across the whole regeneration.
                fresh = sorted(packs)
                rng.shuffle(fresh)
                fresh.sort(key=lambda p: earlier.get(p, 0) + dealt[p])
                deck += fresh
                pick = next((p for p in deck if free(p)), None)
            if pick is None:
                raise SolveFirstError(f"REFUSED: {lesson} has no free pack left")
            deck.remove(pick)
            dealt[pick] += 1
            used.add(pick)
            s.pack = pick
            s.b_seed = random.Random(f"bseed:{seed}:{s.id}").randrange(1, 10**9)
            s.b_path = "solve_first" if s.structure in SOLVE_FIRST else "freehand"
            code = packs[pick]["naics"]
            if table is not None and code not in titles:
                raise SolveFirstError(f"REFUSED: pack {pick}'s NAICS code {code} is not a subsector in the domain table")
            s.domain = {"code": code, "title": titles.get(code, ""), "pack": pick}
    check_pack_uses(slots, earlier)


def skeleton_of(locked: Locked) -> str:
    """The locked model's fact skeleton, as puzzle-check computes it (renamed, order-free), hashed."""

    from role_set.puzzle_check import skeleton

    return hashlib.sha256(skeleton(locked.model).encode("utf-8")).hexdigest()[:16]


def lock_skeletons(slots: list, packs: dict[str, dict], *, prior: dict | None = None) -> dict:
    """Build every solve-first slot's puzzle at plan time and re-seed a slot whose skeleton another slot of its
    structure already has. Records `b_skeleton` and `b_reseeds` on each slot; refuses the plan when a structure falls
    below the floor. `prior` is {structure: {skeleton: uses}} from earlier batches (B4's cumulative state). Returns the
    batch's counts in the same shape."""

    from collections import Counter

    seen: dict[str, Counter] = {k: Counter(v) for k, v in (prior or {}).items()}
    for s in slots:
        if s.kind != "B" or s.b_path != "solve_first":
            continue
        counts = seen.setdefault(s.structure, Counter())
        base, reseeds = s.b_seed, 0
        while True:
            sk = skeleton_of(generate(s.structure, packs[s.pack], s.b_seed))
            if counts[sk] == 0 or reseeds >= RESEED_MAX:
                break
            reseeds += 1
            s.b_seed = random.Random(f"reseed:{base}:{reseeds}").randrange(1, 10**9)
        counts[sk] += 1
        s.b_skeleton, s.b_reseeds = sk, reseeds
    for structure, counts in seen.items():
        total = sum(counts.values())
        if not total:
            continue
        worst = max(counts.values())
        if len(counts) < SKELETON_DISTINCT_SHARE * total or worst > SKELETON_USES_MAX:
            raise SolveFirstError(f"REFUSED: {structure} has {len(counts)} distinct skeletons in {total} puzzles (floor "
                                  f"{SKELETON_DISTINCT_SHARE:.0%}) and its most used is used {worst} times (at most {SKELETON_USES_MAX})")
    return {k: dict(v) for k, v in seen.items()}


def check_pack_uses(slots: list, prior: dict | None = None) -> None:
    from collections import Counter

    uses = Counter(s.pack for s in slots if getattr(s, "pack", None))
    over = {p: n + (prior or {}).get(p, 0) for p, n in uses.items() if n + (prior or {}).get(p, 0) > PACK_USES_MAX}
    if over:
        raise SolveFirstError(f"REFUSED: packs used more than {PACK_USES_MAX} times across the regeneration (earlier "
                              f"batches plus this one): {over}")
    for lesson in {s.lesson for s in slots}:
        mine = [s.pack for s in slots if s.lesson == lesson and getattr(s, "pack", None)]
        if len(set(mine)) != len(mine):
            raise SolveFirstError(f"REFUSED: {lesson} uses a pack twice")


# ---- the writer's brief and the late-answer gate -----------------------------------------------------------------
def brief(locked: Locked) -> str:
    lines = "\n".join(f"{n}. {g}" for n, g in locked.glosses.items())
    when = (
        "it may say the answer is settled at the settling line once it has cited every earlier line, then works through any remaining lines"
        if locked.structure == "stopping_problem"
        else "it reaches the answer only at its end"
    )
    return (
        "## The locked puzzle (made and solved in code; you render it, you never change it)\n"
        f"The model, as data:\n{json.dumps(locked.model, ensure_ascii=False)}\n\n"
        f"What each numbered line must state, in your own words (keep the names and values exactly; a 'not' line keeps "
        f"its 'not', and only a line that says earlier/later may use before, after, earlier or later):\n{lines}\n"
        f"Then a last line that begins 'Question:' and asks: {locked.question}\n\n"
        f"The answer key is fixed: {locked.key}\n"
        "Return only 'puzzle', 'thinking' and 'reply'. The thinking works through the lines in order and cites each "
        f"fact line as 'line N' inside a sentence, for example '(line 3)', never opening every step the same way; {when}. "
        "The reply ends with 'Answer: ' and the key."
    )


def _steps(text: str) -> list[str]:
    return [x for x in re.split(r"\n+|(?<=[.!?])\s+", text) if x.strip()]


# Clause boundaries inside a step: semicolons and colons, and a comma before a connective that starts a new claim.
CLAUSE_SPLIT = re.compile(r"[;:]|,\s*(?=(?:so|therefore|thus|hence|which means|meaning|but|yet)\b)|\b(?:so|therefore|thus|hence)\b",
                          re.IGNORECASE)


def _named(name: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE) is not None


def _cited_lines(text: str) -> set[int]:
    return {int(x) for m in LINE_REF.finditer(text) for x in re.findall(r"\d+", m.group(0))}


def _rivals(locked: Locked) -> list[str]:
    """The answers the thinking could still be weighing: the asked attribute's other values, or for an entity_with
    question the other entities."""

    asks = locked.model.get("asks") or {}
    keyed = {k.lower() for k in locked.keyed}
    if "entity_with" in asks:
        pool = locked.model["entities"]
    elif "attribute" in asks:
        pool = locked.model["attributes"][asks["attribute"]]["values"]
    else:
        return []
    return [str(x) for x in pool if str(x).lower() not in keyed]


def answer_early(locked: Locked, thinking: str) -> str | None:
    """Why the thinking reaches the answer too early, or None (R&D's ANSWER_EARLY, 2026-10-16).

    The keyed answer, paired with what is asked (the asked entity, or the claim's verdict), may not appear in a
    positive step before the thinking's final third (or for a stopping problem, before every earlier needed line has
    been cited); and every fact line of the locked model is cited as 'line N'."""

    text = thinking or ""
    if not text.strip():
        return "the thinking is empty"
    raw_facts = locked.model["facts"]
    if locked.structure == "stopping_problem":
        stop_line = int(locked.key)
        prior_needed = {f["line"] for f in raw_facts if f["line"] < stop_line}
        rival_lines = {f["line"] for f in raw_facts if f["line"] != stop_line}
        seen_lines: set[int] = set()
        for step in _steps(text):
            for clause in CLAUSE_SPLIT.split(step):
                clause_cited = _cited_lines(clause)
                seen_lines |= clause_cited
                low = clause.lower()
                if not low.strip() or NEGATION.search(clause):
                    continue
                if locked.subject.lower() in low and stop_line in clause_cited and not (clause_cited & rival_lines):
                    if not (prior_needed <= seen_lines):
                        return f"the thinking pairs {locked.subject} with line {stop_line} before citing every earlier needed line"
    else:
        cut = len(text) * LATE_SHARE
        rivals = _rivals(locked)
        pos = 0
        for step in _steps(text):
            at = text.find(step, pos)
            pos = at + len(step)
            if at >= cut:
                continue
            # Clause by clause (Argon §9.4 #12): a negation rules out only its own clause, so "X is not A, so X is B"
            # still states B.
            for clause in CLAUSE_SPLIT.split(step):
                low = clause.lower()
                if not low.strip() or NEGATION.search(clause):
                    continue
                if locked.subject is None:  # a claim: the verdict word itself
                    if re.search(rf"\b{re.escape(locked.key)}\b", low):
                        return f"the thinking gives the verdict '{locked.key}' before its final third"
                elif locked.subject.lower() in low and all(_named(k, clause) for k in locked.keyed) \
                        and not any(_named(r, clause) for r in rivals):
                    # A clause that still names a rival answer lists candidates ("A or B", "whether X or Y"); it
                    # doesn't state the answer.
                    return f"the thinking pairs {locked.subject} with the answer before its final third"
    # Citations in any form and anywhere in a sentence ("(line 3)", "line 3 says", "lines 2 and 4"), so no one form
    # becomes the new formula (R&D, 2026-10-16). Every cited number is one of the puzzle's lines, and the distinct fact
    # lines cited reach the pruned set's size: every fact is needed, so every fact is used.
    cited = _cited_lines(text)
    facts = {f["line"] for f in raw_facts}
    lines = facts | set(locked.model.get("flavour") or [])
    stray = sorted(cited - lines)
    if stray:
        return "the thinking cites line" + ("s " if len(stray) > 1 else " ") + ", ".join(map(str, stray)) + ", which the puzzle doesn't have"
    missing = sorted(facts - cited)
    if missing:
        return (f"the thinking cites {len(facts & cited)} of the {len(facts)} fact lines; it never cites line"
                + ("s " if len(missing) > 1 else " ") + ", ".join(map(str, missing)))
    return None


def render_for_mock(locked: Locked) -> dict:
    """The mock writer's fields: the glosses as numbered lines, thinking that cites every line and reaches the key at
    its end, and the reply. Mock text only; never data."""

    puzzle = "\n".join(f"{n}. {g[0].upper() + g[1:]}." for n, g in locked.glosses.items()) + f"\nQuestion: {locked.question}"
    facts = [f["line"] for f in locked.model["facts"]]
    # The gates refuse a sentence template shared by over a quarter of a lesson's B items, so the mock varies its
    # words by the puzzle's own hash.
    k = int(locked.sha256[:8], 16)
    verbs = ("narrows it", "trims the options", "rules some out", "cuts the field", "removes a choice", "shrinks the set",
             "tightens it", "settles a piece")
    ends = ("Nothing else is open now", "No other option survives", "Every other choice is gone", "The rest is ruled out",
            "Only one option stands", "The field is down to one", "No alternative remains", "The other options fall away")
    closes = ("closes it", "finishes the job", "is the last piece", "decides it", "seals it", "leaves no doubt",
              "completes the picture", "is what settles it")
    # Citations sit inside the sentence, never as its opener: the gates count a lesson's shared sentence openers.
    steps = [f"{i}. {locked.glosses[n][0].upper() + locked.glosses[n][1:]}, which {verbs[(k + i) % len(verbs)]} (line {n})."
             for i, n in enumerate(facts, 1)]
    used = [e for e in locked.model["entities"] if any(e in rule_names(f["rule"]) for f in locked.model["facts"])]
    e0, e1 = (used + locked.model["entities"])[:2]
    steps.append(f"{len(facts) + 1}. With {e0} and {e1} in view, {ends[k % len(ends)].lower()}.")
    last = locked.glosses[facts[-1]]
    reply = f"{last[0].upper() + last[1:]}, and that {closes[k % len(closes)]}.\nAnswer: {locked.key}"
    return {"puzzle": puzzle, "thinking": "\n".join(steps), "reply": reply}
