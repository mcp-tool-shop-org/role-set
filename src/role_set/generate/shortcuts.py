"""The shortcut registry: named lazy readings, kept as data, that the generator proves and puzzle-check re-runs (design:
"one pinned shortcut code for generation and review"). R&D reviews each reading before it is registered here.

A reading rewrites the model the way a reader skipping the trait's step would read it, and the solver answers the rewrite.
The divergence rule depends on the answer type (R&D, 2026-10-11):
- claim-keyed: the lazy verdict differs from the key; an ambiguous lazy model is cannot_tell, which counts as different;
- set-keyed (an underdetermined question): the lazy set of possible answers differs from the key's;
- single-value: the lazy answer is unique and different, or it is ambiguous and its possible answers exclude the key.
A lazy model no assignment fits proves nothing: a reader who meets a contradiction knows the reading is wrong, and the
design asks each shortcut for a definite wrong answer (HANS, McCoy et al. 2019).
"""

from __future__ import annotations

import copy

from role_set.puzzle import PuzzleError, Table, parse_model, render, render_set

# Each entry: the lesson it serves, the rewrite it applies, and its parameters. Registered after R&D's review of
# 2026-10-11 ("T10: = -> >=, = -> <=, >= -> =, <= -> ="). T11 and T12 readings join with their recipes.
REGISTRY: dict[str, dict] = {
    "T10.exact_read_as_floor": {"lesson": "T10", "rewrite": "count_op", "from": "=", "to": ">=",
                                "reads": "'exactly k' read as 'at least k'"},
    "T10.exact_read_as_ceiling": {"lesson": "T10", "rewrite": "count_op", "from": "=", "to": "<=",
                                  "reads": "'exactly k' read as 'at most k'"},
    "T10.floor_read_as_exact": {"lesson": "T10", "rewrite": "count_op", "from": ">=", "to": "=",
                                "reads": "'at least k' read as 'exactly k'"},
    "T10.ceiling_read_as_exact": {"lesson": "T10", "rewrite": "count_op", "from": "<=", "to": "=",
                                  "reads": "'at most k' read as 'exactly k'"},
    # T12, approved by R&D on 2026-10-11 ("the latest or only edition; the month name without the span's ends; the
    # boundary day given to the wrong edition"; the highest number). Answer readings: they name the edition a reader
    # skipping the date match would name, from the same clues.
    "T12.newest": {"lesson": "T12", "rewrite": "answer", "reads": "the edition in force last, whatever the date"},
    "T12.highest_number": {"lesson": "T12", "rewrite": "answer", "reads": "the edition with the highest number"},
    "T12.boundary_next": {"lesson": "T12", "rewrite": "answer",
                          "reads": "a span's closing day given to the span that follows (its opening day to the one before)"},
    "T12.month_alone": {"lesson": "T12", "rewrite": "answer",
                        "reads": "any span whose start or end names the asked month, without its day"},
}


def _count_op(rule, frm, to):
    if rule[0] == "count" and rule[3] == frm:
        return [*rule[:3], to, rule[4]], True
    if rule[0] in ("and", "or", "not", "if"):
        parts, hit = [], False
        for r in rule[1:]:
            new, h = _count_op(r, frm, to)
            parts.append(new)
            hit = hit or h
        return [rule[0], *parts], hit
    return rule, False


def applies(name: str, model: dict) -> bool:
    """Whether the reading changes anything in this model (a reading with nothing to rewrite proves nothing)."""

    return rewrite(name, model) is not None


def rewrite(name: str, model: dict) -> dict | None:
    spec = REGISTRY[name]
    if spec["rewrite"] == "count_op":
        out = copy.deepcopy(model)
        hit = False
        for f in out["facts"]:
            f["rule"], h = _count_op(f["rule"], spec["from"], spec["to"])
            hit = hit or h
        return out if hit else None
    if spec["rewrite"] == "answer":
        return None  # an answer reading names an edition directly; see answer_reading
    raise KeyError(f"unknown rewrite {spec['rewrite']!r}")


def answer(model: dict, structure: str) -> dict:
    """The solver's answer to a model without the review-time refusals (a lazy model may well be a bad puzzle): the
    verdict for a claim, else the set of possible answers."""

    m = parse_model(model)
    t = Table(m)
    if m.claim is not None:
        v = t.verdict(t.all)
        return {"kind": "claim", "value": v or "no assignment fits"}
    projs = t.projections(t.all)
    if not projs:
        return {"kind": "none", "value": "no assignment fits"}
    if structure == "underdetermined_with_explicit_set":
        return {"kind": "set", "value": render_set(m, projs)}
    return {"kind": "values", "value": sorted(render(m, p) for p in projs)}


def _solutions(m) -> list[dict]:
    import itertools

    from role_set.puzzle import holds

    spaces = []
    for a in m.attrs.values():
        n = len(m.entities)
        spaces.append(list(itertools.permutations(a["values"])) if a["distinct"] else list(itertools.product(a["values"], repeat=n)))
    names = list(m.attrs)
    sols = (dict(zip(names, combo)) for combo in itertools.product(*spaces))
    return [s for s in sols if all(holds(m, s, r) for _, r in m.facts)]


def answer_reading(name: str, model: dict) -> dict | None:
    """A T12 reading's answer: the edition(s) a reader skipping the date match names, over every placing the clues allow.
    None when the reading has nothing to act on (no numbers, no boundary day, no other span naming the month)."""

    import re

    m = parse_model(model)
    if m.span is None or "contains" not in (m.asks or {}):
        return None
    sp = m.span
    attr = sp.attribute
    order = m.attrs[attr]["values"]
    day = sp.day(m.asks["contains"])
    sols = _solutions(m)
    if not sols:
        return None

    def holders_of(span_values: set) -> list[str]:
        return sorted({m.entities[i] for s in sols for i, v in enumerate(s[attr]) if v in span_values})

    if name == "T12.newest":
        return {"kind": "values", "value": holders_of({order[-1]})}
    if name == "T12.highest_number":
        nums = [re.search(r"\d+", e) for e in m.entities]
        if not all(nums):
            return None
        top = max(range(len(nums)), key=lambda i: int(nums[i].group(0)))
        return {"kind": "values", "value": [m.entities[top]]}
    if name == "T12.boundary_next":
        for i, v in enumerate(order):
            lo, hi = sp.bounds[v]
            if day == hi and i + 1 < len(order):
                return {"kind": "values", "value": holders_of({order[i + 1]})}
            if day == lo and i > 0:
                return {"kind": "values", "value": holders_of({order[i - 1]})}
        return None
    if name == "T12.month_alone":
        month = sp.months[day]
        named = {v for v in order if month in (sp.months[sp.bounds[v][0]], sp.months[sp.bounds[v][1]])}
        containing = {v for v in order if sp.bounds[v][0] <= day <= sp.bounds[v][1]}
        if not named or named <= containing:
            return None
        return {"kind": "values", "value": holders_of(named)}
    raise KeyError(f"unknown answer reading {name!r}")


def differs(key: dict, lazy: dict) -> bool:
    if lazy["kind"] == "none" or lazy["value"] == "no assignment fits":
        return False
    if key["kind"] == "claim":
        return lazy["value"] != key["value"]
    if key["kind"] == "set":
        return lazy["value"] != key["value"]
    (k,) = key["value"]
    return k not in lazy["value"] if len(lazy["value"]) > 1 else lazy["value"] != [k]


def prove(model: dict, structure: str, names: list[str]) -> dict:
    """Run each named reading that applies; report its answer and whether it differs from the key."""

    key = answer(model, structure)
    out = {}
    for name in names:
        if REGISTRY[name]["rewrite"] == "answer":
            lazy = answer_reading(name, model)
            if lazy is not None:
                out[name] = {"answer": lazy["value"], "differs": differs(key, lazy)}
            continue
        lazy_model = rewrite(name, model)
        if lazy_model is None:
            continue
        try:
            lazy = answer(lazy_model, structure)
        except PuzzleError as e:
            lazy = {"kind": "none", "value": str(e)}
        out[name] = {"answer": lazy["value"], "differs": differs(key, lazy)}
    return out
