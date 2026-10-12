"""Template wording for a built model: a setting line, one line per fact, and the Question line. The wording must
round-trip through `check_text` (each line names exactly what its rule names, negative lines read negative, direction
words match the rule), so a template that breaks a tie is refused at render time, not at review.

The clue-form API: TEMPLATES maps each rule form to sentences with named slots. Slots: {x} and {y} (entities), {a}
(attribute), {v}, {v1}, {v2} (values), {k} (a number word), {plural} (the pack's plural noun), and for `if`, {p} and {q}
(clauses rendered by the clause templates). Seat C's packs add sentences per form (at least 3 each before a freeze); the
set here is the slice-1 baseline.
"""

from __future__ import annotations

import random

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six")

# Clauses: the body of a sentence, reused inside `if`.
CLAUSES: dict[str, list[str]] = {
    "is": ["{x} is {v}"],
    "is_not": ["{x} is not {v}"],
}
TEMPLATES: dict[str, list[str]] = {
    "is": ["{X} is {v}."],
    "is_not": ["{X} is not {v}.", "{X} isn't {v}."],
    "in": ["{X} is {v1} or {v2}."],
    "not_in": ["{X} is neither {v1} nor {v2}."],
    "before": ["{X} comes before {y}."],
    "after": ["{X} comes after {y}."],
    "adjacent": ["{X} is next to {y}."],
    "not_adjacent": ["{X} is not next to {y}."],
    "offset1": ["{Y} is next on from {x}."],
    "offset2": ["{Y} is two places on from {x}."],
    "count=": ["Exactly {k} of the {plural} {be} {v}."],
    "count>=": ["At least {k} of the {plural} {be} {v}."],
    "count<=": ["At most {k} of the {plural} {be} {v}."],
    "if": ["If {p}, then {q}."],
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _clause(rule: list, rng: random.Random) -> str:
    return rng.choice(CLAUSES[rule[0]]).format(x=rule[1], v=rule[3])


def sentence(rule: list, plural: str, rng: random.Random) -> str:
    form = rule[0]
    if form in ("is", "is_not"):
        return rng.choice(TEMPLATES[form]).format(X=_cap(rule[1]), v=rule[3])
    if form in ("in", "not_in"):
        v1, v2 = rule[3]
        return rng.choice(TEMPLATES[form]).format(X=_cap(rule[1]), v1=v1, v2=v2)
    if form in ("before", "after", "adjacent", "not_adjacent"):
        return rng.choice(TEMPLATES[form]).format(X=_cap(rule[1]), y=rule[2])
    if form == "offset":
        return rng.choice(TEMPLATES[f"offset{rule[4]}"]).format(Y=_cap(rule[2]), x=rule[1])
    if form == "count":
        return rng.choice(TEMPLATES[f"count{rule[3]}"]).format(k=NUMBER_WORDS[rule[4]], plural=plural, v=rule[2],
                                                               be="is" if rule[4] == 1 else "are")
    if form == "if":
        return rng.choice(TEMPLATES["if"]).format(p=_clause(rule[1], rng), q=_clause(rule[2], rng))
    raise KeyError(f"no template for {form!r}")


def _list(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " and " + names[-1]


def _or(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " or " + names[-1]


def setting_line(model: dict, plural: str) -> str:
    """Line 1: names every entity and every value, states each attribute's domain, and states the order of any ordered
    attribute as a range ("places 1st to 4th"), as check_text requires."""

    ents = model["entities"]
    parts = [f"{_cap(_list(ents))} are the {plural}."]
    for a, spec in model["attributes"].items():
        vals = spec["values"]
        if spec.get("ordered"):
            parts.append(f"Each stands in its own {a}, from {vals[0]} to {vals[-1]}: {_list(vals)}.")
        elif spec.get("distinct", True):
            parts.append(f"Each has its own {a}, one of {_list(vals)}.")
        else:
            parts.append(f"Each is {_or(vals)}.")
    return " ".join(parts)


def question_line(model: dict) -> str:
    if model.get("claim") is not None:
        c = model["claim"]
        return f"Question: Claim: {sentence(c, '', random.Random(0))}"
    asks = model["asks"]
    if "of" in asks:
        vals = model["attributes"][asks["attribute"]]["values"]
        return f"Question: Is {asks['of']} {_or(vals)}?"
    if "entity_with" in asks:
        (a, v), = asks["entity_with"].items()
        return f"Question: Which of them could be {v}?"
    raise KeyError(f"no question template for {asks}")


def render_puzzle(model: dict, plural: str, rng: random.Random) -> str:
    lines = [setting_line(model, plural)] + [sentence(f["rule"], plural, rng) for f in model["facts"]]
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(lines)) + "\n" + question_line(model)
