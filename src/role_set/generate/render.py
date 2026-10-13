"""Template wording for a built model: a setting line, one line per fact, and the Question line. The wording must
round-trip through `check_text` (each line names exactly what its rule names, negative lines read negative, direction
words match the rule), so a template that breaks a tie is refused at render time, not at review.

The clue-form API: TEMPLATES maps each rule form to sentences with named slots. Slots: {X}/{x} and {Y}/{y} (entities,
capitalised or not), {v}, {v1}, {v2} (values), {k} (a number word), {be} ("is" or "are"), {plural} (the pack's plural
noun), and for `if`, {p} and {q} (clauses from CLAUSES). Each form has at least three sentences. A sentence uses only
the slots `sentence` passes for that form.
"""

from __future__ import annotations

import random

NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six")

# Clauses: the body of a sentence, reused inside `if`. {x} is the entity as written, so a clause can sit mid-sentence.
CLAUSES: dict[str, list[str]] = {
    "is": ["{x} is {v}", "we find {x} {v}", "{x} counts as {v}"],
    "is_not": ["{x} is not {v}", "{x} isn't {v}", "we never find {x} {v}"],
}
# {X} and {Y} arrive already capitalised, so those sentences start with the entity. "is" is index 1 because a claim
# question renders through Random(0), and that draw picks index 1.
TEMPLATES: dict[str, list[str]] = {
    "is": ["{X} counts as {v}.", "{X} is {v}.", "{X} shows as {v}."],
    "is_not": ["{X} is not {v}.", "{X} isn't {v}.", "{X} never shows as {v}."],
    "in": ["{X} is {v1} or {v2}.", "{X} is either {v1} or {v2}.", "{X} matches {v1} or {v2}."],
    "not_in": ["{X} is neither {v1} nor {v2}.", "{X} matches neither {v1} nor {v2}.", "{X} is not {v1} and not {v2}."],
    "before": ["{X} comes before {y}.", "{X} is earlier than {y}.", "{X} stands ahead of {y}."],
    "after": ["{X} comes after {y}.", "{X} is later than {y}.", "{X} stands behind {y}."],
    "adjacent": ["{X} is next to {y}.", "{X} sits beside {y}.", "{X} touches {y}."],
    "not_adjacent": ["{X} is not next to {y}.", "{X} isn't beside {y}.", "{X} never touches {y}."],
    "offset1": ["{Y} is next on from {x}.", "{Y} is one place on from {x}.", "{Y} sits one place past {x}."],
    "offset2": ["{Y} is two places on from {x}.", "{Y} sits two places past {x}.", "{Y} comes two places on from {x}."],
    "count=": ["Exactly {k} of the {plural} {be} {v}.", "Of the {plural}, {k} {be} {v}.",
               "The tally of {v} {plural} is {k}."],
    "count>=": ["At least {k} of the {plural} {be} {v}.", "Of the {plural}, {k} or more {be} {v}.",
                "The {plural} include {k} or more that {be} {v}."],
    "count<=": ["At most {k} of the {plural} {be} {v}.", "Of the {plural}, {k} or fewer {be} {v}.",
                "Up to {k} of the {plural} {be} {v}."],
    "if": ["If {p}, then {q}.", "When {p}, {q}.", "Then {q}, if {p}."],
}


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _clause(rule: list, rng: random.Random) -> str:
    return rng.choice(CLAUSES[rule[0]]).format(x=rule[1], v=rule[3])


def sentence(rule: list, plural: str, rng: random.Random, templates: dict | None = None) -> str:
    TEMPLATES = {**globals()["TEMPLATES"], **(templates or {})}  # noqa: N806 - a recipe may override forms
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
    span_attr = (model.get("span") or {}).get("attribute")
    for a, spec in model["attributes"].items():
        vals = spec["values"]
        if a == span_attr:
            # Spans: each named with both ends, listed in calendar order.
            parts.append(f"They were in force one after another, one to each span, in this order: {_list(vals)}.")
        elif spec.get("ordered"):
            parts.append(f"Each stands in its own {a}, from {vals[0]} to {vals[-1]}: {_list(vals)}.")
        elif spec.get("distinct", True):
            parts.append(f"Each has its own {a}, one of {_list(vals)}.")
        else:
            parts.append(f"Each is {_or(vals)}.")
    return " ".join(parts)


def question_line(model: dict, singular: str = "one", templates: dict | None = None) -> str:
    if model.get("claim") is not None:
        c = model["claim"]
        if c[0] == "contains":
            return f"Question: Claim: {_cap(c[1])} was in force on {c[2]}."
        return f"Question: Claim: {sentence(c, '', random.Random(0), templates)}"
    asks = model["asks"]
    if "contains" in asks:
        return f"Question: Which {singular} was in force on {asks['contains']}?"
    if "of" in asks:
        vals = model["attributes"][asks["attribute"]]["values"]
        return f"Question: Is {asks['of']} {_or(vals)}?"
    if "entity_with" in asks:
        (a, v), = asks["entity_with"].items()
        return f"Question: Which of them could be {v}?"
    raise KeyError(f"no question template for {asks}")


def render_puzzle(model: dict, plural: str, rng: random.Random, templates: dict | None = None, singular: str = "one") -> str:
    lines = [setting_line(model, plural)] + [sentence(f["rule"], plural, rng, templates) for f in model["facts"]]
    return "\n".join(f"{i + 1}. {x}" for i, x in enumerate(lines)) + "\n" + question_line(model, singular, templates)
