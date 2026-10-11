"""Kind B's mechanical key: the writer's puzzle as data, solved exhaustively, tied line by line to the printed text.

Pilot 2's B traces each had a wrong forcing step, and three of its five puzzles had more than one solution (R&D
brute-forced them). A puzzle is now written twice: as numbered fact lines a reader solves, and as a `model` the code
solves: entities, attributes with their values, and one rule per fact line in a closed set of forms. The code
refuses the draft unless the model is tied to the text (every line has a rule or is flavour, every rule names only
what its line names, polarity and direction agree), every fact is needed, and the solver gets the key: exactly one
answer to what is asked (a computed set for an underdetermined puzzle, a computed stopping fact, a computed verdict
for a claim). Only the solver decides; the writer is never asked whether its puzzle is unique. Design and five
additions: R&D, 2026-10-11 (PR H).

Known limits: the direction check reads argument order from where names first appear, so a fronted clause ("Before
Ana, Ben arrived") is refused although it is right; and the thinking isn't checked against the model (R&D reads the
B traces in the key-check until a step-by-step check lands).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from role_set.assemble import UNIQUENESS_ASKED, closed_answer

SEARCH_CAP = 200_000  # assignments; a larger puzzle is refused as too large to check
MAX_ENTITIES = 6
MAX_ATTRIBUTES = 3
MAX_FLAVOUR = 2
POSITIVE = {"is", "in", "same", "before", "after", "adjacent", "offset", "count", "and", "or", "if"}
NEGATIVE = {"is_not", "not_in", "different", "not_adjacent", "not"}
ARITY = {"is": 3, "is_not": 3, "in": 3, "not_in": 3, "same": 2, "different": 2, "before": 3, "after": 3, "adjacent": 3,
         "not_adjacent": 3, "offset": 4, "count": 4, "not": 1, "if": 2}
ORDERED_FORMS = {"before", "after", "adjacent", "not_adjacent", "offset"}
COUNT_OPS = {"=": lambda a, b: a == b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}
NEGATION = re.compile(r"\b(not|never|no|neither|nor|none|nobody|\w+n't|cannot)\b", re.IGNORECASE)
BEFORE_WORDS = re.compile(r"\b(before|earlier|left of|ahead of)\b", re.IGNORECASE)
AFTER_WORDS = re.compile(r"\b(after|later|right of|behind)\b", re.IGNORECASE)
SIDE_WORDS = re.compile(r"\b(left|right)\b", re.IGNORECASE)
END_WORDS = re.compile(r"\b(first|last)\b", re.IGNORECASE)
# Values are listed left to right, so "left of" is a lower index (before). "Right to left" would encode every
# left/right line backwards, so it isn't accepted (R&D on #29).
SIDE_STATED = re.compile(r"\bleft to right\b", re.IGNORECASE)
SIDE_REVERSED = re.compile(r"\bright to left\b", re.IGNORECASE)
NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(\S.*)$")
VERDICT_STRUCTURES = ("tree_with_pruning", "falsifier_first", "when_to_stop")
STOPPING_STRUCTURES = ("stopping_problem", "when_to_stop")


class PuzzleError(ValueError):
    """The model doesn't hold, doesn't match the text, or doesn't get the key. The message is the suggestion."""


@dataclass
class Model:
    entities: list[str]
    attrs: dict[str, dict]
    facts: list[tuple[int, list]]
    flavour: list[int]
    asks: dict | None
    claim: list | None


@dataclass
class Checked:
    answer: str  # the computed key, rendered in closed form
    lookup: bool  # the asked answer follows from `is` facts alone
    solutions: int
    stop_at: int | None = None
    notes: list[str] = field(default_factory=list)


def _named(name: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.IGNORECASE) is not None


# ---- shape ---------------------------------------------------------------------------------------------------------
def parse_model(raw) -> Model:
    if not isinstance(raw, dict):
        raise PuzzleError("'model' must be an object with entities, attributes, facts, flavour and asks (or claim)")
    ents = raw.get("entities")
    if not isinstance(ents, list) or not 2 <= len(ents) <= MAX_ENTITIES or not all(isinstance(e, str) and e.strip() for e in ents) \
            or len({e.lower() for e in ents}) != len(ents):
        raise PuzzleError(f"model.entities must be 2 to {MAX_ENTITIES} distinct names")
    attrs = raw.get("attributes")
    if not isinstance(attrs, dict) or not 1 <= len(attrs) <= MAX_ATTRIBUTES:
        raise PuzzleError(f"model.attributes must hold 1 to {MAX_ATTRIBUTES} attributes")
    norm: dict[str, dict] = {}
    for name, spec in attrs.items():
        vals = spec.get("values") if isinstance(spec, dict) else None
        if not isinstance(vals, list) or len(vals) < 2 or not all(isinstance(v, str) and v.strip() for v in vals) \
                or len({v.lower() for v in vals}) != len(vals):
            raise PuzzleError(f"attribute {name!r} needs a list of 2 or more distinct values")
        distinct = spec.get("distinct", True) is not False
        if distinct and len(vals) != len(ents):
            raise PuzzleError(f"attribute {name!r} gives each entity a different value, so it needs exactly {len(ents)} values "
                              "(set \"distinct\": false to allow repeats)")
        norm[name] = {"values": vals, "ordered": spec.get("ordered") is True, "distinct": distinct}
    facts = raw.get("facts")
    if not isinstance(facts, list) or not facts:
        raise PuzzleError("model.facts must list one rule per fact line: {\"line\": n, \"rule\": [...]}")
    parsed = []
    for f in facts:
        if not isinstance(f, dict) or not isinstance(f.get("line"), int):
            raise PuzzleError("each fact is {\"line\": n, \"rule\": [form, ...]}")
        parsed.append((f["line"], f.get("rule")))
    flavour = raw.get("flavour") or []
    if not isinstance(flavour, list) or not all(isinstance(x, int) for x in flavour):
        raise PuzzleError("model.flavour is a list of line numbers")
    if len(flavour) > MAX_FLAVOUR:
        raise PuzzleError(f"at most {MAX_FLAVOUR} flavour lines")
    m = Model(ents, norm, parsed, flavour, raw.get("asks"), raw.get("claim"))
    for _, rule in m.facts:
        _check_rule(m, rule)
    if m.claim is not None:
        _check_rule(m, m.claim)
    elif not isinstance(m.asks, dict):
        raise PuzzleError("model.asks says what the question asks: {\"attribute\": a}, {\"attribute\": a, \"of\": entity} "
                          "or {\"entity_with\": {a: value}}")
    else:
        _check_asks(m, m.asks)
    return m


def _entity(m: Model, name) -> str:
    for e in m.entities:
        if isinstance(name, str) and e.lower() == name.lower():
            return e
    raise PuzzleError(f"{name!r} is not one of model.entities")


def _attr(m: Model, a, ordered: bool = False) -> dict:
    if a not in m.attrs:
        raise PuzzleError(f"{a!r} is not one of model.attributes")
    if ordered and not m.attrs[a]["ordered"]:
        raise PuzzleError(f"attribute {a!r} is not ordered; before, after, adjacent and offset need \"ordered\": true")
    return m.attrs[a]


def _value(m: Model, a, v) -> str:
    for x in _attr(m, a)["values"]:
        if isinstance(v, str) and x.lower() == v.lower():
            return x
    raise PuzzleError(f"{v!r} is not a value of attribute {a!r}")


def _check_ref(m: Model, ref) -> None:
    if isinstance(ref, dict):
        if len(ref) != 1:
            raise PuzzleError("a reference is an entity name or {attribute: value}")
        (a, v), = ref.items()
        if not _attr(m, a)["distinct"]:
            raise PuzzleError(f"{{{a}: {v}}} can't name one entity: attribute {a!r} repeats values")
        _value(m, a, v)
    else:
        _entity(m, ref)


def _check_rule(m: Model, rule) -> None:
    if not isinstance(rule, list) or not rule or not isinstance(rule[0], str):
        raise PuzzleError("a rule is a list: [form, arguments...]")
    form, args = rule[0], rule[1:]
    if form in ("and", "or"):
        if len(args) < 2:
            raise PuzzleError(f"{form} joins two or more rules")
        for r in args:
            _check_rule(m, r)
        return
    if form not in ARITY:
        raise PuzzleError(f"unknown rule form {form!r}; the forms are {', '.join(sorted(ARITY) + ['and', 'or'])}")
    if len(args) != ARITY[form]:
        raise PuzzleError(f"{form} takes {ARITY[form]} arguments")
    if form in ("not", "if"):
        for r in args:
            _check_rule(m, r)
    elif form in ("is", "is_not"):
        _check_ref(m, args[0])
        _value(m, args[1], args[2])
    elif form in ("in", "not_in"):
        _check_ref(m, args[0])
        if not isinstance(args[2], list) or not args[2]:
            raise PuzzleError(f"{form} takes a list of values")
        for v in args[2]:
            _value(m, args[1], v)
    elif form in ("same", "different"):
        _check_ref(m, args[0])
        _check_ref(m, args[1])
    elif form in ORDERED_FORMS:
        _check_ref(m, args[0])
        _check_ref(m, args[1])
        _attr(m, args[2], ordered=True)
        if form == "offset" and not isinstance(args[3], int):
            raise PuzzleError("offset takes a whole number of places")
    elif form == "count":
        _value(m, args[0], args[1])
        if args[2] not in COUNT_OPS or not isinstance(args[3], int):
            raise PuzzleError("count is [\"count\", attribute, value, \"=\" | \"<=\" | \">=\", n]")


def _check_asks(m: Model, asks: dict) -> None:
    if "entity_with" in asks:
        ew = asks["entity_with"]
        if not isinstance(ew, dict) or len(ew) != 1:
            raise PuzzleError("asks.entity_with is {attribute: value}")
        (a, v), = ew.items()
        _value(m, a, v)
    elif "attribute" in asks:
        _attr(m, asks["attribute"])
        if "of" in asks:
            _entity(m, asks["of"])
    else:
        raise PuzzleError("model.asks is {\"attribute\": a}, {\"attribute\": a, \"of\": entity} or {\"entity_with\": {a: value}}")


# ---- names a rule uses, polarity ---------------------------------------------------------------------------------
def _ref_names(ref) -> list[str]:
    if isinstance(ref, dict):
        return [str(v) for v in ref.values()]
    return [str(ref)]


def rule_names(rule) -> list[str]:
    form, args = rule[0], rule[1:]
    if form in ("and", "or", "not", "if"):
        return [n for r in args for n in rule_names(r)]
    if form in ("is", "is_not"):
        return _ref_names(args[0]) + [str(args[2])]
    if form in ("in", "not_in"):
        return _ref_names(args[0]) + [str(v) for v in args[2]]
    if form == "count":
        return [str(args[1])]
    return _ref_names(args[0]) + _ref_names(args[1])


def _forms(rule) -> list[str]:
    out = [rule[0]]
    if rule[0] in ("and", "or", "not", "if"):
        for r in rule[1:]:
            out += _forms(r)
    return out


def _ordered_rules(rule) -> list[list]:
    if rule[0] in ("and", "or", "not", "if"):
        return [x for r in rule[1:] for x in _ordered_rules(r)]
    return [rule] if rule[0] in ("before", "after") else []


# ---- the text ----------------------------------------------------------------------------------------------------
MAX_PROBLEMS = 8  # numbered in one refusal; the rest follow on the retry


def _fact_lines(puzzle: str, problems: list[str]) -> tuple[dict[int, str], str]:
    if "Question:" in puzzle:
        body, question = puzzle.rsplit("Question:", 1)
    else:
        problems.append("a puzzle ends with a line that begins 'Question:'")
        body, question = puzzle, ""
    lines: dict[int, str] = {}
    unnumbered = []
    for raw in body.splitlines():
        if not raw.strip():
            continue
        m = NUMBERED.match(raw)
        if not m:
            unnumbered.append(raw.strip()[:50])
            continue
        lines[int(m.group(1))] = m.group(2).strip()
    if unnumbered:
        problems.append(f"every line before the Question line is numbered (1., 2., ...), setting lines included: {unnumbered[0]!r}")
    if lines and sorted(lines) != list(range(1, len(lines) + 1)):
        problems.append("the puzzle's lines are numbered 1, 2, 3 ... in order, once each")
    return lines, question.strip()


def fact_lines(puzzle: str) -> tuple[dict[int, str], str]:
    """The numbered lines before the Question line, and the Question line. Every line before it is numbered."""

    problems: list[str] = []
    out = _fact_lines(puzzle, problems)
    if problems:
        raise PuzzleError("; ".join(problems))
    return out


def check_text(m: Model, puzzle: str) -> tuple[dict[int, str], str]:
    """Every way the model and the text disagree, in one refusal: the 8B rehearsal spent all three attempts fixing one
    reported problem per retry."""

    problems: list[str] = []
    lines, question = _fact_lines(puzzle, problems)
    cited: dict[int, list] = {}
    for n, rule in m.facts:
        if n not in lines:
            problems.append(f"a fact cites line {n}, and the puzzle has no line {n}")
        else:
            cited.setdefault(n, []).append(rule)
    for n in m.flavour:
        if n not in lines:
            problems.append(f"flavour cites line {n}, and the puzzle has no line {n}")
        elif n in cited:
            problems.append(f"line {n} is both a fact and flavour")
    for n in lines:
        if n not in cited and n not in m.flavour:
            problems.append(f"line {n} has no rule in the model; give it one, or list it in flavour if it states no fact")
    for name in m.entities + [v for a in m.attrs.values() for v in a["values"]]:
        if not _named(name, puzzle):
            problems.append(f"{name!r} is in the model but the puzzle never names it")
    for n in m.flavour:
        if n not in lines:
            continue
        # A setting line that lists an attribute's whole domain declares it; one that names some values reads as a fact.
        named = [v for a in m.attrs.values() if not all(_named(x, lines[n]) for x in a["values"])
                 for v in a["values"] if _named(v, lines[n])]
        if named and not _states_order(m, lines[n]):
            problems.append(f"flavour line {n} names {named[0]!r}, so it reads as a fact; give it a rule or drop the value")
    for n, rules in cited.items():
        line = lines[n]
        for rule in rules:
            missing = [x for x in rule_names(rule) if not _named(x, line)]
            if missing:
                problems.append(f"line {n}'s rule uses {missing[0]!r}, which line {n} doesn't name; a rule encodes only its own line")
        neg_line = NEGATION.search(line) is not None
        neg_rule = any(f in NEGATIVE for r in rules for f in _forms(r))
        if neg_line != neg_rule:
            problems.append(f"line {n} is {'negative' if neg_line else 'positive'} but its rule is "
                            f"{'negative' if neg_rule else 'positive'}; encode the line as it reads")
        try:
            _check_direction(n, line, rules)
        except PuzzleError as e:
            problems.append(str(e))
    try:
        _check_order_stated(m, lines)
    except PuzzleError as e:
        problems.append(str(e))
    for n in sorted(cited):
        m_ = MODEL_SYNTAX.search(lines[n])
        if m_:
            problems.append(f"line {n} reads as model syntax ('{m_.group(0)}'); write the fact in plain English, e.g. "
                            "'Exactly one switch is wired'")
    leaked = brief_leak(puzzle)
    if leaked:
        problems.append(f"the puzzle repeats the brief's instruction ({leaked!r}); a puzzle states facts and a question only")
    if m.claim is not None:
        missing = [x for x in rule_names(m.claim) if not _named(x, question)]
        if missing:
            problems.append(f"the claim uses {missing[0]!r}, which the Question line doesn't name")
    if problems:
        more = len(problems) - MAX_PROBLEMS
        listed = "; ".join(f"{i}) {x}" for i, x in enumerate(problems[:MAX_PROBLEMS], 1)) if len(problems) > 1 else problems[0]
        raise PuzzleError(listed + (f"; and {more} more" if more > 0 else ""))
    return lines, question


# The solver's dialect written into a fact line (fix round 2: "The count of wired is 1"). Only the forms that can be
# told mechanically; R&D reads the rest. A bare number as the predicate ("The choir line is 1.") is the other form.
# Word values ("is open", "is not container") can't be told from good English by rule, so the key-check reads those.
MODEL_SYNTAX = re.compile(r"\bthe count of\b|\bcount of \w+ is\b|\bis (?:not )?\d+(?:\.\d+)?\s*\.?\s*$", re.IGNORECASE)
# The brief's own words about laying out the working, copied into a puzzle (rehearsal 3: "Show your thinking as a
# constraint grid, and stop when it's settled").
BRIEF_SPANS = re.compile(r"show your thinking|constraint grid|stop when it'?s settled|tree glyphs|numbered elimination steps", re.IGNORECASE)


def brief_leak(puzzle: str) -> str | None:
    from role_set.critic import PATTERN_INSTRUCTION

    m = BRIEF_SPANS.search(puzzle) or PATTERN_INSTRUCTION.search(puzzle)
    return " ".join(m.group(0).split())[:60] if m else None


def _first(line: str, names: list[str]) -> int:
    pos = [mm.start() for x in names for mm in [re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", line, re.IGNORECASE)] if mm]
    return min(pos) if pos else -1


def _check_direction(n: int, line: str, rules: list) -> None:
    before, after = BEFORE_WORDS.search(line), AFTER_WORDS.search(line)
    if bool(before) == bool(after):
        return
    want = "before" if before else "after"
    ordered = [r for rule in rules for r in _ordered_rules(rule)]
    if not any(r[0] == want for r in ordered):
        raise PuzzleError(f"line {n} says '{(before or after).group(0)}', so its rule is {want}")
    for r in ordered:
        if r[0] == want:
            a, b = _first(line, _ref_names(r[1])), _first(line, _ref_names(r[2]))
            if a >= 0 and b >= 0 and a > b:
                raise PuzzleError(f"line {n}'s {want} rule has its arguments in the opposite order to the line")


def _states_order(m: Model, line: str) -> bool:
    if SIDE_STATED.search(line):
        return True
    # "seats 1 to 4", "from 10:00 through 13:00": the first and last positions, in that order, as a range.
    for a in m.attrs.values():
        lo, hi = (re.escape(a["values"][0]), re.escape(a["values"][-1]))
        if a["ordered"] and re.search(rf"(?<!\w){lo}(?!\w)\s*(?:to|through|-|–)\s*(?:the\s+)?{hi}(?!\w)", line, re.IGNORECASE):
            return True
    return False


def _check_order_stated(m: Model, lines: dict[int, str]) -> None:
    allt = " ".join(lines.values())
    if SIDE_REVERSED.search(allt):
        raise PuzzleError("list the positions left to right ('seats 1 to 4, left to right'); right to left reverses every left/right fact")
    if SIDE_WORDS.search(allt) and not SIDE_STATED.search(allt):
        raise PuzzleError("the puzzle says left or right, so it states the direction once ('seats 1 to 4, left to right')")
    ordinal = any(v.lower() in ("first", "last") for a in m.attrs.values() for v in a["values"])
    if END_WORDS.search(allt) and not ordinal and not any(_states_order(m, x) for x in lines.values()):
        raise PuzzleError("the puzzle says first or last, so one line states the order, naming its first and last positions")


# ---- the solver --------------------------------------------------------------------------------------------------
def _resolve(m: Model, sol: dict, ref) -> int:
    if isinstance(ref, dict):
        (a, v), = ref.items()
        return sol[a].index(_value(m, a, v))
    return m.entities.index(_entity(m, ref))


def holds(m: Model, sol: dict, rule) -> bool:
    form, args = rule[0], rule[1:]
    if form == "and":
        return all(holds(m, sol, r) for r in args)
    if form == "or":
        return any(holds(m, sol, r) for r in args)
    if form == "not":
        return not holds(m, sol, args[0])
    if form == "if":
        return (not holds(m, sol, args[0])) or holds(m, sol, args[1])
    if form in ("is", "is_not"):
        got = sol[args[1]][_resolve(m, sol, args[0])] == _value(m, args[1], args[2])
        return got if form == "is" else not got
    if form in ("in", "not_in"):
        got = sol[args[1]][_resolve(m, sol, args[0])] in {_value(m, args[1], v) for v in args[2]}
        return got if form == "in" else not got
    if form in ("same", "different"):
        got = _resolve(m, sol, args[0]) == _resolve(m, sol, args[1])
        return got if form == "same" else not got
    if form == "count":
        v = _value(m, args[0], args[1])
        return COUNT_OPS[args[2]](sum(1 for x in sol[args[0]] if x == v), args[3])
    order = m.attrs[args[2]]["values"]
    pa, pb = order.index(sol[args[2]][_resolve(m, sol, args[0])]), order.index(sol[args[2]][_resolve(m, sol, args[1])])
    if form == "before":
        return pa < pb
    if form == "after":
        return pa > pb
    if form == "adjacent":
        return abs(pa - pb) == 1
    if form == "not_adjacent":
        return abs(pa - pb) != 1
    return pb - pa == args[3]  # offset: b is k places after a


def _project(m: Model, sol: dict):
    asks = m.asks or {}
    if "entity_with" in asks:
        (a, v), = asks["entity_with"].items()
        v = _value(m, a, v)
        return tuple(e for e, x in zip(m.entities, sol[a]) if x == v)
    a = asks["attribute"]
    if "of" in asks:
        return sol[a][m.entities.index(_entity(m, asks["of"]))]
    return tuple(sol[a])


def render(m: Model, proj) -> str:
    asks = m.asks or {}
    if "entity_with" in asks:
        return ", ".join(proj)
    if "of" in asks:
        return proj
    return "; ".join(f"{e}={v}" for e, v in zip(m.entities, proj))


class Table:
    """Every assignment, once: a bitmask of the facts it satisfies, its projection and the claim's truth."""

    def __init__(self, m: Model):
        spaces = []
        size = 1
        for a in m.attrs.values():
            n = len(m.entities)
            opts = list(itertools.permutations(a["values"])) if a["distinct"] else list(itertools.product(a["values"], repeat=n))
            spaces.append(opts)
            size *= len(opts)
        if size > SEARCH_CAP:
            raise PuzzleError(f"the puzzle has {size:,} possible assignments; keep it under {SEARCH_CAP:,} (fewer entities or attributes)")
        names = list(m.attrs)
        self.rows = []
        for combo in itertools.product(*spaces):
            sol = dict(zip(names, combo))
            mask = 0
            for i, (_, rule) in enumerate(m.facts):
                if holds(m, sol, rule):
                    mask |= 1 << i
            claim = holds(m, sol, m.claim) if m.claim is not None else None
            proj = _project(m, sol) if m.claim is None else None
            self.rows.append((mask, proj, claim))
        self.all = (1 << len(m.facts)) - 1

    def matching(self, need: int):
        return [r for r in self.rows if r[0] & need == need]

    def projections(self, need: int) -> set:
        return {r[1] for r in self.matching(need)}

    def verdict(self, need: int) -> str | None:
        claims = {r[2] for r in self.matching(need)}
        if not claims:
            return None
        return "supported" if claims == {True} else "unsupported" if claims == {False} else "cannot_tell"


def _plain_direct(m: Model, rule) -> bool:
    """A fact that states a value outright: is, in with one value, same between an entity and a value reference, or
    is_not on an attribute with exactly two values (the other value, stated by negation; R&D on #95)."""

    if rule[0] == "is" or (rule[0] == "in" and len(rule[3]) == 1):
        return True
    if rule[0] == "is_not" and len(m.attrs[rule[2]]["values"]) == 2:
        return True
    return rule[0] == "same" and sum(isinstance(r, dict) for r in rule[1:]) == 1


def _direct(m: Model, rule) -> bool:
    """A direct fact, or an `if` whose antecedent is stated outright as another fact of the model, of any form: the
    answer given as an if plus its own antecedent (fix round 2's T13-B-07 with an `is`, T03-B-06 with a `before`)."""

    if _plain_direct(m, rule):
        return True
    if rule[0] == "if":
        return any(r == rule[1] for _, r in m.facts if r is not rule)
    return False


def _direct_mask(m: Model) -> int:
    """The direct facts, and with each `if` that is direct by its stated antecedent, that antecedent's own line: the
    two together state the answer."""

    out = 0
    for i, (_, r) in enumerate(m.facts):
        if _direct(m, r):
            out |= 1 << i
            if r[0] == "if":
                out |= _mask(j for j, (_, x) in enumerate(m.facts) if x == r[1] and j != i)
    return out


def _count_decoration(m: Model) -> str | None:
    """A count equal to the number of entities, or 0, that is the only fact on its value: it gives every entity the
    value (or none), so it decorates rather than constrains (fix round 2's T05-B-01 and T06-B-01)."""

    for n, rule in m.facts:
        if rule[0] != "count" or rule[3] != "=" or rule[4] not in (0, len(m.entities)):
            continue
        value = str(rule[2]).lower()
        if not any(value in (x.lower() for x in rule_names(r)) for k, r in m.facts if r is not rule):
            return f"line {n} sets the count of {rule[2]!r} to {rule[4]}, which settles every entity at once; that's a statement, not a clue"
    return None


def _mask(indices) -> int:
    out = 0
    for i in indices:
        out |= 1 << i
    return out


def solve(m: Model, structure: str | None, question: str) -> Checked:
    decoration = _count_decoration(m)
    if decoration:
        raise PuzzleError(decoration)
    t = Table(m)
    full = t.all
    if m.claim is not None:
        v = t.verdict(full)
        if v is None:
            raise PuzzleError("no assignment fits every fact; the facts contradict each other")
        for i, (n, _) in enumerate(m.facts):
            without = full & ~(1 << i)
            # A decided verdict: a fact that only narrows the solutions without moving it is decoration. cannot_tell
            # survives losing almost anything, so there a fact must also leave the solutions as they were (R&D on #29).
            if t.verdict(without) == v and (v != "cannot_tell" or len(t.matching(without)) == len(t.matching(full))):
                raise PuzzleError(f"line {n} changes nothing; drop it, or list it in flavour if it states no fact")
        direct = _direct_mask(m)
        settled = [direct] if direct else []
        settled += [1 << i for i in range(len(m.facts))]
        lookup = v != "cannot_tell" and any(t.verdict(x) == v for x in settled)
        return Checked(answer=v, lookup=lookup, solutions=len(t.matching(full)))

    projs = t.projections(full)
    if not projs:
        raise PuzzleError("no assignment fits every fact; the facts contradict each other")
    stop_at = None
    if structure in STOPPING_STRUCTURES:
        order = sorted(range(len(m.facts)), key=lambda i: m.facts[i][0])
        for k in range(1, len(order) + 1):
            if len(t.projections(_mask(order[:k]))) == 1:
                stop_at = m.facts[order[k - 1]][0]
                considered = order[:k]
                break
        if stop_at is None:
            raise PuzzleError("the facts never settle what is asked, so there is no fact to stop at")
    else:
        considered = list(range(len(m.facts)))
    if structure == "underdetermined_with_explicit_set":
        if "of" not in (m.asks or {}) and "entity_with" not in (m.asks or {}):
            raise PuzzleError("an underdetermined puzzle asks for one value (asks.of or asks.entity_with), so its possible answers form a list")
        if len(projs) < 2:
            raise PuzzleError("an underdetermined puzzle leaves at least two answers; these facts settle it")
    elif stop_at is None and len(projs) != 1:
        a, b = sorted(projs, key=str)[:2]
        raise PuzzleError(f"more than one answer fits every fact, for example {render(m, a)!r} and {render(m, b)!r}; "
                          "add a fact that separates them (one that doesn't state the answer outright)")
    need = _mask(considered)
    base = t.projections(need)
    for i in considered:
        if t.projections(need & ~(1 << i)) == base:
            raise PuzzleError(f"line {m.facts[i][0]} changes nothing about the answer; drop it, or list it in flavour if it states no fact")
    direct = _direct_mask(m)
    lookup = False
    if structure not in ("underdetermined_with_explicit_set",) + STOPPING_STRUCTURES:
        # A lookup: the direct facts alone settle the answer, or any single fact does by itself (R&D on #95).
        lookup = (direct != 0 and len(t.projections(direct)) == 1) or any(len(t.projections(1 << i)) == 1 for i in considered)
    if structure == "underdetermined_with_explicit_set":
        answer = ", ".join(sorted({render(m, p) for p in projs}))
    elif stop_at is not None:
        answer = str(stop_at)
    else:
        answer = render(m, next(iter(projs)))
    return Checked(answer=answer, lookup=lookup, solutions=len(t.matching(full)), stop_at=stop_at)


# ---- the whole check ---------------------------------------------------------------------------------------------
def _same(a: str, b: str) -> bool:
    from role_set.critic import same_answer

    ca, cb = closed_answer(a), closed_answer(b)
    return ca is not None and cb is not None and same_answer(ca, cb)


def check_puzzle(fields: dict, structure: str | None) -> Checked:
    """Shape, text, solver, key and reply, in that order; the first failure is the refusal."""

    m = parse_model(fields.get("model"))
    if structure in VERDICT_STRUCTURES and m.claim is None and structure != "when_to_stop":
        raise PuzzleError(f"a {structure.replace('_', ' ')} puzzle is keyed by a verdict: give model.claim, the rule the question asks about")
    _, question = check_text(m, str(fields.get("puzzle") or ""))
    got = solve(m, structure, question)
    key = str(fields.get("answer") or "").strip()
    if structure == "underdetermined_with_explicit_set":
        if re.fullmatch(r"undetermined\.?", key, re.IGNORECASE):
            if not UNIQUENESS_ASKED.search(question):
                raise PuzzleError("'undetermined' answers a question that asks whether the answer is unique; this one doesn't")
        elif {x.strip().lower() for x in key.split(",") if x.strip()} != {x.strip().lower() for x in got.answer.split(",")}:
            raise PuzzleError(f"the facts leave exactly {got.answer!r} possible, and the key says {key!r}")
    elif got.stop_at is not None:
        digits = re.search(r"\d+", key)
        if not digits or int(digits.group(0)) != got.stop_at:
            raise PuzzleError(f"the answer is settled after line {got.stop_at}, so the key is {got.stop_at}, not {key!r}")
    elif not _same(key, got.answer):
        raise PuzzleError(f"the facts give {got.answer!r}, and the key says {key!r}")
    last = [x for x in str(fields.get("reply") or "").splitlines() if x.strip()]
    tail = last[-1].split(":", 1)[-1] if last else ""
    loose = got.stop_at is not None or structure == "underdetermined_with_explicit_set"
    if not last or not (_same(tail, key) or (loose and " ".join(tail.lower().split()).strip(" .") == " ".join(key.lower().split()).strip(" ."))
                        or (got.stop_at is not None and re.search(rf"\b{got.stop_at}\b", tail))):
        raise PuzzleError("the reply ends with the answer on its own line, in the key's closed form ('Answer: ...')")
    return got
