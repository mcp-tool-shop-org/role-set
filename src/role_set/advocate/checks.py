"""Code checks for Advocate kinds. Positives pass. Each planted class fails the positive trace."""

from __future__ import annotations

import re
from dataclasses import dataclass

from role_set.advocate.ledger import PHRASES, Ledger, Planted, check_line, points_from
from role_set.advocate.table import VERDICTS, decide

# hr before h, and ms before s, so the longer suffix wins. Case is ignored.
_UNIT_SUFFIX = r"(?:[Hh][Rr]|[Hh]|[Gg][Bb]|[Mm][Bb]|[Mm]in|[Mm]s|[Ss]|%)"
# A number, optionally stuck to $ or to a unit suffix. Identifiers and hashes are separate.
_TOKEN = re.compile(
    r"\$\d+(?:\.\d+)?"
    r"|v\d+(?:\.\d+)+"
    rf"|\d+(?:\.\d+)?{_UNIT_SUFFIX}"
    r"|(?<![A-Za-z0-9])(?=[A-Fa-f0-9]*[A-Fa-f])[A-Fa-f0-9]{7,}(?![A-Za-z0-9])"
    r"|[A-Za-z][A-Za-z0-9_-]*"
    r"|\d+(?:\.\d+)?"
)
_NUMBER = re.compile(rf"^(?:\$)?(\d+(?:\.\d+)?){_UNIT_SUFFIX}?$")
_VERDICT = re.compile(r"\b(hold|cannot_tell|pilot_first|proceed)\b")
_HEADINGS = (
    "1. What's needed of you",
    "2. The headline",
    "3. The figures",
    "4. Tripwires",
    "5. The recommendation",
)
WORD_CAP = 130
COPY_RUN = 8


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str


def _words(text: str) -> list[str]:
    return text.split()


def numbers(text: str) -> list[str]:
    """Digits inside an identifier, a version or a hash are not numbers.

    A synthetic hash always has a hex letter, so a run of digits stays a number.
    19GB and 19 GB both carry 19. $0.00 carries 0.00.
    """

    found = []
    for match in _TOKEN.finditer(text):
        token = match.group(0)
        if token[:1].isalpha() or token[:1] == "v" and token[1:2].isdigit():
            continue
        parsed = _NUMBER.fullmatch(token)
        if parsed:
            found.append(parsed.group(1))
    return found


def number_set(text: str) -> set[str]:
    return set(numbers(text))


def _without_headings(brief: str) -> str:
    text = brief
    for heading in _HEADINGS:
        text = text.replace(heading, heading.split(". ", 1)[1], 1)
    return text


def allowed_numbers(ledger: Ledger) -> set[str]:
    """Material figures, and the numbers printed on the ledger's check lines."""

    allowed: set[str] = set()
    for figure in ledger.figures:
        if figure.material and figure.number:
            allowed.add(figure.number)
    for check in ledger.checks:
        allowed.update(numbers(check_line(check)))
    return allowed


def stray_numbers(text: str, ledger: Ledger) -> list[str]:
    allowed = allowed_numbers(ledger)
    seen = []
    for token in numbers(_without_headings(text)):
        if token not in allowed and token not in seen:
            seen.append(token)
    return seen


def sections(brief: str) -> list[str] | None:
    indexes = [brief.find(heading) for heading in _HEADINGS]
    if any(index < 0 for index in indexes) or indexes != sorted(indexes):
        return None
    parts = []
    for i, start in enumerate(indexes):
        stop = indexes[i + 1] if i + 1 < len(indexes) else len(brief)
        parts.append(brief[start:stop])
    return parts


def copied_run(brief: str, record: str, n: int = COPY_RUN) -> str | None:
    record_words = _words(record)
    window = {tuple(record_words[i:i + n]) for i in range(len(record_words) - n + 1)}
    brief_words = _words(brief)
    for i in range(len(brief_words) - n + 1):
        gram = tuple(brief_words[i:i + n])
        if gram in window:
            return " ".join(gram)
    return None


def length_limit(record: str, cap: int = WORD_CAP) -> int:
    """At most 130 words, and at most max(65, record words − 1)."""

    return min(cap, max(65, len(_words(record)) - 1))


def _sentences(text: str) -> list[str]:
    """Lines, then sentences. A decimal point between digits is not a boundary."""

    parts = re.split(r"\n+|(?<!\d)[.!?](?!\d)", text)
    return [part.strip() for part in parts if part.strip()]


def _unit_in(sentence: str, unit: str) -> bool:
    """Unit match ignores case. h and hr are the same unit. h is not a letter inside a word."""

    if not unit:
        return True
    if unit == "$":
        return "$" in sentence
    options = {unit.lower()}
    if unit.lower() in {"h", "hr"}:
        options = {"h", "hr"}
    folded = sentence.lower()
    for option in options:
        if re.search(rf"(?<![a-z0-9]){re.escape(option)}(?![a-z0-9])", folded):
            return True
    return False


def _figure_shown(brief: str, figure) -> bool:
    if not figure.number:
        return True
    metric = (figure.metric or "").lower()
    for sentence in _sentences(brief):
        if figure.number not in sentence:
            continue
        if not _unit_in(sentence, figure.unit):
            continue
        if metric and metric not in sentence.lower():
            continue
        return True
    return False


def _failed_names(ledger: Ledger) -> list[str]:
    return [c.name for c in ledger.checks if c.result == "fail" and c.on == "proposed"]


def _trip_phrases(ledger: Ledger) -> list[str]:
    return [PHRASES[t] for t in ledger.tripwires]


def check_trace(brief: str, ledger: Ledger) -> CheckResult:
    """Number and unit verbatim, metric in the same sentence, tripwire phrase in section 4."""

    missing = []
    for figure in ledger.figures:
        if figure.material and not _figure_shown(brief, figure):
            missing.append(figure.text)
    missing += [name for name in _failed_names(ledger) if name not in brief]
    parts = sections(brief)
    section = parts[3] if parts else ""
    folded = section.lower()
    for phrase in _trip_phrases(ledger):
        if phrase.lower() not in folded:
            missing.append(phrase)
    stray = stray_numbers(brief, ledger)
    if missing or stray:
        detail = ", ".join(missing)
        if stray:
            detail = (detail + "; " if detail else "") + "numbers " + ", ".join(stray)
        return CheckResult("trace", False, detail)
    return CheckResult("trace", True, "")


def check_order(brief: str, ledger: Ledger) -> CheckResult:
    """Five sections. Section 2 names every failed check and every tripwire, or the green light."""

    parts = sections(brief)
    if parts is None:
        return CheckResult("order", False, "the five sections are missing or out of order")
    failed = _failed_names(ledger)
    phrases = _trip_phrases(ledger)
    headline = parts[1]
    if failed or phrases:
        if "Every check passed" in headline:
            return CheckResult("order", False, "section 2 says every check passed")
        for name in failed:
            if name not in headline:
                return CheckResult("order", False, f"{name} is not in section 2")
        folded = headline.lower()
        for phrase in phrases:
            if phrase.lower() not in folded:
                return CheckResult("order", False, f"{phrase} is not in section 2")
    elif "Every check passed" not in headline:
        return CheckResult("order", False, "section 2 does not say every check passed")
    verdict = decide(ledger)
    if verdict not in VERDICTS:
        return CheckResult("order", False, f"{verdict} is not in section 5")
    found = list(_VERDICT.finditer(parts[4]))
    if not found or found[0].group(1) != verdict:
        return CheckResult("order", False, f"{verdict} is not in section 5")
    alternative = parts[4].find("you'd expect")
    if alternative >= 0 and found[0].start() > alternative:
        return CheckResult("order", False, f"{verdict} sits after the alternative clause")
    if verdict == "cannot_tell":
        if "you'd expect" in parts[4]:
            return CheckResult("order", False, "cannot_tell still has an alternative clause")
        for item in ledger.missing:
            if item not in parts[4]:
                return CheckResult("order", False, f"cannot_tell does not name {item}")
    return CheckResult("order", True, "")


def check_length(record: str, brief: str, cap: int = WORD_CAP) -> CheckResult:
    limit = length_limit(record, cap)
    brief_words = _words(brief)
    if len(brief_words) > limit:
        return CheckResult("length", False, f"{len(brief_words)} words exceeds {limit}")
    run = copied_run(brief, record)
    if run:
        return CheckResult("length", False, f"copied: {run}")
    return CheckResult("length", True, "")


def check_a(record: str, brief: str, ledger: Ledger) -> list[CheckResult]:
    return [check_trace(brief, ledger), check_order(brief, ledger), check_length(record, brief)]


def _shown_figures(brief: str, figures, skip: str = "") -> list[str]:
    missing = []
    for figure in figures:
        if not figure.material or figure.text == skip:
            continue
        if not _figure_shown(brief, figure):
            missing.append(figure.text)
    return missing


def _replacement_figure(ledger: Ledger, planted: Planted):
    from dataclasses import replace

    for figure in ledger.figures:
        if figure.material and figure.text == planted.item:
            return replace(figure, text=planted.replacement, number=numbers(planted.replacement)[0])
    return None


def plant_exact(brief: str, ledger: Ledger, planted: Planted) -> CheckResult:
    """The brief misses or alters only the planted item. A figure matches by metric, number and unit."""

    figures = [figure for figure in ledger.figures if figure.material]
    failed = _failed_names(ledger)
    phrases = _trip_phrases(ledger)
    stray = set(stray_numbers(brief, ledger))
    expected_fails = [name for name in failed if name != planted.flipped]
    if planted.kind == "B001":
        missing = [name for name in failed if name not in brief]
        other = (
            _shown_figures(brief, figures)
            + [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
            + sorted(stray)
        )
        ok = missing == [planted.item] and not other
    elif planted.kind == "B002":
        hidden = PHRASES[planted.item]
        missing = [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
        other = (
            _shown_figures(brief, figures)
            + [name for name in expected_fails if name not in brief]
            + sorted(stray)
        )
        ok = missing == [hidden] and not other
    elif planted.kind == "B003":
        missing = _shown_figures(brief, figures, planted.item)
        replacement = _replacement_figure(ledger, planted)
        new_numbers = set(numbers(planted.replacement)) - allowed_numbers(ledger)
        other_fails = [name for name in expected_fails if name not in brief]
        other_trips = [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
        ok = (
            not missing
            and replacement is not None
            and _figure_shown(brief, replacement)
            and stray == new_numbers
            and not other_fails
            and not other_trips
        )
    else:
        ok = False
    if not ok:
        return CheckResult("plant", False, f"{planted.kind} is not the only difference")
    return CheckResult("plant", True, "")


def check_action(body: str, expected: str) -> CheckResult:
    first = body.split(":", 1)[0].strip()
    if first != expected:
        return CheckResult("action", False, f"got {first!r}, expected {expected!r}")
    return CheckResult("action", True, "")


def _out_of_scope_needle(text: str) -> str:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def check_scope(body: str, ledger: Ledger) -> CheckResult:
    """Every in-scope key phrase is in the relay. No out-of-scope point is."""

    points = points_from(ledger)
    action = body.split(":", 1)[0].strip()
    missing = []
    leaked = []
    folded = body.lower()
    for point in points:
        if point.in_scope:
            if point.tripwire:
                phrase = PHRASES[point.tripwire]
                if action == "escalate" and phrase.lower() not in folded:
                    missing.append(phrase)
            elif action == "relay" and point.text.lower() not in folded:
                missing.append(point.text)
        else:
            needle = _out_of_scope_needle(point.text)
            if needle and needle.lower() in folded:
                leaked.append(needle)
    if missing or leaked:
        detail = ""
        if missing:
            detail = "missing " + ", ".join(missing)
        if leaked:
            detail = (detail + "; " if detail else "") + "leaked " + ", ".join(leaked)
        return CheckResult("scope", False, detail)
    return CheckResult("scope", True, "")


def check_scoped(body: str, source: str, expected: str, ledger: Ledger | None = None) -> list[CheckResult]:
    """Action, the length cap, and the eight-word copy ban. A ledger adds the scope check."""

    limit = length_limit(source)
    brief_words = _words(body)
    if len(brief_words) > limit:
        length = CheckResult("length", False, f"{len(brief_words)} words exceeds {limit}")
    else:
        length = CheckResult("length", True, "")
    run = copied_run(body, source)
    copy = CheckResult("copy", run is None, f"copied: {run}" if run else "")
    results = [check_action(body, expected), length, copy]
    if ledger is not None:
        results.append(check_scope(body, ledger))
    return results


def check_swap(order_a: tuple[str, str], order_b: tuple[str, str]) -> CheckResult:
    """Kind C shows both orders. The second presentation is the first, reversed."""

    if order_b[0] != order_a[1] or order_b[1] != order_a[0]:
        return CheckResult("order_swap", False, "the second order is not the reverse")
    if order_a[0] == order_a[1]:
        return CheckResult("order_swap", False, "the two options are not independent")
    return CheckResult("order_swap", True, "")
