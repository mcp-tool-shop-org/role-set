"""Code checks for Advocate kinds. Positives pass. Each planted class fails the positive trace."""

from __future__ import annotations

import re
from dataclasses import dataclass

from role_set.advocate.ledger import PHRASES, Ledger, Planted
from role_set.advocate.table import VERDICTS, decide

# A number, optionally stuck to $ or to a unit suffix. Identifiers and hashes are separate.
_TOKEN = re.compile(
    r"\$\d+(?:\.\d+)?"
    r"|v\d+(?:\.\d+)+"
    r"|\d+(?:\.\d+)?(?:GB|MB|min|ms|s|%)"
    r"|(?<![A-Za-z0-9])(?=[A-Fa-f0-9]*[A-Fa-f])[A-Fa-f0-9]{7,}(?![A-Za-z0-9])"
    r"|[A-Za-z][A-Za-z0-9_-]*"
    r"|\d+(?:\.\d+)?"
)
_NUMBER = re.compile(r"^(?:\$)?(\d+(?:\.\d+)?)(?:GB|MB|min|ms|s|%)?$")
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
    allowed: set[str] = set()
    for figure in ledger.figures:
        if figure.material and figure.number:
            allowed.add(figure.number)
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


def _figure_shown(brief: str, figure) -> bool:
    if not figure.number:
        return True
    metric = (figure.metric or figure.text.split()[0]).lower()
    for sentence in _sentences(brief):
        if figure.number not in sentence:
            continue
        if figure.unit and figure.unit not in sentence:
            continue
        if metric in sentence.lower():
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
    """Five sections. The first verdict token in section 5 is the key, before the alternative."""

    parts = sections(brief)
    if parts is None:
        return CheckResult("order", False, "the five sections are missing or out of order")
    for name in _failed_names(ledger):
        if name not in parts[1]:
            return CheckResult("order", False, f"{name} is not in section 2")
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


def plant_exact(brief: str, ledger: Ledger, planted: Planted) -> CheckResult:
    """The brief misses or alters only the planted item. Headline and verdict may follow the hidden ledger."""

    figures = [figure.text for figure in ledger.figures if figure.material]
    failed = _failed_names(ledger)
    phrases = _trip_phrases(ledger)
    stray = set(stray_numbers(brief, ledger))
    flipped = planted.flipped
    expected_fails = [name for name in failed if name != flipped]
    if planted.kind == "B001":
        missing = [name for name in failed if name not in brief]
        other = (
            [text for text in figures if text not in brief]
            + [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
            + sorted(stray)
        )
        ok = missing == [planted.item] and not other
    elif planted.kind == "B002":
        hidden = PHRASES[planted.item]
        missing = [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
        other = (
            [text for text in figures if text not in brief]
            + [name for name in expected_fails if name not in brief]
            + sorted(stray)
        )
        ok = missing == [hidden] and not other
    elif planted.kind == "B003":
        missing = [text for text in figures if text not in brief]
        new_numbers = set(numbers(planted.replacement)) - allowed_numbers(ledger)
        other_fails = [name for name in expected_fails if name not in brief]
        other_trips = [phrase for phrase in phrases if phrase.lower() not in brief.lower()]
        ok = (
            missing == [planted.item]
            and planted.replacement in brief
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


def check_scoped(body: str, source: str, expected: str) -> list[CheckResult]:
    """Action, the length cap, and the eight-word copy ban. Used for B and D."""

    limit = length_limit(source)
    brief_words = _words(body)
    if len(brief_words) > limit:
        length = CheckResult("length", False, f"{len(brief_words)} words exceeds {limit}")
    else:
        length = CheckResult("length", True, "")
    run = copied_run(body, source)
    copy = CheckResult("copy", run is None, f"copied: {run}" if run else "")
    return [check_action(body, expected), length, copy]


def check_swap(order_a: tuple[str, str], order_b: tuple[str, str]) -> CheckResult:
    """Kind C shows both orders. The second presentation is the first, reversed."""

    if order_b[0] != order_a[1] or order_b[1] != order_a[0]:
        return CheckResult("order_swap", False, "the second order is not the reverse")
    if order_a[0] == order_a[1]:
        return CheckResult("order_swap", False, "the two options are not independent")
    return CheckResult("order_swap", True, "")
