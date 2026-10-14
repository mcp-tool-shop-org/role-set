"""Code checks for Advocate kinds. Positives pass. Each planted class fails the positive trace."""

from __future__ import annotations

import re
from dataclasses import dataclass

from role_set.advocate.ledger import TRIPWIRES, Ledger, Planted
from role_set.advocate.table import decide

_NUM = re.compile(r"\d+(?:\.\d+)?")
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
    return _NUM.findall(text)


def allowed_numbers(ledger: Ledger) -> set[str]:
    allowed = {"1", "2", "3", "4", "5"}
    for figure in ledger.figures:
        if figure.material:
            allowed.update(numbers(figure.text))
    return allowed


def stray_numbers(text: str, ledger: Ledger) -> list[str]:
    allowed = allowed_numbers(ledger)
    return [token for token in numbers(text) if token not in allowed]


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


def _failed_names(ledger: Ledger) -> list[str]:
    return [c.name for c in ledger.checks if c.result == "fail" and c.on == "proposed"]


def _trip_names(ledger: Ledger) -> list[str]:
    return [TRIPWIRES[t] for t in ledger.tripwires]


def _figure_texts(ledger: Ledger) -> list[str]:
    return [figure.text for figure in ledger.figures if figure.material]


def check_trace(brief: str, ledger: Ledger) -> CheckResult:
    """Every material figure, proposed failure and tripwire is present, and no other number."""

    missing = [text for text in _figure_texts(ledger) if text not in brief]
    missing += [name for name in _failed_names(ledger) if name not in brief]
    missing += [name for name in _trip_names(ledger) if name not in brief]
    stray = stray_numbers(brief, ledger)
    if missing or stray:
        detail = ", ".join(missing)
        if stray:
            detail = (detail + "; " if detail else "") + "numbers " + ", ".join(stray)
        return CheckResult("trace", False, detail)
    return CheckResult("trace", True, "")


def check_order(brief: str, ledger: Ledger) -> CheckResult:
    """Five sections in order. Proposed failures sit in section 2. The verdict is in section 5."""

    parts = sections(brief)
    if parts is None:
        return CheckResult("order", False, "the five sections are missing or out of order")
    for name in _failed_names(ledger):
        if name not in parts[1]:
            return CheckResult("order", False, f"{name} is not in section 2")
    verdict = decide(ledger)
    section = parts[4]
    if verdict not in section:
        return CheckResult("order", False, f"{verdict} is not in section 5")
    alternative = section.find("you'd expect")
    if alternative >= 0 and section.find(verdict) > alternative:
        return CheckResult("order", False, f"{verdict} sits after the alternative clause")
    if verdict == "cannot_tell":
        for item in ledger.missing:
            if item not in section:
                return CheckResult("order", False, f"cannot_tell does not name {item}")
    return CheckResult("order", True, "")


def check_length(record: str, brief: str, cap: int = WORD_CAP) -> CheckResult:
    brief_words = _words(brief)
    record_words = _words(record)
    if len(brief_words) > cap:
        return CheckResult("length", False, f"{len(brief_words)} words exceeds {cap}")
    if len(brief_words) >= len(record_words):
        return CheckResult("length", False, "the brief is not shorter than the record")
    run = copied_run(brief, record)
    if run:
        return CheckResult("length", False, f"copied: {run}")
    return CheckResult("length", True, "")


def check_a(record: str, brief: str, ledger: Ledger) -> list[CheckResult]:
    return [check_trace(brief, ledger), check_order(brief, ledger), check_length(record, brief)]


def plant_exact(brief: str, ledger: Ledger, planted: Planted) -> CheckResult:
    """The brief misses or alters only the one planted item."""

    figures = _figure_texts(ledger)
    failed = _failed_names(ledger)
    trips = _trip_names(ledger)
    stray = stray_numbers(brief, ledger)
    if planted.kind == "B001":
        missing = [name for name in failed if name not in brief]
        other = (
            [text for text in figures if text not in brief]
            + [name for name in trips if name not in brief]
            + stray
        )
        ok = missing == [planted.item] and not other
    elif planted.kind == "B002":
        hidden = TRIPWIRES[planted.item]
        missing = [name for name in trips if name not in brief]
        other = (
            [text for text in figures if text not in brief]
            + [name for name in failed if name not in brief]
            + stray
        )
        ok = missing == [hidden] and not other
    elif planted.kind == "B003":
        missing = [text for text in figures if text not in brief]
        new_numbers = numbers(planted.replacement)
        other = [name for name in failed if name not in brief] + [name for name in trips if name not in brief]
        ok = (
            missing == [planted.item]
            and planted.replacement in brief
            and stray == new_numbers
            and not other
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

    brief_words = _words(body)
    source_words = _words(source)
    if len(brief_words) > WORD_CAP:
        length = CheckResult("length", False, f"{len(brief_words)} words exceeds {WORD_CAP}")
    elif len(brief_words) >= len(source_words):
        length = CheckResult("length", False, "the text is not shorter than its source")
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
