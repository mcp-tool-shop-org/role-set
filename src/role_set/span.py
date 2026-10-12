"""Dated spans for kind B: which edition was in force on a date, worked out by the solver (R&D's rulings of 2026-10-11 and
2026-10-12 on role-set's `contains`).

A model declares a span attribute: an ordered attribute whose values are spans, each with a start and an end day drawn from
the item's own named days. Clues map entities to spans with the ordinary forms (offset, before, is). The solver then
computes `contains`: which entity holds a named day. A question can ask it ({"contains": day}) and a claim can state it
(["contains", entity, day]); a fact line never can.

    "span": {"attribute": "span",
             "days": [{"label": "1 May", "day": 1, "month": 5, "ordinal": 0}, ...],
             "bounds": {"1 May to 31 August": ["1 May", "31 August"], ...},
             "owner": {"rule": "later", "line": 4}}           # optional

Days are only the days the item names, in calendar order; a year wrap needs "year" on the records. A day label carries no
year ("1 January", never "1 January 2026"): the year is the record's "year" field, which only orders a wrap. A day held by
two spans goes to the owner rule's span (by start), or else both holders stand; an owner rule needs two spans that share a
day.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December")
MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
OWNER_RULES = ("earlier", "later")


class SpanError(ValueError):
    """The span declaration doesn't hold. The message is the suggestion."""


@dataclass(frozen=True)
class Spans:
    attribute: str
    ordinals: dict[str, int]  # day label (lower case) -> ordinal
    labels: dict[str, str]  # day label (lower case) -> the label as declared
    bounds: dict[str, tuple[int, int]]  # span value -> (start ordinal, end ordinal)
    owner: dict | None  # {"rule": "earlier" | "later", "line": n}
    months: dict[int, int] = None  # ordinal -> month, for readings that look at the month alone (R&D: typed records)

    def day(self, label) -> int:
        if not isinstance(label, str) or label.lower() not in self.ordinals:
            raise SpanError(f"{label!r} is not one of the item's named days; a question or claim names a declared day, "
                            "never a date between them")
        return self.ordinals[label.lower()]

    def label(self, label: str) -> str:
        return self.labels[label.lower()]

    def holders(self, values: list[str], day: int) -> list[int]:
        """The indexes of the entities holding `day`, given each entity's span value: every span containing the day
        (inclusive at both ends), narrowed by the owner rule to the earliest- or latest-starting span."""

        inside = [i for i, v in enumerate(values) if self.bounds[v][0] <= day <= self.bounds[v][1]]
        if len({values[i] for i in inside}) > 1 and self.owner:
            starts = {values[i]: self.bounds[values[i]][0] for i in inside}
            pick = (max if self.owner["rule"] == "later" else min)(starts.values())
            inside = [i for i in inside if starts[values[i]] == pick]
        return inside


def _label_for(rec: dict) -> str:
    return f"{rec['day']} {MONTHS[rec['month'] - 1]}"


def parse_spans(raw, attrs: dict[str, dict]) -> Spans:
    if not isinstance(raw, dict):
        raise SpanError("model.span is {\"attribute\": a, \"days\": [...], \"bounds\": {...}} with an optional \"owner\"")
    a = raw.get("attribute")
    if a not in attrs or not attrs[a]["ordered"]:
        raise SpanError("span.attribute names an ordered attribute whose values are the spans")
    days = raw.get("days")
    if not isinstance(days, list) or len(days) < 2:
        raise SpanError("span.days lists the item's named days, two or more")
    ordinals: dict[str, int] = {}
    months: dict[int, int] = {}
    labels: dict[str, str] = {}
    prev = None
    for rec in days:
        if not isinstance(rec, dict) or not all(isinstance(rec.get(k), int) for k in ("day", "month", "ordinal")) \
                or not isinstance(rec.get("label"), str) or ("year" in rec and not isinstance(rec["year"], int)):
            raise SpanError("a day record is {\"label\", \"day\", \"month\", \"ordinal\"} with an optional whole-number \"year\"")
        if not 1 <= rec["month"] <= 12 or not 1 <= rec["day"] <= MONTH_DAYS[rec["month"] - 1]:
            raise SpanError(f"{rec['label']!r}: day {rec['day']} of month {rec['month']} is not a calendar date")
        if rec["label"].strip().lower() != _label_for(rec).lower():
            raise SpanError(f"the day record {rec['label']!r} says day {rec['day']}, month {rec['month']}; its label should "
                            f"read {_label_for(rec)!r}")
        key = rec["label"].strip().lower()
        if key in ordinals:
            raise SpanError(f"{rec['label']!r} is declared twice")
        cal = (rec.get("year", 0), rec["month"], rec["day"])
        if prev is not None:
            if rec["ordinal"] <= prev[0]:
                raise SpanError("day ordinals rise strictly through span.days")
            if cal <= prev[1]:
                raise SpanError(f"{rec['label']!r} comes before {prev[2]!r} in the calendar but after it in ordinal order; "
                                "list the days in calendar order, and mark a year wrap with \"year\" on the records")
        prev = (rec["ordinal"], cal, rec["label"])
        ordinals[key] = rec["ordinal"]
        months[rec["ordinal"]] = rec["month"]
        labels[key] = rec["label"].strip()
    bounds_raw = raw.get("bounds")
    values = attrs[a]["values"]
    if not isinstance(bounds_raw, dict) or {k.lower() for k in bounds_raw} != {v.lower() for v in values}:
        raise SpanError(f"span.bounds gives a start and an end day for every value of {a!r}, and nothing else")
    bounds: dict[str, tuple[int, int]] = {}
    for v in values:
        pair = next(p for k, p in bounds_raw.items() if k.lower() == v.lower())
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) and x.lower() in ordinals for x in pair):
            raise SpanError(f"the span {v!r} needs [start day, end day], both from span.days")
        lo, hi = ordinals[pair[0].lower()], ordinals[pair[1].lower()]
        if lo > hi:
            raise SpanError(f"the span {v!r} starts after it ends")
        expect = f"{labels[pair[0].lower()]} to {labels[pair[1].lower()]}"
        if not re.fullmatch(re.escape(expect), v.strip(), re.IGNORECASE):
            raise SpanError(f"the span {v!r} has the bounds {pair[0]!r} and {pair[1]!r}; its label should read {expect!r}")
        bounds[v] = (lo, hi)
    starts = [bounds[v][0] for v in values]
    if len(set(starts)) != len(starts):
        raise SpanError("two spans share a start day, so ordering by start has a tie; give each span its own start")
    if starts != sorted(starts):
        raise SpanError(f"the values of {a!r} are listed in start order, earliest first")
    owner = raw.get("owner")
    if owner is not None and (not isinstance(owner, dict) or owner.get("rule") not in OWNER_RULES
                              or not isinstance(owner.get("line"), int)):
        raise SpanError("span.owner is {\"rule\": \"earlier\" | \"later\", \"line\": n}, the line that states which span "
                        "owns a day both hold")
    if owner is not None and not overlapping(bounds):
        # An owner rule over spans that never share a day decides nothing, and its line would count against nothing
        # (R&D on role-set#47).
        raise SpanError("span.owner rules on a day two spans share, and no two declared spans share a day; drop the owner "
                        "rule and its line")
    return Spans(a, ordinals, labels, bounds, owner, months)


def overlapping(bounds: dict[str, tuple[int, int]]) -> bool:
    """Whether any two declared spans share a day (inclusive bounds)."""

    spans = sorted(bounds.values())
    return any(nxt[0] <= cur[1] for cur, nxt in zip(spans, spans[1:]))
