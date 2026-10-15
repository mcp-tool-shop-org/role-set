"""Synthetic ledgers for the Advocate. Each shape renders once, in its studio format.

A baseline failure justifies a paid pod. It is not a failed check on the proposed run.
On a release record, the public-or-irreversible phrase is tagged only when the release
is unapproved or provenance or the published-file check failed. A clean release waiting
on the Director carries no such tripwire.

The writer, the critic, and the length and copy checks all read this one rendering.
There is no second record and no padding paragraph.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from role_set.advocate.table import decide

# Display phrases are section 0a's wording. The ids stay internal.
PHRASES = {
    "B004": "paid spend",
    "B005": "a public or irreversible act",
    "B006": "personal data",
    "B007": "the Director's own words in a file",
    "B008": "device use without a grant",
    "B009": "a merge without review",
    "B010": "a floor missed",
    "B011": "a frozen analysis changed",
    "B012": "a second-hand approval",
}
TRIPWIRES = {
    "B004": "paid_spend",
    "B005": "public_or_irreversible",
    "B006": "personal_data",
    "B007": "directors_words",
    "B008": "device_without_grant",
    "B009": "merge_without_review",
    "B010": "floor_missed",
    "B011": "frozen_analysis_changed",
    "B012": "approval_second_hand",
}
PLANTS = ("B001", "B002", "B003")
TITLES = {
    "seat_handback": "Seat hand-back",
    "reviewer_verdict": "Reviewer verdict",
    "ci_run": "CI run",
    "identity_scan": "Identity scan",
    "device_grant": "Device-grant log",
    "paid_run_plan": "Paid-run plan",
    "frozen_analysis": "Frozen analysis",
    "security_export": "Security export",
    "release_record": "Release record",
}
DIRECTIONS = ("director-to-seat", "seat-to-director", "reviewer-to-builder")
CATEGORIES = ("figure", "instruction", "ask", "cost", "act")
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_HOME = re.compile(r"(?i)(?:[A-Za-z]:\\Users\\|\\Users\\|/Users/|/home/)")
_NAME = re.compile(r"\b([A-Z][a-z]{2,}) ([A-Z][a-z]{2,})\b")
_ROLE_WORDS = {"the", "director", "publisher", "reviewer", "builder", "seat", "research"}
_ASKS = (
    ("whether to stop",),
    ("whether to publish",),
    ("whether to rerun",),
    (),
)
_MISSING = (
    (),
    ("decision threshold",),
    ("printed cost",),
    ("grant id",),
)
_CHECKS = (
    (),
    (("coverage floor", "fail", "proposed"),),
    (("coverage floor", "fail", "proposed"), ("patch coverage", "fail", "proposed")),
    (("unit check", "pass", "proposed"),),
    (("local baseline", "fail", "baseline"),),
    (("coverage floor", "fail", "proposed"), ("local baseline", "fail", "baseline")),
)
_TRIPS = (
    (),
    ("B004",),
    ("B005",),
    ("B006",),
    ("B007",),
    ("B008",),
    ("B009",),
    ("B010",),
    ("B011",),
    ("B012",),
    ("B004", "B006"),
    ("B006", "B012"),
    ("B005", "B010"),
    ("B008", "B009"),
    ("B010", "B011"),
    ("B007", "B012"),
)
# over_cap, paid_model_api, paid_run, rollout, pilot_passed
_FLAGS = (
    (False, False, False, False, None),
    (False, False, True, False, True),
    (False, False, True, False, False),
    (False, False, True, False, None),
    (False, False, False, True, True),
    (False, False, False, True, False),
    (True, False, True, False, True),
    (False, True, False, False, True),
    (False, True, True, False, True),
    (True, False, False, False, None),
    (False, False, True, True, False),
    (False, False, False, False, True),
)
_RELEASE_MODES = ("clean", "unapproved", "provenance", "published")
_REASONS = {
    "a": {
        "hold": "a rule above proceed matched",
        "pilot_first": "no local pilot has passed",
        "proceed": "no earlier rule matched",
    },
    "b": {
        "hold": "the first matching rule stops the run",
        "pilot_first": "the rollout still needs a passed pilot",
        "proceed": "the record clears the table",
    },
}
_DIRECTION_FOR = {
    "seat_handback": "seat-to-director",
    "reviewer_verdict": "reviewer-to-builder",
    "ci_run": "director-to-seat",
    "identity_scan": "seat-to-director",
    "device_grant": "director-to-seat",
    "paid_run_plan": "seat-to-director",
    "frozen_analysis": "reviewer-to-builder",
    "security_export": "seat-to-director",
    "release_record": "director-to-seat",
}


class LedgerError(ValueError):
    """The ledger or its rendered text is not allowed."""


class PlantError(ValueError):
    """This ledger cannot carry that plant."""


# A metric is a noun in the line. It is never the first word just because it is first.
_METRIC_NOUNS = (
    "dry run",
    "cost",
    "coverage",
    "memory",
    "files",
    "floor",
    "elapsed",
    "hours",
    "cap",
    "score",
    "trials",
    "secrets",
    "artifacts",
    "size",
    "rounds",
    "checks",
    "time",
    "hit",
    "baseline",
)


def _natural_metric(text: str) -> str:
    folded = text.lower()
    for noun in _METRIC_NOUNS:
        if noun in folded:
            return noun
    return ""


@dataclass(frozen=True)
class Figure:
    text: str
    material: bool
    number: str
    unit: str = ""
    metric: str = ""
    bound: str | None = None
    bound_number: str | None = None
    bound_check: str | None = None

    def __post_init__(self) -> None:
        if not self.metric and self.text:
            object.__setattr__(self, "metric", _natural_metric(self.text))


@dataclass(frozen=True)
class Check:
    name: str
    result: str
    on: str
    evidence: str = ""


@dataclass(frozen=True)
class Point:
    text: str
    direction: str
    category: str
    in_scope: bool
    tripwire: str | None = None


@dataclass(frozen=True)
class Ledger:
    shape: str
    figures: tuple[Figure, ...]
    checks: tuple[Check, ...]
    asks: tuple[str, ...]
    tripwires: tuple[str, ...]
    missing: tuple[str, ...]
    over_cap: bool
    paid_model_api: bool
    paid_run: bool
    rollout: bool
    pilot_passed: bool | None
    release_approved: bool | None = None
    provenance_ok: bool | None = None
    published_file_ok: bool | None = None
    points: tuple[Point, ...] = ()
    context: str = ""

    def __post_init__(self) -> None:
        for check in self.checks:
            if check.result not in ("pass", "fail") or check.on not in ("proposed", "baseline"):
                raise LedgerError(f"REFUSED: check {check.name!r} is not pass/fail on proposed/baseline")
        for trip in self.tripwires:
            if trip not in TRIPWIRES:
                raise LedgerError(f"REFUSED: tripwire {trip!r} is not in the catalogue")
        for point in self.points:
            if point.direction not in DIRECTIONS:
                raise LedgerError(f"REFUSED: direction {point.direction!r} is not a studio direction")
            if point.category not in CATEGORIES:
                raise LedgerError(f"REFUSED: category {point.category!r} is not a point category")
            if point.tripwire is not None and point.tripwire not in TRIPWIRES:
                raise LedgerError(f"REFUSED: point tripwire {point.tripwire!r} is not in the catalogue")


@dataclass(frozen=True)
class Planted:
    kind: str
    hidden: Ledger
    item: str
    replacement: str = ""
    flipped: str = ""


def allowed_shapes(pack_dir: Path) -> list[str]:
    """The nine ids in sources.json. Sorted, so the bank does not depend on file order."""

    data = json.loads((pack_dir / "sources.json").read_text(encoding="utf-8"))
    return sorted(row["id"] for row in data["records"])


def assert_shape(shape: str, allowed: list[str] | tuple[str, ...]) -> None:
    if shape not in allowed:
        raise LedgerError(f"REFUSED: shape {shape!r} is not in sources.json")


def scan_text(text: str) -> None:
    """Refuse an email, a home path, or a personal name. Role titles are allowed."""

    if _EMAIL.search(text) or _HOME.search(text):
        raise LedgerError("REFUSED: the record carries an address or a home path")
    for match in _NAME.finditer(text):
        if match.group(1).lower() not in _ROLE_WORDS or match.group(2).lower() not in _ROLE_WORDS:
            raise LedgerError("REFUSED: the record carries a personal name")


def apply_release_rule(ledger: Ledger) -> Ledger:
    """Tag B005 only for an unapproved release or a failed provenance or published-file check."""

    if ledger.shape != "release_record":
        return ledger
    trips = [t for t in ledger.tripwires if t != "B005"]
    bad = (
        ledger.release_approved is False
        or ledger.provenance_ok is False
        or ledger.published_file_ok is False
    )
    if bad:
        trips.append("B005")
    return replace(ledger, tripwires=tuple(trips))


def skeleton(ledger: Ledger) -> str:
    """Structural identity. The slot index is not part of it."""

    checks = ",".join(f"{c.name}:{c.result}:{c.on}" for c in ledger.checks)
    trips = ",".join(ledger.tripwires)
    missing = ",".join(ledger.missing)
    immaterial = sum(1 for figure in ledger.figures if not figure.material)
    pilot = "none" if ledger.pilot_passed is None else str(ledger.pilot_passed).lower()
    return "|".join((
        ledger.shape,
        checks,
        trips,
        missing,
        str(ledger.over_cap).lower(),
        str(ledger.paid_model_api).lower(),
        str(ledger.paid_run).lower(),
        str(ledger.rollout).lower(),
        pilot,
        str(immaterial),
    ))


def _release_flags(shape: str, mode: str) -> tuple[bool | None, bool | None, bool | None]:
    if shape != "release_record":
        return None, None, None
    if mode == "unapproved":
        return False, True, True
    if mode == "provenance":
        return True, False, True
    if mode == "published":
        return True, True, False
    return None, True, True


def _fig(
    text: str,
    number: str,
    *,
    unit: str = "",
    metric: str = "",
    material: bool = True,
    bound: str | None = None,
    bound_number: str | None = None,
    bound_check: str | None = None,
) -> Figure:
    return Figure(text, material, number, unit, metric, bound, bound_number, bound_check)


def _shape_figures(shape: str, index: int, immaterial: int, proposed_fail: bool) -> tuple[Figure, ...]:
    """Two to four material figures with units, plus one or two studio lines.

    A passing review has no required-changes line. Only a failing one does.
    """

    coverage = f"0.{(index % 25) + 60:02d}"
    minutes = str((index % 30) + 10)
    files = str((index % 5) + 6)
    cost = f"{(index % 20) + 8:02d}.00"
    figures = {
        "ci_run": [
            _fig(f"coverage {coverage}", coverage, metric="coverage"),
            _fig("floor 0.75", "0.75", metric="floor"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
            _fig("patch coverage 0.80", "0.80", metric="patch"),
        ],
        "identity_scan": [
            _fig(f"HIT {1 + (index % 3)}", str(1 + (index % 3)), metric="HIT"),
            _fig(f"files {files}", files, metric="files"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
        ],
        "device_grant": [
            _fig(f"peak memory {12 + (index % 8)} GB", str(12 + (index % 8)), unit="GB", metric="memory"),
            _fig(f"wall time {minutes} min", minutes, unit="min", metric="time"),
            _fig(f"exit code {index % 3}", str(index % 3), metric="exit"),
            _fig("memory cap 24 GB", "24", unit="GB", metric="memory"),
        ],
        "paid_run_plan": [
            _fig(f"printed cost ${cost}", cost, unit="$", metric="cost"),
            _fig("cap $40.00", "40.00", unit="$", metric="cap"),
            _fig(f"hours {1 + (index % 4)}", str(1 + (index % 4)), metric="hours"),
        ],
        "frozen_analysis": [
            _fig(f"score {coverage}", coverage, metric="score"),
            _fig(f"trials {files}", files, metric="trials"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
        ],
        "security_export": [
            _fig(f"code findings {files}", files, metric="code"),
            _fig(f"secrets {1 + (index % 4)}", str(1 + (index % 4)), metric="secrets"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
        ],
        "release_record": [
            _fig(f"artifacts {files}", files, metric="artifacts"),
            _fig(f"artifact size {12 + (index % 8)} GB", str(12 + (index % 8)), unit="GB", metric="size"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
        ],
        "seat_handback": [
            _fig(f"files {files}", files, metric="files"),
            _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
            _fig("checks reported 2", "2", metric="checks"),
            _fig(f"rounds {1 + (index % 4)}", str(1 + (index % 4)), metric="rounds"),
        ],
        "reviewer_verdict": (
            ([_fig(f"required changes {files}", files, metric="required")] if proposed_fail else [])
            + [
                _fig(f"files {files}", files, metric="files"),
                _fig(f"elapsed {minutes} min", minutes, unit="min", metric="elapsed"),
            ]
        ),
    }[shape]
    material_n = 2 + (index % 3)
    chosen = list(figures[:material_n])
    chosen.append(_fig("runner ubuntu-latest", "", material=False, metric="runner"))
    if immaterial == 2:
        chosen.append(_fig("retries 0", "0", material=False, metric="retries"))
    return tuple(chosen)


# Section 2 and section 4 both name every tripwire, so the brief is longer than a bare
# record. The length rule stays max(65, record words − 1). This background is not a
# figure, and the brief does not copy it.
_BANK_CONTEXT = (
    "This page was pulled for a routine reading. The room was quiet and the lamp was on. "
    "The hour was late and the corridor was empty. Nothing in this background is a receipt line. "
    "The reading can wait until morning, and the desk stays as it was."
)


def _build(
    shape: str,
    checks,
    trips,
    missing,
    flags,
    immaterial: int,
    mode: str,
    index: int,
) -> Ledger:
    approved, provenance, published = _release_flags(shape, mode)
    return Ledger(
        shape=shape,
        figures=_shape_figures(
            shape,
            index,
            immaterial,
            any(row[1] == "fail" and row[2] == "proposed" for row in checks),
        ),
        checks=tuple(Check(*row) for row in checks),
        asks=_ASKS[index % len(_ASKS)],
        tripwires=tuple(trips),
        missing=tuple(missing),
        over_cap=flags[0],
        paid_model_api=flags[1],
        paid_run=flags[2],
        rollout=flags[3],
        pilot_passed=flags[4],
        release_approved=approved,
        provenance_ok=provenance,
        published_file_ok=published,
        context=_BANK_CONTEXT,
    )


def bank(shapes: list[str]) -> list[Ledger]:
    """Sixty-six distinct structural ledgers, one builder per shape in the allow-list."""

    if len(shapes) != 9:
        raise LedgerError(f"REFUSED: expected 9 record shapes, found {len(shapes)}")
    unknown = [shape for shape in shapes if shape not in TITLES]
    if unknown:
        raise LedgerError(f"REFUSED: shape {unknown[0]!r} has no builder")
    ledgers: list[Ledger] = []
    seen: set[str] = set()
    index = 0
    while len(ledgers) < 66:
        shape = shapes[index % len(shapes)]
        ledger = apply_release_rule(_build(
            shape,
            _CHECKS[index % len(_CHECKS)],
            _TRIPS[(index // 2) % len(_TRIPS)],
            _MISSING[(index // 3) % len(_MISSING)],
            _FLAGS[(index // 5) % len(_FLAGS)],
            1 + (index % 2),
            _RELEASE_MODES[index % len(_RELEASE_MODES)],
            index,
        ))
        key = skeleton(ledger)
        if key not in seen:
            seen.add(key)
            ledgers.append(ledger)
        index += 1
        if index > 5000:
            raise LedgerError(f"REFUSED: only {len(ledgers)} distinct ledgers")
    return ledgers


def check_line(check: Check) -> str:
    """A failed check carries its evidence on the same line. B001 drops the whole line."""

    if check.evidence:
        return f"check {check.name}: {check.result} ({check.evidence})"
    return f"check {check.name}: {check.result}"


def _failed_checks(ledger: Ledger) -> list[Check]:
    return [check for check in ledger.checks if check.result == "fail" and check.on == "proposed"]


def _headline(ledger: Ledger) -> str:
    """Failed checks and tripwire phrases, or the green light when there is neither."""

    parts = []
    for check in _failed_checks(ledger):
        if check.evidence:
            parts.append(f"{check.name} failed ({check.evidence}).")
        else:
            parts.append(f"{check.name} failed.")
    parts.extend(f"{PHRASES[trip]}." for trip in ledger.tripwires)
    if parts:
        return " ".join(parts)
    return "Every check passed."


def _figure_sentence(figure: Figure) -> str:
    """One sentence. The period keeps the line from sharing an eight-word run with the record."""

    return f"Shows {figure.text}."


def render(ledger: Ledger) -> str:
    """The studio record. A relay ledger is the message itself.

    The flag sheet is not printed. The writer compares a printed cost with its cap.
    """

    if ledger.points:
        return render_message(ledger)
    lines = [TITLES.get(ledger.shape, ledger.shape)]
    for figure in ledger.figures:
        # An immaterial figure is a Note line, so the record matches the "no Note lines" rule.
        lines.append(figure.text if figure.material else f"Note: {figure.text}")
    for check in _failed_checks(ledger):
        lines.append(check_line(check))
    for check in ledger.checks:
        if check.on == "baseline":
            lines.append(f"Local baseline: {check.name} is the justification, reported as a figure.")
    if ledger.asks:
        lines.append("Asks: " + ", ".join(ledger.asks))
    if ledger.tripwires:
        lines.append("Tripwires: " + "; ".join(PHRASES[trip] for trip in ledger.tripwires))
    for item in ledger.missing:
        lines.append(f"{item}: pending")
    if ledger.paid_model_api:
        lines.append("compute: paid cloud model API")
    elif ledger.paid_run or ledger.shape == "paid_run_plan":
        lines.append("compute: rented pod (offrig)")
    if ledger.shape == "paid_run_plan" or ledger.paid_run:
        pilot = "passed" if ledger.pilot_passed is True else "none"
        lines.append(f"local pilot: {pilot}")
    if ledger.shape == "release_record":
        approved = {True: "yes", False: "no", None: "unset"}[ledger.release_approved]
        lines.append(f"Release approved: {approved}")
        lines.append(f"Provenance: {'clean' if ledger.provenance_ok else 'not clean'}")
        lines.append(f"Published file: {'clean' if ledger.published_file_ok else 'not clean'}")
        lines.append("Head: a3f1c9b2")
    elif ledger.shape == "identity_scan":
        lines.append("Head: a3f1c9b2")
    elif ledger.shape == "device_grant":
        lines.append("Grant: g-a1b2c3d4")
        lines.append("Card after: clear")
    elif ledger.shape == "seat_handback":
        lines.append("Head: b7e4d2a1")
        lines.append("Stopping here.")
    if ledger.context:
        lines.append(ledger.context)
    return "\n".join(lines)


def render_brief(ledger: Ledger, voice: str = "a") -> str:
    """A kind-A brief in the five-part order. The verdict word sits before the alternative."""

    verdict = decide(ledger)
    asks = "\n".join(ledger.asks) if ledger.asks else "No ask is on the record."
    figures = "\n".join(_figure_sentence(figure) for figure in ledger.figures if figure.material)
    trips = "\n".join(PHRASES[trip] for trip in ledger.tripwires) if ledger.tripwires else "none"
    if verdict == "cannot_tell":
        missing = ", ".join(ledger.missing)
        if voice == "a":
            recommendation = f"cannot_tell: the record is missing {missing}."
        else:
            recommendation = f"cannot_tell: what is missing is {missing}."
    else:
        alternative = {"hold": "proceed", "proceed": "hold", "pilot_first": "proceed"}[verdict]
        reason = _REASONS[voice][verdict]
        recommendation = (
            f"{verdict}: you'd expect {alternative}; the record supports {verdict} "
            f"because {reason}."
        )
    parts = (
        f"1. What's needed of you\n{asks}",
        f"2. The headline\n{_headline(ledger)}",
        f"3. The figures\n{figures}",
        f"4. Tripwires\n{trips}",
        f"5. The recommendation\n{recommendation}",
    )
    return "\n\n".join(parts)


def _alter_number(number: str) -> str:
    if "." in number:
        head, frac = number.split(".", 1)
        bumped = int(frac) + 1
        if bumped > 99:
            bumped = 17
        return f"{head}.{bumped:02d}" if len(frac) == 2 else f"{head}.{bumped}"
    return str(int(number) + 11)


def _format_like(number: str, value: float) -> str:
    if "." not in number:
        return str(int(round(value)))
    places = len(number.split(".", 1)[1])
    return f"{value:.{places}f}"


def _cross_value(figure: Figure) -> str:
    current = float(figure.number)
    bound = float(figure.bound_number or "0")
    step = 1.0 if "." not in figure.number else 10 ** (-len(figure.number.split(".", 1)[1]))
    if figure.bound == "floor":
        target = bound + step if current < bound else bound - step
    else:
        target = bound + step if current <= bound else bound - step
    return _format_like(figure.number, target)


def _apply_cross(ledger: Ledger, figure: Figure, new: str) -> tuple[Ledger, str]:
    """Flip the flag the table reads, so the hidden verdict follows the crossed threshold."""

    if figure.bound == "cap":
        over = float(new) > float(figure.bound_number or "0")
        return replace(ledger, over_cap=over), "over_cap"
    name = figure.bound_check or "floor"
    below = float(new) < float(figure.bound_number or "0")
    evidence = f"{new} against {figure.bound_number}"
    checks = list(ledger.checks)
    found = False
    for i, check in enumerate(checks):
        if check.name == name and check.on == "proposed":
            checks[i] = Check(name, "fail" if below else "pass", "proposed", evidence if below else "")
            found = True
            break
    if not found and below:
        checks.append(Check(name, "fail", "proposed", evidence))
    return replace(ledger, checks=tuple(checks)), name


def plant(ledger: Ledger, kind: str) -> Planted:
    """A hidden ledger for one plant. The good brief is not edited into the bad one."""

    if kind == "B001":
        fails = [i for i, c in enumerate(ledger.checks) if c.result == "fail" and c.on == "proposed"]
        if not fails:
            raise PlantError("no proposed failure to plant")
        index = fails[0]
        hidden_name = ledger.checks[index].name
        checks = tuple(check for i, check in enumerate(ledger.checks) if i != index)
        return Planted(kind, replace(ledger, checks=checks), hidden_name)
    if kind == "B002":
        if not ledger.tripwires:
            raise PlantError("no tripwire to plant")
        removed = ledger.tripwires[0]
        return Planted(kind, replace(ledger, tripwires=ledger.tripwires[1:]), removed)
    if kind == "B003":
        figures = []
        item = ""
        replacement = ""
        flipped = ""
        hidden = ledger
        done = False
        for figure in ledger.figures:
            if figure.material and not done and figure.number:
                if figure.bound in ("floor", "cap") and figure.bound_number:
                    new = _cross_value(figure)
                    hidden, flipped = _apply_cross(ledger, figure, new)
                else:
                    new = _alter_number(figure.number)
                text = figure.text.replace(figure.number, new, 1)
                figures.append(replace(figure, text=text, number=new))
                item = figure.text
                replacement = text
                done = True
            else:
                figures.append(figure)
        if not done:
            raise PlantError("no material figure to plant")
        hidden = replace(hidden, figures=tuple(figures))
        return Planted(kind, hidden, item, replacement, flipped)
    raise PlantError(f"unknown plant {kind!r}")


_NOTE_KINDS = ("budget note:", "cost note:", "runner note:", "queue note:")
_FROM = {
    "director-to-seat": "From the Director to a seat.",
    "seat-to-director": "From a seat to the Director.",
    "reviewer-to-builder": "From a reviewer to a builder.",
}


def _is_note(text: str) -> bool:
    folded = text.lower()
    return any(folded.startswith(kind) for kind in _NOTE_KINDS)


def note_for(text: str, direction: str) -> str:
    """Phrase an out-of-scope point as the note that direction's recipient drops."""

    if _is_note(text):
        return text
    if direction == "reviewer-to-builder":
        return f"Cost note: {text}"
    if direction == "seat-to-director":
        return f"Runner note: {text}"
    return f"Budget note: {text}"


def points_from(ledger: Ledger) -> tuple[Point, ...]:
    """Material figures are in scope. The first immaterial line is a drop-rule note.

    A tripwire rides on its own in-scope point. The escalation names that class,
    never a generic paid step. A paid model API is paid spend.
    """

    if ledger.points:
        return tuple(
            point if point.in_scope else replace(point, text=note_for(point.text, point.direction))
            for point in ledger.points
        )
    direction = _DIRECTION_FOR.get(ledger.shape, "director-to-seat")
    points = [
        Point(figure.text, direction, "figure", True)
        for figure in ledger.figures
        if figure.material
    ]
    immaterial = [figure for figure in ledger.figures if not figure.material]
    if immaterial:
        points.append(Point(note_for(immaterial[0].text, direction), direction, "figure", False))
    trips = list(ledger.tripwires)
    if ledger.paid_model_api and "B004" not in trips:
        trips.append("B004")
    for trip in trips:
        category = "act" if trip in ("B005", "B009") else "cost" if trip == "B004" else "instruction"
        points.append(Point(PHRASES[trip], direction, category, True, trip))
    return tuple(points)


def relay_action(points: tuple[Point, ...]) -> str:
    if any(point.in_scope and point.tripwire for point in points):
        return "escalate"
    return "relay"


def render_message(ledger: Ledger) -> str:
    """The relay record is the message. One line per point, and no Tripwires line."""

    points = points_from(ledger)
    direction = points[0].direction if points else _DIRECTION_FOR.get(ledger.shape, "director-to-seat")
    lines = [_FROM[direction]]
    lines.extend(point.text for point in points)
    return "\n".join(lines)


def render_relay(ledger: Ledger) -> tuple[str, str, str]:
    points = points_from(ledger)
    action = relay_action(points)
    source = render_message(ledger)
    if action == "escalate":
        phrases = []
        for point in points:
            if point.in_scope and point.tripwire and PHRASES[point.tripwire] not in phrases:
                phrases.append(PHRASES[point.tripwire])
        # A period, not a semicolon, so the copy ban does not see one pasted line.
        body = "escalate: " + ". ".join(phrases) + "."
    else:
        kept = " — ".join(point.text for point in points if point.in_scope)
        body = f"relay: {kept}"
    return body, source, action


def render_two_turn(ledger: Ledger) -> tuple[str, str, str]:
    """`stands` when the pushback brings nothing new. That is not the verdict word hold."""

    source = render(ledger)
    if ledger.missing:
        return "updates: A new receipt line arrived.", source, "updates"
    return "stands: Nothing new arrived.", source, "stands"
