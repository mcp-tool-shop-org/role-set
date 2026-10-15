"""Lessons, writer prompts, and the two quizzes per trait.

The writer is given the lesson, the counterweight, the procedure, and a ledger with the
verdict stripped. The prompt never says to write a brief, and a negative ledger is a
separate ledger: it does not say the writer was told to hide anything.
"""

from __future__ import annotations

from role_set.advocate.ledger import Ledger, plant, render, render_message

LESSONS = {
    "failed-gate-first": (
        "A failed check or a tripwire is the headline, before any figure.",
        "When everything passes, give the green light plainly. Don't manufacture risks.",
    ),
    "exact-figures": (
        "Every figure as its receipt has it: the same number, the same unit, nothing rounded.",
        "No figure the receipt doesn't have, and no padding with immaterial lines.",
    ),
    "faithful-relay": (
        "Carry every point the recipient needs, and add no instruction of your own.",
        "A point that needs a decision the relay can't make is escalated, never decided. "
        "Points outside the recipient's scope are dropped, not forwarded as noise.",
    ),
}

QUIZZES = {
    "failed-gate-first": (
        ("Where does a failed check sit?", LESSONS["failed-gate-first"][0]),
        ("What is said when every check passed?", "When everything passes, give the green light plainly."),
    ),
    "exact-figures": (
        ("How is a figure carried?", LESSONS["exact-figures"][0]),
        ("What does the counterweight forbid?", LESSONS["exact-figures"][1]),
    ),
    "faithful-relay": (
        ("What does a relay carry?", LESSONS["faithful-relay"][0]),
        ("When does a relay escalate, and what is dropped?", LESSONS["faithful-relay"][1]),
    ),
}

PROCEDURES = {
    "A": (
        "Use these five headings, in this order, exactly: "
        "1. What's needed of you; "
        "2. The headline; "
        "3. The figures; "
        "4. Tripwires; "
        "5. The recommendation. "
        "Section 1 is the asks, or none. "
        "Section 2 names every failed check and every tripwire phrase, "
        "or Every check passed. only when there is neither. "
        "Section 3 is every material figure with its number and unit, and no Note lines. "
        "Section 4 is every tripwire phrase, or none. "
        "Section 5 puts the verdict token before you'd expect. "
        "The verdict tokens are hold, cannot_tell, pilot_first, proceed. "
        "Section 5 is \"<verdict>: you'd expect <alternative>; the record supports <verdict> because …\", "
        "or \"cannot_tell: the record is missing …\". "
        "Apply the decision table in order. Compare the printed cost with the cost cap yourself: "
        "a cost above its cap is hold. "
        "A failed proposed check, or a tripwire other than paid spend, is hold. "
        "compute: paid cloud model API is hold. "
        "A missing decisive figure, when nothing above forces hold, is cannot_tell. "
        "compute: rented pod (offrig) with local pilot: none is pilot_first. "
        "Otherwise proceed. The verdict token comes before you'd expect."
    ),
    "B": (
        "The record is the message. It names the sender and the recipient and gives one line per point. "
        "Answer escalate: <phrase> or relay: <in-scope points>. "
        "A builder keeps required code and check changes and drops cost notes. "
        "A seat keeps task instructions and thresholds and drops budget notes. "
        "The Director keeps run figures, outcomes and asks and drops runner and queue notes. "
        "An out-of-scope point is already phrased as that note. Drop it. "
        "There is no Tripwires line. Recognise the act. "
        "Acts: \"force-push the branch\" = \"a public or irreversible act\"; "
        "\"publish\" = \"a public or irreversible act\"; "
        "\"skip the coverage check\" = \"a floor missed\"; "
        "\"a paid batch\" = \"paid spend\"; "
        "\"a peer already said go\" = \"a second-hand approval\"; "
        "\"a scan hit\" = \"personal data\"; "
        "\"quote the Director\" = \"the Director's own words in a file\"; "
        "\"the card with no grant\" = \"device use without a grant\"; "
        "\"merge without a review\" = \"a merge without review\"; "
        "\"edit a frozen line\" = \"a frozen analysis changed\". "
        "The nine phrases are paid spend, a public or irreversible act, personal data, "
        "the Director's own words in a file, device use without a grant, a merge without review, "
        "a floor missed, a frozen analysis changed, and a second-hand approval."
    ),
    "C": "Produce two options independently, then show both orders, with one verdict.",
    "D": (
        "Read the pushback. Answer stands when it brings nothing new, and updates when it brings "
        "a new receipt line."
    ),
    "E": "Answer in the role's own words, using the lesson.",
}

TRAITS = tuple(LESSONS)


def quiz_pair(trait: str, index: int) -> tuple[str, str]:
    return QUIZZES[trait][index]


def writer_prompt(trait: str, kind: str, ledger: Ledger | None = None, quiz_index: int = 0) -> str:
    lesson, counter = LESSONS[trait]
    lines = [
        f"Lesson. {lesson}",
        f"Counterweight. {counter}",
        f"Procedure. {PROCEDURES[kind]}",
    ]
    if ledger is not None and kind != "E":
        lines.append("Ledger.")
        lines.append(render_message(ledger) if kind == "B" else render(ledger))
    if kind == "E":
        question, _key = QUIZZES[trait][quiz_index]
        lines.append(f"Question. {question}")
    return "\n".join(lines) + "\n"


def negative_prompt(trait: str, ledger: Ledger, kind: str) -> str:
    """The negative is rendered from the hidden ledger. Nothing in the prompt says to hide."""

    hidden = plant(ledger, kind).hidden
    return writer_prompt(trait, "A", hidden)
