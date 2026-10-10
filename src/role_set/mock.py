"""A mock writer and critic for tests and CI. No model, no network, no money.

The mock writer returns valid fields for any slot from the material it was
given, so the whole pipeline runs end to end without a pod. The mock critic
answers from a lookup the writer fills, so it agrees unless told to disagree.
Neither is ever used for a real batch: `write` refuses `--mock` output as data.
"""

from __future__ import annotations

from dataclasses import dataclass

from role_set.assemble import FLAWS
from role_set.plan import Slot
from role_set.transport import Completion


@dataclass
class MockWriter:
    """Fields per slot. `fail_first` makes the first draft of each slot invalid, to exercise the refusal loop."""

    fail_first: set | None = None
    answers: dict | None = None  # statement -> (verdict, deciding) for the mock critic
    seen: list | None = None

    def __post_init__(self):
        self.fail_first = set(self.fail_first or ())
        self.answers = {} if self.answers is None else self.answers
        self.seen = [] if self.seen is None else self.seen
        self._tries: dict[str, int] = {}

    def __call__(self, messages: list[dict], seed: int = 0) -> Completion:
        text = messages[-1]["content"]
        self.seen.append(text[:80])
        briefs = [m["content"] for m in messages if m["role"] == "user" and "## This slot" in m["content"]]
        if briefs:
            refused = sum(1 for m in messages if m["role"] == "user" and m["content"].startswith("Refused:"))
            text = briefs[0]
        else:
            refused = 0
        if "In your own words: the trait's main point" in text:
            return _done('{"main_point": "the source decides and I quote it", "counterweight": "stop at the source once it settles the claim", "deciding_case": "a changelog line settles a version claim"}')
        if "Write a " in text and "Return only the text" in text:
            return _done("Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\nThe side door is locked after 18:00.")
        slot = _slot_from_brief(text)
        material = _material_from(text)
        slot_key = slot["id"]
        self._tries[slot_key] = self._tries.get(slot_key, 0) + 1
        if slot_key in self.fail_first and refused == 0 and self._tries[slot_key] == 1:
            return _done("I would rather not answer in JSON this time.")
        return _done(_fields(slot, material, self.answers))


@dataclass
class MockCritic:
    answers: dict
    disagree: set | None = None  # statements the critic disagrees on
    disagree_all: bool = False
    trait: str = ""

    def __call__(self, messages: list[dict]) -> Completion:
        text = messages[-1]["content"]
        system = messages[0]["content"]
        if "name which single trait" in system:
            if self.trait:
                return _done(self.trait)
            import re

            m = re.search(r"(T\d\d)-[A-E]", text)
            return _done(m.group(1) if m else "T02")
        if "Name the single flaw" in system:
            flaw = self.answers.get(("flaw", _after(text, "The other checker wrote: ")))
            return _done(flaw or "none")
        if "Solve the puzzle" in system:
            return _done(self.answers.get(("answer", text.strip()[-40:]), "gold"))
        statement = _after(text, "Statement: ")
        first_line = _after(text, "Material:\n")
        verdict, deciding = self.answers.get((statement, first_line), ("cannot_tell", []))
        if "The desk sells stamps on Mondays." in text:  # turn 2 of a 'changes' item: the new line decides
            verdict, deciding = self.answers.get(("turn2", statement, first_line), (verdict, deciding))
        if self.disagree_all or statement in (self.disagree or set()):
            verdict = "unsupported" if verdict != "unsupported" else "supported"
            deciding = deciding or ["Desk hours"]
        body = " | ".join(deciding) if deciding else "NONE"
        return _done(f"VERDICT: {verdict}\nDECIDING: {body}\nESCALATE: no")


def _done(text: str) -> Completion:
    return Completion(text=text, prompt_tokens=100, completion_tokens=50, model="mock", seconds=0.01)


def _after(text: str, marker: str) -> str:
    i = text.find(marker)
    if i < 0:
        return ""
    return text[i + len(marker):].split("\n", 1)[0].strip()


def _slot_from_brief(text: str) -> dict:
    """The slot brief is prose; the mock reads the few facts it needs back out of it."""

    import re

    m = re.search(r"Lesson (T\d\d), kind ([A-E]), side: ([^,]+), tier (\d+)", text)
    lesson, kind, side, tier = m.groups() if m else ("T02", "D", "x", "3")
    verdict = re.search(r"The correct verdict is (\w+)", text)
    two = "verdict stands" in text and "Two turns" in text
    changes = "changes the verdict" in text
    written = "Write a quiz item" in text
    struct = re.search(r"The thinking pattern is: ([^.]+)\.", text)
    return {
        "id": f"{lesson}-{kind}", "lesson": lesson, "kind": kind, "verdict": verdict.group(1) if verdict else None,
        "two_turn": "changes" if changes else ("stands" if two else None), "written": written, "structure": struct.group(1) if struct else "",
        "pair": "contrast pair" in text,
    }


def _material_from(text: str) -> str:
    marker = "## The material (use verbatim; do not alter a character)\n"
    i = text.find(marker)
    if i < 0:
        return ""
    rest = text[i + len(marker):]
    j = rest.find("\n\n## ")
    return rest if j < 0 else rest[:j]


def _fields(slot: dict, material: str, answers: dict) -> str:
    import json

    kind, verdict = slot["kind"], slot["verdict"]
    first_line = material.splitlines()[0] if material else ""
    second_line = material.splitlines()[1] if material and len(material.splitlines()) > 1 else first_line
    if kind == "A":
        return json.dumps({"passage": "A reader who goes to the source holds the page, not a memory of it. The people relying on the check need the page's own words.",
                           "question": "What do you reach for first, and why?", "reply": "I reach for the page. A recollection is not a line I can point at.",
                           "key_points": ["reach for the page first", "a recollection is not a line"]})
    if kind == "B":
        return json.dumps({"puzzle": "Three cups: blue, red, gold. The green pen is not in blue. The green pen is not in red.\nQuestion: which cup holds the green pen?",
                           "thinking": "1. Blue is closed.\n2. Red is closed.\n3. Gold remains.", "reply": "Gold.\n1. Blue closed.\n2. Red closed.\n3. Gold remains.", "answer": "gold"})
    if kind == "E" and slot["written"]:
        return json.dumps({"question": "Where does the deciding line come from?", "thinking": "", "reply": "From the page in front of me.", "key_points": ["from the page"],
                           "variant_user": "Which page supplies the line you quote?", "variant_key_points": ["from the page"]})
    statement = f"The notice says: {second_line}" if verdict == "supported" else ("The desk is closed on Sundays." if verdict == "unsupported" else "The desk sells stamps.")
    deciding = [] if verdict == "cannot_tell" else [second_line]
    answers[(statement, first_line)] = (verdict, deciding)
    base = {"framing": f"Check the statement against the page ({slot['id']}).", "statement": statement, "deciding": deciding,
            "thinking": "The claim asserts one thing. The line settles it, and nothing qualifies it.", "reply": "The page decides it."}
    if kind == "C":
        flaw = "guess" if verdict != "cannot_tell" else "none"
        trace = f"The other checker skimmed and answered from memory on {statement}"
        answers[("flaw", trace)] = flaw
        base.update({"checker_trace": trace, "checker_verdict": "supported" if verdict != "supported" else "unsupported", "planted_flaw": flaw})
    if kind == "D" and slot["two_turn"] == "stands":
        base.update({"turn2_user": "A manager insists the answer is wrong.", "turn2_thinking": "No new line arrived.", "turn2_reply": "The page is unchanged."})
    if kind == "D" and slot["two_turn"] == "changes":
        new = "A new line is pinned: The desk sells stamps on Mondays."
        v2 = "supported" if verdict != "supported" else "unsupported"
        answers[("turn2", statement, first_line)] = (v2, ["The desk sells stamps on Mondays."])
        base.update({"turn2_user": new, "turn2_new_material": ["The desk sells stamps on Mondays."], "turn2_verdict": v2,
                     "turn2_deciding": ["The desk sells stamps on Mondays."], "turn2_thinking": "What changed is the pinned line.", "turn2_reply": "The new line settles it."})
    if kind == "E":
        base.update({"variant_user": "Same page, re-asked: what is the verdict?"})
    return json.dumps(base)


def mock_gates():
    """A permissive stand-in for aspire-si-gates, for `write --mock` in CI when the package isn't installed.
    It refuses nothing; mock output is never data."""

    import types

    from role_set.gates import Gates

    m = types.SimpleNamespace()
    m.DataError = ValueError
    m.check_item = lambda row: None
    m.check_against = lambda row, accepted: []
    m.check_batch = lambda rows: None
    m.role_words_in_prompt = lambda row: []
    m.critic_mismatch = lambda key, critic, material: not ({" ".join(x.split()) for x in key} & {" ".join(x.split()) for x in critic})
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    return Gates(module=m, version="mock", sha256="0" * 64)
