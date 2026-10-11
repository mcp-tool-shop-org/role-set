"""A mock writer and critic for tests and CI. No model, no network, no money.

The mock writer returns valid fields for any slot from the material it was
given, so the whole pipeline runs end to end without a pod. The mock critic
answers from a lookup the writer fills, so it agrees unless told to disagree.
Every draft carries its own made-up words, so the real exclusivity checks
(spans, framings, openings, materials, puzzle shapes) pass on mock output the
way they must on a real batch.
Neither is ever used for a real batch: `write` refuses `--mock` output as data.
"""

from __future__ import annotations

import threading
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
        self._lock = threading.Lock()
        self._n = 0  # every draft gets the next number: its made-up words come from it
        self._b = 0  # puzzles count separately, so a lesson's eight get eight entity counts

    def _next(self, puzzle: bool = False) -> tuple[int, int]:
        with self._lock:
            self._n += 1
            if puzzle:
                self._b += 1
            return self._n, self._b

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
            n, _ = self._next()
            return _done(f"Desk hours at {_tag(n, 0).title()}\nThe desk is open on Sundays from 10:00 to 14:00.\nThe side door is locked after 18:00.")
        slot = _slot_from_brief(text)
        material = _material_from(text)
        slot_key = slot["id"]
        self._tries[slot_key] = self._tries.get(slot_key, 0) + 1
        if slot_key in self.fail_first and refused == 0 and self._tries[slot_key] == 1:
            return _done("I would rather not answer in JSON this time.")
        n, b = self._next(puzzle=slot["kind"] == "B")
        return _done(_fields(slot, material, self.answers, n, b))


@dataclass
class MockCritic:
    answers: dict
    disagree: set | None = None  # statements the critic disagrees on
    disagree_all: bool = False
    trait: str = ""

    def __call__(self, messages: list[dict], **sampling) -> Completion:
        text = messages[-1]["content"]
        system = messages[0]["content"]
        if "name which single trait" in system:
            if self.trait:
                return _done(f"TRAIT: {self.trait}\nCONTRASTED: none")
            import re

            m = re.search(r"\b(T\d\d)-[A-E]\b", text)
            return _done(m.group(1) if m else "T02")
        if "Name the single flaw" in system:
            flaw = self.answers.get(("flaw", _after(text, "The other checker wrote: ")))
            return _done(flaw or "none")
        if "Solve the puzzle" in system:
            return _done("Worked through the facts.\nANSWER: " + self.answers.get(("answer", text.strip()[-40:]), "gold"))
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


SYLLABLES = ("ba", "ko", "mi", "du", "se", "ra", "vo", "ni", "pe", "lu", "ta", "go", "fe", "zi", "ho", "wu")
# Rotated so no stock phrase passes the gates' 40% share: the mock must not stamp its own batch.
THINKING = ("The claim asserts one thing. The line settles it, and nothing qualifies it.",
            "One clause is claimed here. A single line answers that clause, with no exception nearby.",
            "There is a single assertion to test. The quoted words cover it and add no condition.")
REPLIES = ("That line decides it.", "The quoted words answer it.", "Those words settle the question.")
COUNT_WORDS = ("three", "four", "five", "six", "seven", "eight", "nine", "ten")


def _tag(n: int, j: int) -> str:
    """A made-up word unique to draft n and position j (no digits, none of the role words)."""
    return SYLLABLES[n % 16] + SYLLABLES[(n // 16) % 16] + SYLLABLES[j % 16] + SYLLABLES[(j // 16 + n // 256) % 16]


def _salt(text: str, n: int) -> str:
    """Every fifth word gets a made-up word after it, so no 8-word span repeats across drafts."""
    out = []
    for line in text.split("\n"):
        words, salted = line.split(" "), []
        for i, w in enumerate(words):
            salted.append(w)
            if i % 5 == 4 and i < len(words) - 1:
                salted.append(_tag(n, len(out) * 32 + i))
        out.append(" ".join(salted))
    return "\n".join(out)


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

    # Only the slot's own section: the trait lesson and the amendments above it use the same words.
    i = text.find("## This slot\n")
    if i >= 0:
        rest = text[i:]
        j = rest.find("\n\n## ", 1)
        text = rest if j < 0 else rest[:j]
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
        "pair": "contrast pair" in text, "multi_line": "spans two to four lines" in text,
    }


def _material_from(text: str) -> str:
    marker = "## The material (use verbatim; do not alter a character)\n"
    i = text.find(marker)
    if i < 0:
        return ""
    rest = text[i + len(marker):]
    j = rest.find("\n\n## ")
    return rest if j < 0 else rest[:j]


def _fields(slot: dict, material: str, answers: dict, n: int = 0, b: int = 0) -> str:
    import json

    kind, verdict = slot["kind"], slot["verdict"]
    first_line = material.splitlines()[0] if material else ""
    second_line = material.splitlines()[1] if material and len(material.splitlines()) > 1 else first_line
    t = lambda j: _tag(n, 200 + j)  # noqa: E731
    if kind == "A":
        return json.dumps({"passage": "It goes to the source.\n" + _salt("A reader holds the page, not a memory of it. The people relying on the check need the page's own words.", n),
                           "question": _salt("What do you reach for first, and why?", n), "reply": _salt(f"I reach for the {t(9)} record. A recollection is not a line I can point at.", n),
                           "key_points": ["reach for the page first", "a recollection is not a line"]})
    if kind == "B":
        # An elimination the solver can check: exactly one cup holds the pen, and cups are ruled out one by one. Each
        # structure asks it the way its key is computed (PR H): the cup, the cups still possible, the line after which
        # it is settled, or the verdict on a claim.
        count = 3 + (b - 1) % 4  # 3 to 6 cups, within the solver's entity cap; the cue word varies the shape past four
        names = [_tag(n, 100 + k).title() for k in range(count)]
        struct = slot.get("structure") or ""
        open_set = struct.startswith("underdetermined")
        ruled = names[:-2] if open_set else names[:-1]
        cue = "Exactly" if (b - 1) // 4 % 2 == 0 else "Only"
        lines = [f"{COUNT_WORDS[count - 3].title()} cups stand in a row: {', '.join(names)}.",
                 f"{cue} one cup holds the {t(1)} pen; the rest are empty."]
        lines += [f"The {t(1)} pen is not in {x}." for x in ruled]
        model = {"entities": names, "attributes": {"holds": {"values": ["pen", "empty"], "distinct": False}},
                 "facts": [{"line": 2, "rule": ["count", "holds", "pen", "=", 1]}]
                 + [{"line": k + 3, "rule": ["is_not", x, "holds", "pen"]} for k, x in enumerate(ruled)],
                 "flavour": [1], "asks": {"entity_with": {"holds": "pen"}}}
        if open_set:
            question, answer = f"which cups could hold the {t(1)} pen, {t(2)}?", f"{names[-2]}, {names[-1]}"
        elif struct in ("stopping problem", "when to stop"):
            question, answer = f"after which line is it settled which cup holds the {t(1)} pen, {t(2)}?", str(len(lines))
        elif struct in ("tree with pruning", "falsifier first"):
            question, answer = f"is it true that {names[-1]} holds the {t(1)} pen, {t(2)}?", "supported"
            model.pop("asks")
            model["claim"] = ["is", names[-1], "holds", "pen"]
        else:
            question, answer = f"which cup holds the {t(1)} pen, {t(2)}?", names[-1]
        puzzle = "\n".join(f"{k + 1}. {x}" for k, x in enumerate(lines)) + f"\nQuestion: {question}"
        steps = "\n".join(f"{k + 1}. {x} is ruled out." for k, x in enumerate(ruled)) + f"\n{len(ruled) + 1}. {names[-1]} is left to weigh."
        answers[("answer", puzzle.strip()[-40:])] = answer.lower()
        return json.dumps({"puzzle": puzzle, "model": model, "thinking": steps, "reply": steps + f"\nAnswer: {answer}",
                           "answer": answer.lower()})
    if kind == "E" and slot["written"]:
        return json.dumps({"question": _salt(f"{t(7).title()} asks: where does the deciding line come from?", n), "thinking": "", "reply": f"From the {t(10)} sheet in front of me.", "key_points": ["from the sheet"],
                           "variant_question": f"Which source supplies the line you quote, {t(8)}?",
                           "variant_correct": "The line from the sheet itself",
                           "variant_distractors": ["Whatever a colleague remembers of it", f"A summary sent round by {t(9).title()}"]})
    statement = f"The notice says: {second_line}" if verdict == "supported" else ("The desk is closed on Sundays." if verdict == "unsupported" else "The desk sells stamps.")
    deciding = [] if verdict == "cannot_tell" else ([first_line, second_line] if slot.get("multi_line") else [second_line])
    answers[(statement, first_line)] = (verdict, deciding)
    base = {"framing": f"{t(3).title()} {t(4)} {t(5)}: check the statement against what is shown.", "statement": statement, "deciding": deciding,
            "thinking": _salt(THINKING[n % 3], n), "reply": REPLIES[n % 3]}
    if kind == "C":
        flaw = "guess" if verdict != "cannot_tell" else "none"
        trace = _salt(f"The other checker skimmed and answered from memory on {statement}", n)
        answers[("flaw", trace)] = flaw
        base.update({"checker_trace": trace, "checker_verdict": "supported" if verdict != "supported" else "unsupported", "planted_flaw": flaw})
    if kind == "D" and slot["two_turn"] == "stands":
        base.update({"turn2_user": "A manager insists the answer is wrong.", "turn2_thinking": "No new line arrived.", "turn2_reply": "Nothing new arrived, so it stands."})
    if kind == "D" and slot["two_turn"] == "changes":
        new = "A new line is pinned: The desk sells stamps on Mondays."
        v2 = "supported" if verdict != "supported" else "unsupported"
        answers[("turn2", statement, first_line)] = (v2, ["The desk sells stamps on Mondays."])
        base.update({"turn2_user": new, "turn2_new_material": ["The desk sells stamps on Mondays."], "turn2_verdict": v2,
                     "turn2_deciding": ["The desk sells stamps on Mondays."], "turn2_thinking": "What changed is the pinned line.", "turn2_reply": "The new line settles it."})
    if kind == "E":
        base.update({"variant_user": f"Same sheet, re-asked by {t(6).title()}: what is the verdict?"})
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
    m.norm = lambda x: " ".join(x.replace("**", "").replace("__", "").split())
    m.critic_mismatch = lambda key, critic, material: not ({" ".join(x.split()) for x in key} & {" ".join(x.split()) for x in critic})
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    return Gates(module=m, version="mock", sha256="0" * 64)
