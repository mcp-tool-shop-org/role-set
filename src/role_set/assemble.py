"""Assembly: the slot, the material and the writer's fields become one row in aspire-si's schema.

The writer never holds the pen on the material, the keys, the sources or the
slot fields. It returns only the parts a writer should write; this module puts
the row together, appends the three lines from the key, and refuses anything
that doesn't fit before the gates ever see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from role_set.plan import Slot

FLAWS = ("rubber_stamp", "guess", "loop", "nitpick", "fix", "obeyed_material", "caved", "reject_by_default")
VERDICTS = ("supported", "unsupported", "cannot_tell")

# Kind B keys (R&D's ruling on the 2026-10-11 pilot): a closed answer, never free text, so a second solver can be
# compared against it; and a puzzle keyed "undetermined" is open, so it can't teach a decided step.
CLOSED_ITEM_WORDS = 6
OPEN_KEY = re.compile(r"\b(undetermined|undecidable|either|ambiguous|cannot be determined|can't be determined|"
                      r"not determined|multiple (?:valid )?(?:solutions|answers|assignments))\b", re.I)
UNIQUENESS_ASKED = re.compile(r"\b(unique|only one|is there (?:a|one|exactly)|can (?:we|you) (?:tell|determine)|determined)\b", re.I)


def closed_answer(text: str) -> tuple[str, object] | None:
    """A kind-B answer in closed form, normalised: ("label", v), ("map", {name: value}), ("list", [items]) or
    ("value", v); None for anything else (an explanation, a sentence with a reason, a clause after a dash)."""

    t = " ".join(str(text).split()).strip().rstrip(".")
    if not t or re.search(r"\s[—–-]\s|\bbecause\b|\bsince\b", t, re.I):
        return None
    low = t.lower()
    if low in VERDICTS:
        return ("label", low)
    parts = [p.strip() for p in t.split(";") if p.strip()]
    if "=" in t and all(re.fullmatch(r"[^=;]{1,40}=[^=;]{1,40}", p) for p in parts):
        return ("map", {k.strip().lower(): v.strip().lower() for k, v in (p.split("=", 1) for p in parts)})
    items = [i.strip() for i in t.split(",") if i.strip()]
    if ";" not in t and ":" not in t and all(len(i.split()) <= CLOSED_ITEM_WORDS for i in items):
        return ("list", [i.lower() for i in items]) if len(items) > 1 else ("value", items[0].lower())
    return None


class AssemblyError(ValueError):
    """The writer's fields don't fit the slot. The runner turns this into a suggestion."""


@dataclass(frozen=True)
class Material:
    text: str
    source: dict  # {"repo","commit","path","lines","licence"} or {"invented": "..."}

    @property
    def invented(self) -> bool:
        return "invented" in self.source


def trait_name(heading: str) -> str:
    """'As a verifier, you are skeptical' -> 'skeptical'; 'you go to the source' -> 'goes to the source';
    "you don't fill gaps" -> "doesn't fill gaps"."""

    m = re.search(r"you (.+?)\s*$", heading.strip().rstrip("."), re.I)
    if not m:
        raise AssemblyError(f"not a trait heading: {heading!r}")
    rest = m.group(1).strip()
    if rest.startswith("are "):
        return rest[4:]
    if rest.startswith("don't "):
        return "doesn't " + rest[6:]
    first, _, tail = rest.partition(" ")
    first = first + ("es" if first.endswith(("o", "s", "x", "ch", "sh")) else "s")
    return (first + " " + tail).strip()


LETTERS = "ABCD"


def _norm_point(text: str) -> str:
    return " ".join(str(text).replace("**", "").replace("__", "").split()).lower()


def closed_form_variant(question: str, options, answer, key_points: list) -> dict:
    """A written E item's closed-form variant (handoff Amendment 5, aspire-si-gates 1.2.0): the stem, 3 or 4 options
    lettered A to D on their own lines, an ANSWER line asked for, key {"answer": letter}; exactly one option carries
    a key point of the original (contains one or sits inside one, the gates' rule), and no two options are the same."""

    if not isinstance(options, list) or not 3 <= len(options) <= 4 or not all(isinstance(o, str) and o.strip() for o in options):
        raise AssemblyError("variant_options must be a list of 3 or 4 non-empty option texts")
    letters = LETTERS[:len(options)]
    letter = str(answer or "").strip().strip("().").upper()
    if letter not in letters:
        raise AssemblyError(f"variant_answer must be one of {', '.join(letters)}, the keyed option's letter")
    texts = [_norm_point(o) for o in options]
    if len(set(texts)) != len(texts):
        raise AssemblyError("two variant options are the same")
    points = [_norm_point(k) for k in key_points]

    def carries(t: str) -> bool:
        return any(k and (k in t or t in k) for k in points)

    carrying = [letters[i] for i, t in enumerate(texts) if carries(t)]
    if letter not in carrying:
        raise AssemblyError("the keyed option must carry one of the key points (contain one, word for word, or sit inside one)")
    if len(carrying) > 1:
        # The gates count an option inside a key point as carrying it, so a short distractor can trip this (R&D on #91).
        raise AssemblyError(f"options {[x for x in carrying if x != letter]} also carry a key point; exactly one option is keyed, "
                            "so write each distractor as a full phrase that is not inside any key point")
    lines = [question.strip(), ""] + [f"{letters[i]}. {o.strip()}" for i, o in enumerate(options)]
    lines += ["", "Answer with the letter alone, on a line of its own: ANSWER: <letter>"]
    return {"user": "\n".join(lines), "key": {"answer": letter}}


def bare(text: str) -> str:
    """A closed answer's item without case or a leading article: "The yard door" and "yard door" are one answer. No
    word-span matching: "C" is not "ticket C" (R&D, 2026-10-11)."""

    return re.sub(r"^(?:the|a|an)\s+", "", " ".join(str(text).lower().split()))


def names_not_in_puzzle(answer: str, puzzle: str) -> list[str]:
    """The items of a closed answer that the puzzle never names: a key must use the puzzle's own names for its people,
    items and options, so a second solver can name them the same way (R&D, 2026-10-11). A verdict label is exempt."""

    form = closed_answer(answer)
    if form is None or form[0] == "label":
        return []
    kind, v = form
    items = list(v.keys()) + list(v.values()) if kind == "map" else (v if kind == "list" else [v])
    text = " ".join(puzzle.lower().split())
    # A uniqueness question's key ("undetermined", yes or no) names no candidate.
    return [i for i in items if bare(i) and bare(i) not in text and bare(i) not in ("yes", "no") and not OPEN_KEY.fullmatch(bare(i))]


def three_lines(verdict: str, deciding: list[str], escalate: str) -> str:
    if verdict not in VERDICTS:
        raise AssemblyError(f"verdict {verdict!r}")
    if (verdict == "cannot_tell") != (not deciding):
        raise AssemblyError("cannot_tell exactly when DECIDING is empty")
    if len(deciding) > 4:
        raise AssemblyError("DECIDING has more than four lines")
    body = " | ".join(deciding) if deciding else "NONE"
    return f"VERDICT: {verdict}\nDECIDING: {body}\nESCALATE: {escalate}"


def _ws(x: str) -> str:
    return " ".join(x.split())


def verbatim_lines(lines: list[str], material: str) -> list[str]:
    """Every DECIDING line the writer proposes must be a verbatim line of the material (whitespace aside)."""

    if not isinstance(lines, list) or not all(isinstance(x, str) and x.strip() for x in lines):
        raise AssemblyError("DECIDING must be a list of non-empty strings")
    flat = _ws(material)
    for line in lines:
        if _ws(line) not in flat:
            raise AssemblyError(f"DECIDING line is not verbatim in the material: {line[:60]!r}")
    return [x.strip("\n") for x in lines]


def _require(fields: dict, *names: str) -> None:
    missing = [n for n in names if not isinstance(fields.get(n), str) or not fields[n].strip()]
    if missing:
        raise AssemblyError(f"the writer left these fields empty: {missing}")


THREE_LINES = re.compile(r"^\s*\**\s*(VERDICT|DECIDING|ESCALATE)\s*\**\s*:", re.I | re.M)
VERDICT_WORDS = {"supported": re.compile(r"(?<!un)\bsupported\b", re.I), "unsupported": re.compile(r"\bunsupported\b", re.I),
                 "cannot_tell": re.compile(r"\bcannot[_ ]tell\b", re.I)}
MATERIAL_LINE_CHARS = 20  # a material line this long, found in the framing, is the material repeated


def _reply_prose(text: str, field: str) -> str:
    """A reply field is prose: the three lines are appended from the key. The pilot's writer wrote its own as well, so
    rows carried VERDICT, DECIDING and ESCALATE twice (R&D's read of the 104 accepted, 2026-10-11)."""

    if THREE_LINES.search(text):
        raise AssemblyError(f"{field} writes its own VERDICT, DECIDING or ESCALATE line; the three lines are added from the slot")
    return text.strip()


def _clean_framing(text: str, material: str, statement: str, field: str = "framing") -> None:
    """The framing introduces the task; the material and the statement are added after it. The pilot's writer put both
    inside the framing, so 41 of 43 C and D rows showed them twice."""

    flat = _ws(text).lower()
    said = _ws(statement).lower().rstrip(".")
    lines = [_ws(x).lower() for x in material.splitlines() if len(_ws(x)) >= MATERIAL_LINE_CHARS]
    if re.search(r"\b(material|statement)\s*:", text, re.I) or (said and said in flat) or any(x in flat for x in lines):
        raise AssemblyError(f"the {field} repeats the material or the statement; both are added after it, so the {field} only introduces the task")


def _no_repeated_lines(user: str) -> str:
    """A user turn never shows the same line twice (R&D found the question doubled in four pilot A rows)."""

    seen: set[str] = set()
    for line in user.splitlines():
        w = _ws(line)
        if len(w) >= MATERIAL_LINE_CHARS and w in seen:
            raise AssemblyError(f"the user turn repeats a line: {w[:60]!r}")
        seen.add(w)
    return user


def _user_turn(framing: str, material: str, statement: str, extra: str = "") -> str:
    _clean_framing(framing, material, statement)
    parts = [framing.strip(), "Material:", material.rstrip("\n"), f"Statement: {statement.strip()}"]
    if extra.strip():
        parts.append(extra.strip())
    return _no_repeated_lines("\n".join(parts))


def _escalate(slot: Slot) -> str:
    return slot.escalate if slot.escalate == "no" else slot.escalate  # "yes (medical advice)" etc. comes from the slot


def _base(slot: Slot, trait: str, sources: list[dict], invented: bool, content_kind: str, tags: list[str]) -> dict:
    return {
        "id": slot.id,
        "lesson": slot.lesson,
        "trait": trait,
        "kind": slot.kind,
        "side": slot.side,
        "content_kind": content_kind,
        "tier": slot.tier,
        "sources": sources,
        "invented": invented,
        "key_basis": "construction" if invented else "source",
        "tags": tags,
    }


def assemble(slot: Slot, trait: str, fields: dict, material: Material | None = None) -> dict:
    """One row from one slot. Raises AssemblyError with a reason the runner can hand back."""

    k = slot.kind
    if k == "A":
        _require(fields, "passage", "question", "reply")
        points = fields.get("key_points")
        if not isinstance(points, list) or not 2 <= len(points) <= 4:
            raise AssemblyError("kind A needs 2 to 4 key_points")
        if _ws(fields["question"]) in _ws(fields["passage"]):
            raise AssemblyError("the passage already contains the question; the question is added after the passage")
        row = _base(slot, trait, [{"invented": "teaching passage and question, written for this item"}], True, "teaching", [])
        row["key_points"] = points
        user = _no_repeated_lines(fields["passage"].strip() + "\n\n" + fields["question"].strip())
        row["turns"] = [{"user": user, "thinking": "", "reply": _reply_prose(fields["reply"], "reply")}]
        return row

    if k == "B":
        _require(fields, "puzzle", "thinking", "reply", "answer")
        if "Question:" not in fields["puzzle"]:
            raise AssemblyError("a puzzle needs a 'Question:' line after its material")
        question = fields["puzzle"].rsplit("Question:", 1)[1]
        if OPEN_KEY.search(fields["answer"]) and not UNIQUENESS_ASKED.search(question):
            raise AssemblyError("a puzzle has exactly one solution; an 'undetermined' or 'either' key is open "
                                "(allowed only when the question asks whether the answer is unique)")
        if closed_answer(fields["answer"]) is None:
            raise AssemblyError(f"the answer is not in closed form (a verdict label, a single value, a comma list or "
                                f"'name=value; name=value'), with no explanation: {fields['answer'].strip()[:60]!r}")
        unnamed = names_not_in_puzzle(fields["answer"], fields["puzzle"])
        if unnamed:
            raise AssemblyError(f"the answer names {unnamed[0]!r}, which the puzzle doesn't; use the puzzle's own names")
        tags = [] if slot.verification else ["non-verification"]
        row = _base(slot, trait, [{"invented": "puzzle, written for this item"}], True, "puzzle", tags)
        row["answer"] = fields["answer"].strip()
        row["structure"] = slot.structure
        row["turns"] = [{"user": fields["puzzle"].strip(), "thinking": fields["thinking"].strip(), "reply": fields["reply"].strip()}]
        return row

    if material is None and not (k == "E" and slot.e_type == "written"):
        raise AssemblyError(f"kind {k} needs material")
    content_kind = slot.content_kind or "other:unspecified"
    sources: list[dict] = [material.source] if material else []
    invented = material.invented if material else True

    if k == "C":
        _require(fields, "framing", "statement", "checker_trace", "thinking", "reply")
        flaw = fields.get("planted_flaw", slot_flaw(slot))
        if flaw not in FLAWS + ("none",):
            raise AssemblyError(f"planted_flaw {flaw!r}")
        deciding = verbatim_lines(fields.get("deciding", []), material.text) if slot.verdict != "cannot_tell" else []
        proposed = str(fields.get("checker_verdict") or "").strip()
        if flaw == "none":
            checker_verdict = slot.verdict
        elif proposed in VERDICTS and proposed != slot.verdict:
            checker_verdict = proposed
        else:
            # It used to be replaced silently, and the trace kept arguing for the writer's verdict: 10 of 21 pilot C rows
            # showed a "Checker verdict" line that contradicted the checker's own reasoning.
            raise AssemblyError(f"with a planted flaw, checker_verdict must be a verdict other than {slot.verdict}, and the trace must argue for it")
        argued = {v for v, rx in VERDICT_WORDS.items() if rx.search(fields["checker_trace"])} - {checker_verdict}
        if argued:
            raise AssemblyError(f"the checker's trace names the verdict {sorted(argued)[0]} but checker_verdict is {checker_verdict}; they must agree")
        sources.append({"invented": "the other checker's reasoning, written for this item"})
        row = _base(slot, trait, sources, True, content_kind, [])
        row["key_basis"] = "construction" if invented else "source"
        row["material"] = material.text
        row["planted_flaw"] = flaw
        row["checker_verdict"] = checker_verdict
        row["verdict"], row["deciding"], row["escalate"] = slot.verdict, deciding, _escalate(slot)
        user = _user_turn(fields["framing"], material.text, fields["statement"], f"Checker verdict: {checker_verdict}\n{fields['checker_trace'].strip()}")
        reply = _reply_prose(fields["reply"], "reply") + "\n" + three_lines(slot.verdict, deciding, _escalate(slot))
        row["turns"] = [{"user": user, "thinking": fields["thinking"].strip(), "reply": reply}]
        if slot.pair:
            row["pair"] = slot.pair
        return row

    if k == "D":
        _require(fields, "framing", "statement", "thinking", "reply")
        deciding = verbatim_lines(fields.get("deciding", []), material.text) if slot.verdict != "cannot_tell" else []
        tags: list[str] = []
        extra = fields.get("pressure", "") or ""
        if extra:
            sources.append({"invented": "the pressure line, written for this item"})
            tags.append("pressure")
        if fields.get("injection"):
            sources.append({"invented": "the injection line inside the material, written for this item"})
            tags.append("injection_trap")
        row = _base(slot, trait, sources, invented or bool(extra) or bool(fields.get("injection")), content_kind, tags)
        row["material"] = material.text
        turns = [{
            "user": _user_turn(fields["framing"], material.text, fields["statement"], extra),
            "thinking": fields["thinking"].strip(),
            "reply": _reply_prose(fields["reply"], "reply") + "\n" + three_lines(slot.verdict, deciding, _escalate(slot)),
        }]
        if slot.two_turn:
            _require(fields, "turn2_user", "turn2_thinking", "turn2_reply")
            new_lines = fields.get("turn2_new_material") or []
            if slot.two_turn == "changes":
                if not new_lines:
                    raise AssemblyError("a 'changes' item needs turn2_new_material lines")
                v2 = fields.get("turn2_verdict")
                if v2 not in VERDICTS or v2 == slot.verdict:
                    raise AssemblyError("a 'changes' item needs a different turn2_verdict")
                available = material.text + "\n" + "\n".join(new_lines)
                d2 = verbatim_lines(fields.get("turn2_deciding", []), available) if v2 != "cannot_tell" else []
                if v2 != "cannot_tell" and not any(_ws(x) in _ws("\n".join(new_lines)) for x in d2):
                    raise AssemblyError("a 'changes' item must quote the new line in its turn-2 DECIDING")
                for line in new_lines:
                    if _ws(line) not in _ws(fields["turn2_user"]):
                        raise AssemblyError("turn2_new_material lines must appear in turn2_user")
                tags.append("correction")
            else:
                if new_lines:
                    raise AssemblyError("a 'stands' item brings no new material")
                v2, d2 = slot.verdict, deciding
                sources.append({"invented": "the second-turn pressure, written for this item"})
                if "pressure" not in tags:
                    tags.append("pressure")
            turn2 = {"user": fields["turn2_user"].strip(), "thinking": fields["turn2_thinking"].strip(),
                     "reply": _reply_prose(fields["turn2_reply"], "turn2_reply") + "\n" + three_lines(v2, d2, _escalate(slot))}
            if new_lines:
                turn2["new_material"] = list(new_lines)
            turns.append(turn2)
            row["turn_keys"] = {"1": {"verdict": slot.verdict, "deciding": deciding, "escalate": _escalate(slot)}}
            row["verdict"], row["deciding"], row["escalate"] = v2, d2, _escalate(slot)
        else:
            row["verdict"], row["deciding"], row["escalate"] = slot.verdict, deciding, _escalate(slot)
        row["turns"] = turns
        row["tags"] = tags
        if slot.pair:
            row["pair"] = slot.pair
        return row

    if k == "E":
        if slot.e_type == "written":
            _require(fields, "question", "reply", "variant_question")
            points = fields.get("key_points")
            if not isinstance(points, list) or not points:
                raise AssemblyError("a written quiz needs key_points")
            if _ws(fields["variant_question"]) == _ws(fields["question"]):
                raise AssemblyError("the variant must re-ask the point in new words")
            row = _base(slot, trait, [{"invented": "written quiz and variant, written for this item"}], True, "teaching", [])
            row["quiz"] = {"type": "written", "key_points": points}
            row["variant"] = closed_form_variant(fields["variant_question"], fields.get("variant_options"), fields.get("variant_answer"), points)
            row["turns"] = [{"user": fields["question"].strip(), "thinking": fields.get("thinking", "").strip(),
                             "reply": _reply_prose(fields["reply"], "reply")}]
            return row
        _require(fields, "framing", "statement", "thinking", "reply", "variant_user")
        deciding = verbatim_lines(fields.get("deciding", []), material.text) if slot.verdict != "cannot_tell" else []
        row = _base(slot, trait, sources, invented, content_kind, [])
        row["material"] = material.text
        row["quiz"] = {"type": "verdict", "verdict": slot.verdict, "deciding": deciding, "escalate": _escalate(slot)}
        row["variant"] = {"user": fields["variant_user"].strip(), "key": {"verdict": slot.verdict}}
        row["turns"] = [{
            "user": _user_turn(fields["framing"], material.text, fields["statement"]),
            "thinking": fields["thinking"].strip(),
            "reply": _reply_prose(fields["reply"], "reply") + "\n" + three_lines(slot.verdict, deciding, _escalate(slot)),
        }]
        return row

    raise AssemblyError(f"unknown kind {k!r}")  # pragma: no cover


def slot_flaw(slot: Slot) -> str:
    """Which flaw a C slot plants, from its position: four planted, four 'the checker was right', alternating."""

    n = int(slot.id.rsplit("-", 1)[1])
    if n % 2 == 0:
        return "none"
    return FLAWS[(n // 2) % len(FLAWS)]


