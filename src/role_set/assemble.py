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


def _user_turn(framing: str, material: str, statement: str, extra: str = "") -> str:
    parts = [framing.strip(), "Material:", material.rstrip("\n"), f"Statement: {statement.strip()}"]
    if extra.strip():
        parts.append(extra.strip())
    return "\n".join(parts)


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
        row = _base(slot, trait, [{"invented": "teaching passage and question, written for this item"}], True, "teaching", [])
        row["key_points"] = points
        row["turns"] = [{"user": fields["passage"].strip() + "\n\n" + fields["question"].strip(), "thinking": "", "reply": fields["reply"].strip()}]
        return row

    if k == "B":
        _require(fields, "puzzle", "thinking", "reply", "answer")
        if "Question:" not in fields["puzzle"]:
            raise AssemblyError("a puzzle needs a 'Question:' line after its material")
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
        checker_verdict = slot.verdict if flaw == "none" else _other_verdict(slot.verdict, fields.get("checker_verdict"))
        sources.append({"invented": "the other checker's reasoning, written for this item"})
        row = _base(slot, trait, sources, True, content_kind, [])
        row["key_basis"] = "construction" if invented else "source"
        row["material"] = material.text
        row["planted_flaw"] = flaw
        row["checker_verdict"] = checker_verdict
        row["verdict"], row["deciding"], row["escalate"] = slot.verdict, deciding, _escalate(slot)
        user = _user_turn(fields["framing"], material.text, fields["statement"], f"Checker verdict: {checker_verdict}\n{fields['checker_trace'].strip()}")
        reply = fields["reply"].strip() + "\n" + three_lines(slot.verdict, deciding, _escalate(slot))
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
            "reply": fields["reply"].strip() + "\n" + three_lines(slot.verdict, deciding, _escalate(slot)),
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
                     "reply": fields["turn2_reply"].strip() + "\n" + three_lines(v2, d2, _escalate(slot))}
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
            _require(fields, "question", "reply", "variant_user")
            points, vpoints = fields.get("key_points"), fields.get("variant_key_points")
            if not isinstance(points, list) or not points or not isinstance(vpoints, list) or not vpoints:
                raise AssemblyError("a written quiz needs key_points and variant_key_points")
            if _ws(fields["variant_user"]) == _ws(fields["question"]):
                raise AssemblyError("the variant must re-ask the point in new words")
            row = _base(slot, trait, [{"invented": "written quiz and variant, written for this item"}], True, "teaching", [])
            row["quiz"] = {"type": "written", "key_points": points}
            row["variant"] = {"user": fields["variant_user"].strip(), "key": vpoints}
            row["turns"] = [{"user": fields["question"].strip(), "thinking": fields.get("thinking", "").strip(), "reply": fields["reply"].strip()}]
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
            "reply": fields["reply"].strip() + "\n" + three_lines(slot.verdict, deciding, _escalate(slot)),
        }]
        return row

    raise AssemblyError(f"unknown kind {k!r}")  # pragma: no cover


def slot_flaw(slot: Slot) -> str:
    """Which flaw a C slot plants, from its position: four planted, four 'the checker was right', alternating."""

    n = int(slot.id.rsplit("-", 1)[1])
    if n % 2 == 0:
        return "none"
    return FLAWS[(n // 2) % len(FLAWS)]


def _other_verdict(key: str, proposed) -> str:
    if proposed in VERDICTS and proposed != key:
        return proposed
    return "unsupported" if key != "unsupported" else "supported"
