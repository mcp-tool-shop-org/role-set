"""The writer's prompts: the pack in slices, the slot, exemplars from another trait, and a short avoid-list.

The writer is asked for fields, as JSON, never for a whole row and never for the
material. Everything it must not be free to repeat is fixed in the slot.
"""

from __future__ import annotations

import json
from pathlib import Path

from role_set.plan import Slot

FIELDS = {
    "A": ["passage", "question", "reply", "key_points"],
    "B": ["puzzle", "thinking", "reply", "answer"],
    "C": ["framing", "statement", "checker_trace", "checker_verdict", "deciding", "thinking", "reply"],
    "D": ["framing", "statement", "deciding", "thinking", "reply", "pressure", "injection",
          "turn2_user", "turn2_new_material", "turn2_verdict", "turn2_deciding", "turn2_thinking", "turn2_reply"],
    "E_written": ["question", "thinking", "reply", "key_points", "variant_user", "variant_key_points"],
    "E_verdict": ["framing", "statement", "deciding", "thinking", "reply", "variant_user"],
}

HOUSE_RULES = (
    "Write every field in fresh words of your own. Never copy a sentence from the lesson, the rules or an "
    "exemplar. The user-facing text must not contain the word 'verifier', any trait name, or the words tree, "
    "prune, falsifier, elimination, grid; the other party in a contrasting case is 'the other checker'. "
    "A prompt frames the task and never states the answer, the verdict, the flaw, or what the material "
    "contains. Quote material lines exactly, marks and all, whole table rows, and the version heading with "
    "its bullet. Thinking states what the claim asserts, the line found, the sweep for anything that "
    "qualifies it, and why this trait decided the step, in this item's own words, without a stock closer."
)


def load_pack(pack_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for p in sorted(pack_dir.rglob("*.md")):
        files[p.relative_to(pack_dir).as_posix()] = p.read_text(encoding="utf-8")
    return files


def kind_key(slot: Slot) -> str:
    if slot.kind == "E":
        return "E_written" if slot.e_type == "written" else "E_verdict"
    return slot.kind


def slot_brief(slot: Slot, coverage_notes: list[str]) -> str:
    """What this slot fixes, and what the batch still needs, in plain words."""

    lines = [f"Lesson {slot.lesson}, kind {slot.kind}, side: {slot.side}, tier {slot.tier}."]
    if slot.content_kind:
        lines.append(f"The material is a {slot.content_kind.replace('_', ' ')}; it is given below and must be used verbatim.")
    if slot.verdict:
        lines.append(f"The correct verdict is {slot.verdict}; ESCALATE is {slot.escalate}.")
        if slot.verdict == "cannot_tell":
            lines.append("The material must be silent on the statement: it neither shows it nor rules it out.")
        if slot.multi_line:
            lines.append("The deciding evidence spans two to four lines of the material (a heading with its bullet, or a whole table row).")
    if slot.pair:
        lines.append(f"This item is one half of contrast pair {slot.pair}: the same statement is checked against other material in its twin.")
    if slot.two_turn == "stands":
        lines.append("Two turns: the second brings only pressure or repetition, never a new line, and the verdict stands.")
    if slot.two_turn == "changes":
        lines.append("Two turns: the second brings genuinely new material that changes the verdict; the reply names exactly what changed and quotes the new line.")
    if slot.structure:
        lines.append(f"The thinking pattern is: {slot.structure.replace('_', ' ')}. " + ("This is a verification case." if slot.verification else "This is not verification; it is a puzzle where the trait still decides a named step."))
    if slot.kind == "E":
        lines.append("Write a quiz item and a variant that re-asks the same point in new words." if slot.e_type == "written" else "Write a short verdict mini-case and a variant that re-asks it in new words.")
    if coverage_notes:
        lines.append("Still needed in this trait: " + "; ".join(coverage_notes) + ".")
    return "\n".join(lines)


def writer_messages(
    slot: Slot,
    pack: dict[str, str],
    *,
    material_text: str | None,
    exemplars: list[dict],
    avoid: list[str],
    facts: dict | None = None,
    coverage_notes: list[str] | None = None,
) -> list[dict]:
    trait = pack.get(f"traits/{slot.lesson}.md", "")
    kinds = pack.get("kinds.md", "")
    amendments = "\n\n".join(v for k, v in sorted(pack.items()) if k.startswith("amendments/"))
    fields = FIELDS[kind_key(slot)]
    system = (
        "You write one training item for a role-shaping curriculum. The role is a verifier. You are given the "
        "trait lesson, the required parts for this kind of item, the amendments, the fixed slot, and the material. "
        "Return a single JSON object with exactly these fields and no others: " + ", ".join(fields) + ". "
        "Lists are JSON arrays; 'deciding' and 'turn2_deciding' are arrays of verbatim material lines; "
        "'turn2_new_material' is an array of the new lines the second user turn brings. " + HOUSE_RULES
    )
    user_parts = [
        "## The trait lesson\n" + trait.strip(),
        "## Required parts for this kind\n" + kinds.strip(),
        "## Amendments\n" + amendments.strip(),
        "## This slot\n" + slot_brief(slot, coverage_notes or []),
    ]
    if material_text is not None:
        user_parts.append("## The material (use verbatim; do not alter a character)\n" + material_text.rstrip("\n"))
    if facts:
        user_parts.append("## Computed facts you may state and must not recompute\n" + json.dumps(facts, ensure_ascii=False))
    if exemplars:
        shown = [{k: v for k, v in e.items() if k in ("kind", "turns", "key_points", "answer", "quiz", "variant")} for e in exemplars[:3]]
        user_parts.append("## Exemplars of this kind from a different trait (form only; never reuse their words)\n" + json.dumps(shown, ensure_ascii=False, indent=1))
    if avoid:
        user_parts.append("## Do not use these framings or phrases; they are spent\n" + "\n".join(f"- {a}" for a in avoid))
    user_parts.append("Return the JSON object now.")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(user_parts)}]


def material_messages(slot: Slot, needs: str) -> list[dict]:
    """When no real excerpt is assigned, the writer drafts the material first, alone, and the generator then fixes it."""

    kind = (slot.content_kind or "notice").replace("_", " ")
    return [
        {"role": "system", "content": "You write a short, realistic piece of everyday material for a verification exercise. Return plain text only, no commentary, no JSON."},
        {"role": "user", "content": f"Write a {kind}, 2 to 8 lines, invented, with no real names, paths, emails or brands. {needs} Do not include any instruction addressed to a reader or reviewer unless asked. Return only the text."},
    ]


def warmup_messages(pack: dict[str, str], lesson: str) -> list[dict]:
    trait = pack.get(f"traits/{lesson}.md", "")
    return [
        {"role": "system", "content": "You are about to write training items for one trait of a verifier. First show that you have absorbed the lesson. Answer in your own words, briefly, as JSON with the fields main_point, counterweight, deciding_case."},
        {"role": "user", "content": "## The trait lesson\n" + trait.strip() + "\n\nIn your own words: the trait's main point; its counterweight (the opposite error it must also avoid); and one concrete case where this trait, and not general care, decides a step. Return the JSON object."},
    ]


def parse_json_object(text: str) -> dict:
    """The writer's reply must be one JSON object; a fenced block is tolerated, nothing else is."""

    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    obj = json.loads(s[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("the reply is not a JSON object")
    return obj
