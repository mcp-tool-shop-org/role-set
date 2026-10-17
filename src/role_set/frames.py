"""Opening frames (BUILD-SPEC B2, as R&D ruled on 2026-10-16): the shape of a C, D or verdict-E item's framing.

The user turn's layout never changes: the framing, then `Material:`, then `Statement:`. The student meets one format
at evaluation, and the gates read the framing as the text before `Material:`. A frame fixes the framing's structure,
not its words, so the writer still varies the wording, and the framing still has to pass the gates' framing checks.
The plan assigns frames by seed, balanced within a lesson and across a batch. The FRAME gate checks the structure
mechanically: sentence count, final punctuation and, for `line_question`, that the framing doesn't point at the
answer (no digits, and no four words in a row shared with the material).

The instruction lines are prompt text, written by Claude for this specialized tool; they never fill a target field
(provenance: B9).
"""

from __future__ import annotations

import re

# id -> the one-line instruction the writer gets. Order matters: it is the rotation order.
FRAMES = {
    "ask": "Open with one sentence that asks for the check plainly, as an instruction, not a question.",
    "question": "Open with a question the user asks about the statement; the framing ends with a question mark.",
    "situation_then_ask": "Open with a sentence that sets the scene (who has the material and why), then a second sentence that asks for the check.",
    "line_question": "Open with a question asking which line of the material decides the statement. Don't name, number, quote or paraphrase any line.",
}
FRAMED_KINDS = ("C", "D")  # and verdict-type E
NGRAM = 4
# line_question points at no line (Argon §9.4 #13): no ordinal or position ("the first line", "the last item"), no run
# of three words from the material that carries a content word, and no two of the material's content words.
POINTER = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last|final|opening|closing|top|"
                     r"bottom|middle|penultimate|next-to-last)\s+(line|item|row|rule|clause|entry|sentence|point)s?\b", re.IGNORECASE)
LQ_NGRAM = 3
LQ_SHARED_WORDS = 2
STOP = frozenset("""a an the of to in on at for from by with as is are was were be been it its this that these those which what
who whom whose where when how does do did can could would should will shall may might must and or but if not no line lines
material statement notice settle settles settled decide decides decided here there one any some each part""".split())


def framed(kind: str, e_type: str | None) -> bool:
    return kind in FRAMED_KINDS or (kind == "E" and e_type == "verdict")


def framing_text(user: str) -> str:
    """The framing as the gates read it: the user turn before "Material:"."""

    at = user.find("Material:")
    return user[:at].strip() if at > 0 else ""


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [p for p in parts if p.strip()]


def _grams(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _stem(word: str) -> str:
    for end in ("ing", "ed", "es", "s"):
        if word.endswith(end) and len(word) - len(end) >= 3:
            return word[: -len(end)]
    return word


def _content(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 2}


def check_frame(frame: str, user: str, material: str) -> str | None:
    """Why the framing doesn't open in its frame, or None."""

    if frame not in FRAMES:
        return f"unknown frame {frame!r}"
    text = framing_text(user)
    if not text:
        return "the framing is empty"
    parts = sentences(text)
    asks = text.rstrip().endswith("?")
    if frame == "ask":
        if len(parts) != 1 or asks:
            return f"frame ask is one sentence that is an instruction, not a question; the framing has {len(parts)} sentence(s){' and ends with ?' if asks else ''}"
    elif frame == "question":
        if not asks:
            return "frame question ends with a question mark; the framing doesn't"
    elif frame == "situation_then_ask":
        if len(parts) < 2 or parts[0].endswith("?"):
            return "frame situation_then_ask sets the scene in one sentence, then asks in another; the framing has " + (
                f"{len(parts)} sentence(s)" if len(parts) < 2 else "a question first")
    elif frame == "line_question":
        if not asks or not re.search(r"\blines?\b", text, re.I):
            return "frame line_question asks which line decides, ending with a question mark"
        if re.search(r"\d", text):
            return "frame line_question names no line: the framing holds a digit"
        hit = POINTER.search(text)
        if hit:
            return f"frame line_question names no line: '{hit.group(0)}' points at one"
        shared = sorted(g for g in _grams(text, LQ_NGRAM) & _grams(material, LQ_NGRAM) if any(w not in STOP for w in g))
        if shared:
            return "frame line_question doesn't point at the answer: the framing repeats '" + " ".join(shared[0]) + "' from the material"
        common = sorted(_content(text) & _content(material))
        if len(common) >= LQ_SHARED_WORDS:
            return "frame line_question doesn't point at the answer: the framing uses the material's words " + ", ".join(repr(w) for w in common[:3])
    return None


def assign(count: int, lesson_index: int, rng) -> list[str]:
    """count frames for one lesson: every frame within one of the others, and which frames take the extra rotates with
    the lesson's place in the batch, so each frame stays near its share across a batch."""

    names = list(FRAMES)
    base, extra = divmod(count, len(names))
    start = (lesson_index * extra) % len(names) if extra else 0
    plus = {names[(start + i) % len(names)] for i in range(extra)}
    out = [n for n in names for _ in range(base + (1 if n in plus else 0))]
    rng.shuffle(out)
    return out
