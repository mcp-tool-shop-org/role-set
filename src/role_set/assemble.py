"""Assembly: the slot, the material and the writer's fields become one row in aspire-si's schema.

The writer never holds the pen on the material, the keys, the sources or the
slot fields. It returns only the parts a writer should write; this module puts
the row together, appends the three lines from the key, and refuses anything
that doesn't fit before the gates ever see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from role_set.plan import ORDERED_KINDS, Slot

FLAWS = ("rubber_stamp", "guess", "loop", "nitpick", "fix", "obeyed_material", "caved", "reject_by_default")
VERDICTS = ("supported", "unsupported", "cannot_tell")
SILENT_NEEDED = ("T01", "T04", "T14")  # apply items whose twin drops the deciding lines (aspire-si#137)

# Kind B keys (R&D's ruling on the 2026-10-11 pilot): a closed answer, never free text, so a second solver can be
# compared against it; and a puzzle keyed "undetermined" is open, so it can't teach a decided step.
CLOSED_ITEM_WORDS = 6
OPEN_KEY = re.compile(r"\b(undetermined|undecidable|either|ambiguous|cannot be determined|can't be determined|"
                      r"not determined|multiple (?:valid )?(?:solutions|answers|assignments))\b", re.I)
UNIQUENESS_ASKED = re.compile(r"\b(unique|only one|is there (?:a|one|exactly)|can (?:we|you) (?:tell|determine)|determined)\b", re.I)


def closed_answer(text: str) -> tuple[str, object] | None:
    """A kind-B answer in closed form, normalised: ("label", v), ("map", {name: value}), ("list", [items]),
    ("groups", [sorted names per group], sorted) for an underdetermined set of groups ("red and blue; red and green",
    role-set#45) or ("value", v); None for anything else (an explanation, a sentence with a reason, a clause after a
    dash)."""

    t = " ".join(str(text).split()).strip().rstrip(".")
    if not t or re.search(r"\s[—–-]\s|\bbecause\b|\bsince\b", t, re.I):
        return None
    low = t.lower()
    if low in VERDICTS:
        return ("label", low)
    parts = [p.strip() for p in t.split(";") if p.strip()]
    if "=" in t and all(re.fullmatch(r"[^=;]{1,40}=[^=;]{1,40}", p) for p in parts):
        return ("map", {k.strip().lower(): v.strip().lower() for k, v in (p.split("=", 1) for p in parts)})
    if "=" not in t and ":" not in t and len(parts) > 1:
        groups = [[n.strip().lower() for n in re.split(r"\band\b|,", p) if n.strip()] for p in parts]
        if all(g and all(len(n.split()) <= CLOSED_ITEM_WORDS for n in g) for g in groups):
            return ("groups", sorted(sorted(g) for g in groups))
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


OPTION_LABEL = re.compile(r"^\s*\(?[A-Da-d][.)]\s+")
FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|me|we|our|us)\b", re.IGNORECASE)
SECOND_PERSON = re.compile(r"\b(you|your|you're)\b", re.IGNORECASE)


def register(option: str) -> tuple[bool, bool, str]:
    """An option's form: opens with a capital, ends as a sentence, and its person."""

    letters = [c for c in option if c.isalpha()]
    person = "first" if FIRST_PERSON.search(option) else "second" if SECOND_PERSON.search(option) else "third"
    return (bool(letters) and letters[0].isupper(), option.rstrip().endswith((".", "!", "?")), person)


def register_tell(options: list[str]) -> str | None:
    """R&D's pilot-2 key-check: in all 10 written E items the keyed option was its key point pasted as a lowercase
    fragment among full-sentence distractors. Length was balanced; the form gave the answer away."""

    forms = [register(o) for o in options]
    for i, what in enumerate(("opening capital", "sentence ending", "person")):
        if len({f[i] for f in forms}) > 1:
            return what
    return None


def closed_form_variant(question: str, correct, distractors, seed: str) -> dict:
    """A written E item's closed-form variant (handoff Amendment 5): the stem, 3 or 4 options lettered A to D on their
    own lines, an ANSWER line asked for, key {"answer": letter}. The writer gives the correct option and 2 or 3
    distractors as separate fields, so its position can't be wrong. Assembly shuffles them with an RNG seeded by the
    slot id and sets the letter, so the keyed letter is uniform by construction (R&D on #26). No two options are the
    same, and they share one form. Whether the correct option carries a key point, and how near a distractor may come,
    is the gates' closed-form rule with the run's settings (keyed carry verbatim, paraphrase or either;
    aspire-si-gates 1.4.0), checked by `check_item` right after assembly; role-set keeps no copy of it. A writer's own
    "A." labels are stripped."""

    import random

    if not isinstance(correct, str) or not correct.strip():
        raise AssemblyError("variant_correct must be the right option's text")
    if not isinstance(distractors, list) or not 2 <= len(distractors) <= 3 or not all(isinstance(o, str) and o.strip() for o in distractors):
        raise AssemblyError("variant_distractors must be a list of 2 or 3 non-empty option texts")
    options = [OPTION_LABEL.sub("", o).strip() for o in [correct] + distractors]
    texts = [_norm_point(o) for o in options]
    if len(set(texts)) != len(texts):
        raise AssemblyError("two variant options are the same")
    tell = register_tell(options)
    if tell:
        raise AssemblyError(f"the options differ in form ({tell}); write every option the same way: a full sentence with a "
                            "capital and a full stop, in the same person")
    order = list(range(len(options)))
    random.Random(seed).shuffle(order)
    letters = LETTERS[:len(options)]
    shown = [options[i] for i in order]
    letter = letters[order.index(0)]
    lines = [question.strip(), ""] + [f"{letters[i]}. {o}" for i, o in enumerate(shown)]
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
    items = (list(v.keys()) + list(v.values()) if kind == "map" else v if kind == "list"
             else [n for g in v for n in g if n != "none"] if kind == "groups" else [v])
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


MATERIAL_LINE_CHARS = 20  # a material line this long, found in the framing, is the material repeated
MARKDOWN_LINK = re.compile(r"\[([^\[\]\n]+)\]\([^)\n]+\)")
_MARK_ATOM_RE = re.compile(
    r"\[[^\[\]\n]+\]\([^)\n]+\)"
    r"|\*\*(?:`[^`\n]+`|[^*`\n])+\*\*"
    r"|__(?:`[^`\n]+`|[^_`\n])+__"
    r"|`[^`\n]+`"
    r"|\S"
)


def _ws(x: str) -> str:
    return " ".join(x.split())


def _strip_inline_marks(s: str) -> str:
    """Remove backticks, ** and __, replace [text](url) with text, and collapse whitespace."""

    s = MARKDOWN_LINK.sub(r"\1", s)
    s = s.replace("`", "").replace("**", "").replace("__", "")
    return _ws(s)


def _extract_exact_span(mat_line: str, q_stripped: str) -> str | None:
    raw_atoms = [(m, _strip_inline_marks(m.group(0))) for m in _MARK_ATOM_RE.finditer(mat_line)]
    atoms = [(m, s) for m, s in raw_atoms if s]
    if not atoms:
        return None
    has_space_before = [
        i > 0 and any(c.isspace() for c in mat_line[atoms[i - 1][0].end():atoms[i][0].start()])
        for i in range(len(atoms))
    ]
    target_len = len(q_stripped)
    spans: list[str] = []
    for i, (m_start, s_start) in enumerate(atoms):
        if not q_stripped.startswith(s_start):
            continue
        cur_len = 0
        for j in range(i, len(atoms)):
            m_end, s_end = atoms[j]
            if j > i and has_space_before[j]:
                cur_len += 1
            cur_len += len(s_end)
            if cur_len < target_len:
                continue
            if cur_len == target_len and q_stripped.endswith(s_end):
                cand = mat_line[m_start.start():m_end.end()]
                if _strip_inline_marks(cand) == q_stripped:
                    spans.append(cand)
                    if len(spans) > 1:
                        return None
            break
    return spans[0] if len(spans) == 1 else None


def _first_diff(a: str, b: str) -> int:
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))


def _closest_material_line(line: str, material: str) -> tuple[str, int]:
    from difflib import SequenceMatcher  # noqa: PLC0415

    q = _ws(line)
    candidates = [_ws(m) for m in material.splitlines() if m.strip()]
    if not candidates:
        return "", 0
    best = max(candidates, key=lambda m: (_first_diff(q, m), SequenceMatcher(None, q, m).ratio()))
    return best, _first_diff(q, best)


def resolve_verbatim_line(line: str, material: str) -> tuple[str, dict | None] | None:
    """Resolve one DECIDING quote against `material`:
    - If strict (`_ws(line) in _ws(material)`), returns `(line.strip("\\n"), None)`.
    - Otherwise compares `_strip_inline_marks(line)` against non-empty lines of `material.splitlines()`.
      A material line matches when its stripped form equals the stripped quote, or when it is not a table row and
      the stripped quote (at least `MATERIAL_LINE_CHARS` chars) is a substring of its stripped form.
    - When exactly one material line matches, returns `(repaired_text, {"from": line.strip("\\n"), "to": repaired_text})`.
    - When zero or two or more material lines match (or a substring span cannot be extracted unambiguously), returns `None`.
    """

    cleaned = line.strip("\n")
    flat = _ws(material)
    if _ws(cleaned) and _ws(cleaned) in flat:
        return cleaned, None
    q_stripped = _strip_inline_marks(cleaned)
    if not q_stripped:
        return None
    mat_lines = [m.strip() for m in material.splitlines() if m.strip()]
    matches: list[tuple[str, bool]] = []
    for m in mat_lines:
        m_stripped = _strip_inline_marks(m)
        if q_stripped == m_stripped:
            matches.append((m, True))
        elif (
            len(q_stripped) >= MATERIAL_LINE_CHARS
            and not m.lstrip().startswith("|")
            and q_stripped in m_stripped
        ):
            matches.append((m, False))
    if len(matches) != 1:
        return None
    matched_line, is_full = matches[0]
    if is_full:
        return matched_line, {"from": cleaned, "to": matched_line}
    span = _extract_exact_span(matched_line, q_stripped)
    if span is None:
        return None
    return span, {"from": cleaned, "to": span}


class VerbatimLines(list):
    """A list[str] of resolved DECIDING lines carrying `.repairs` (`[{"from": quote, "to": material_text}, ...]`)."""

    def __init__(self, items: list[str], repairs: list[dict] | None = None) -> None:
        super().__init__(items)
        self.repairs: list[dict] = list(repairs or [])


def verbatim_lines(lines: list[str], material: str, repairs: list[dict] | None = None) -> VerbatimLines:
    """Every DECIDING line the writer proposes must be a verbatim line of the material (whitespace aside), or resolve
    unambiguously to a single material line when inline markdown marks (backticks, **, __, [text](url)) are ignored."""

    if not isinstance(lines, list) or not all(isinstance(x, str) and x.strip() for x in lines):
        raise AssemblyError("DECIDING must be a list of non-empty strings")
    out: list[str] = []
    recorded: list[dict] = []
    for line in lines:
        res = resolve_verbatim_line(line, material)
        if res is None:
            closest, diff_at = _closest_material_line(line, material)
            raise AssemblyError(
                f"DECIDING line is not verbatim in the material (first differs at char {diff_at}); "
                f"line: {line.strip(chr(10))!r}; closest material line: {closest!r}"
            )
        resolved, repair = res
        out.append(resolved)
        if repair is not None:
            recorded.append(repair)
            if repairs is not None:
                repairs.append(repair)
    return VerbatimLines(out, recorded)


def _require(fields: dict, *names: str) -> None:
    missing = [n for n in names if not isinstance(fields.get(n), str) or not fields[n].strip()]
    if missing:
        raise AssemblyError(f"the writer left these fields empty: {missing}")


THREE_LINES = re.compile(r"^\s*\**\s*(VERDICT|DECIDING|ESCALATE)\s*\**\s*:", re.I | re.M)
VERDICT_WORDS = {"supported": re.compile(r"(?<!un)\bsupported\b", re.I), "unsupported": re.compile(r"\bunsupported\b", re.I),
                 "cannot_tell": re.compile(r"\bcannot[_ ]tell\b", re.I)}


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
        **({"split": slot.split} if slot.split else {}),
    }


def _apply_fields(slot: Slot, fields: dict, row: dict) -> None:
    """An apply item (aspire-si#127): one turn, asked once and never trained. A T10 apply item also carries its keyed
    companion claim, `r10_twin: {statement, verdict}`, a strengthened or weakened claim on the same material that the
    apply sitting puts in place of the item's one Statement line; its keyed verdict must differ from the item's (R&D on
    aspire-si#128). R&D key-checks both."""

    if slot.split != "apply":
        return
    if len(row["turns"]) != 1:
        raise AssemblyError("an apply item is one turn: the model is asked its first user turn once")
    if slot.lesson == "T09":
        # R9's reversed-material twin is built only where line order carries no meaning (Argon's review, settled by
        # R&D; aspire-si#131). The writer says which; R&D key-checks it; ordered content can't be order-free.
        order_free = fields.get("order_free")
        if not isinstance(order_free, bool):
            raise AssemblyError("a T09 apply item needs order_free: true when reversing the material's lines can't change "
                                "the right answer, false when the order carries meaning")
        if order_free and slot.content_kind in ORDERED_KINDS:
            raise AssemblyError(f"a {slot.content_kind}'s line order carries meaning, so order_free must be false")
        row["order_free"] = order_free
    if slot.lesson in SILENT_NEEDED:
        # The twin that removes the deciding lines is built only where that leaves the material silent on the
        # statement (R&D's ruling on Argon's review; aspire-si#137's SILENT_NEEDED). The writer says which; R&D
        # key-checks it, as order_free on T09.
        silent = fields.get("silent_without_deciding")
        if not isinstance(silent, bool):
            raise AssemblyError(f"a {slot.lesson} apply item needs silent_without_deciding: true when removing its deciding "
                                "lines leaves the material silent on the statement, false otherwise")
        if silent and not row.get("deciding"):
            raise AssemblyError("silent_without_deciding needs deciding lines to remove; a cannot_tell item has none")
        row["silent_without_deciding"] = silent
    if slot.lesson != "T10":
        return
    twin = fields.get("r10_twin")
    if not isinstance(twin, dict) or not str(twin.get("statement") or "").strip() or twin.get("verdict") not in VERDICTS:
        raise AssemblyError("a T10 apply item needs r10_twin: {statement, verdict}, a strengthened or weakened claim on the "
                            "same material, keyed")
    if twin["verdict"] == row["verdict"]:
        raise AssemblyError(f"r10_twin's verdict must differ from the item's ({row['verdict']}): the pair tests literal reading")
    statements = [ln for ln in row["turns"][0]["user"].splitlines() if ln.startswith("Statement:")]
    if len(statements) != 1:
        raise AssemblyError(f"the apply sitting puts r10_twin in place of the one Statement line; this user turn has {len(statements)}")
    if _ws(twin["statement"]) == _ws(statements[0][len("Statement:"):]):
        raise AssemblyError("r10_twin's statement repeats the item's own")
    row["r10_twin"] = {"statement": twin["statement"].strip(), "verdict": twin["verdict"]}


def assemble(slot: Slot, trait: str, fields: dict, material: Material | None = None, repairs: list[dict] | None = None) -> dict:
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
        deciding = list(verbatim_lines(fields.get("deciding", []), material.text, repairs=repairs)) if slot.verdict != "cannot_tell" else []
        proposed = str(fields.get("checker_verdict") or "").strip()
        if flaw == "none":
            checker_verdict = slot.verdict
        elif proposed in VERDICTS and proposed != slot.verdict and proposed in FLAW_VERDICTS.get(flaw, VERDICTS):
            checker_verdict = proposed
        elif proposed in VERDICTS and proposed != slot.verdict:
            raise AssemblyError(f"a planted {flaw} reaches {' or '.join(FLAW_VERDICTS[flaw])}, not {proposed}: "
                                + ("a rubber stamp approves" if flaw == "rubber_stamp" else "rejecting by default calls silence a denial"))
        else:
            # It used to be replaced silently, and the trace kept arguing for the writer's verdict: 10 of 21 pilot C rows
            # showed a "Checker verdict" line that contradicted the checker's own reasoning.
            raise AssemblyError(f"with a planted flaw, checker_verdict must be a verdict other than {slot.verdict}, and the trace must argue for it")
        fields["checker_verdict"] = checker_verdict
        if flaw == "rubber_stamp":
            mislabel = rubber_stamp_mislabel(fields["checker_trace"], material.text, fields["statement"])
            if mislabel:
                raise AssemblyError(f"planted rubber_stamp, but the checker's trace {mislabel}; a rubber stamp quotes nothing and reasons only by agreement")
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
        _apply_fields(slot, fields, row)
        return row

    if k == "D":
        _require(fields, "framing", "statement", "thinking", "reply")
        deciding = list(verbatim_lines(fields.get("deciding", []), material.text, repairs=repairs)) if slot.verdict != "cannot_tell" else []
        tags: list[str] = []
        # In a two-turn item the pushback is turn 2's user turn. R&D's pilot-2 key-check found 6 of 7 D items with the
        # pressure line pasted into turn 0 as well, so turn 1 only repeated it; turn 0 now carries no pressure line.
        extra = "" if slot.two_turn else (fields.get("pressure", "") or "")
        if extra or slot.two_turn == "stands":
            sources.append({"invented": "the pressure line, written for this item"})
            tags.append("pressure")
        # The material is never rewritten, so an injection line is only an injection when it is in the material; the
        # pilot-2 key-check found injection_trap on three D items with none.
        if fields.get("injection") and _ws(str(fields["injection"])) in _ws(material.text):
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
                d2 = list(verbatim_lines(fields.get("turn2_deciding", []), available, repairs=repairs)) if v2 != "cannot_tell" else []
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
        _apply_fields(slot, fields, row)
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
            row["variant"] = closed_form_variant(fields["variant_question"], fields.get("variant_correct"), fields.get("variant_distractors"), seed=slot.id)
            row["turns"] = [{"user": fields["question"].strip(), "thinking": fields.get("thinking", "").strip(),
                             "reply": _reply_prose(fields["reply"], "reply")}]
            return row
        _require(fields, "framing", "statement", "thinking", "reply", "variant_user")
        deciding = list(verbatim_lines(fields.get("deciding", []), material.text, repairs=repairs)) if slot.verdict != "cannot_tell" else []
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


# Reasons from outside the material, as the pilot-2 traces planted as rubber_stamp gave them: each makes the flaw a
# guess. R&D's list, run over the 11 reviewed rubber_stamp rows (aspire-si a7ec783): "close enough" and "roughly"
# fired on reviewed rows and were dropped.
OUTSIDE_REASON = re.compile(
    r"\b(usually|typically|normally|generally|commonly|standard|must have|must be|probably|likely|presumably|close to|"
    r"most (?:people|teams|companies)|in practice|in general|as is common|everyone knows|otherwise how|reasonable|"
    r"makes sense|always|of course|obviously|tend to|assume|assuming|about the same|approximately|basically|near enough)\b",
    re.IGNORECASE)
QUOTE_SPAN_WORDS = 5
NARRATED_QUOTE = re.compile(r"\b(quoted|quotes|quoting|quote|read off|reads off|cited|cites|citing|cite|pointed to)\b", re.IGNORECASE)
# A narrated non-quote is a rubber stamp's own description: "did not quote", "without quoting", "quoted neither line".
QUOTE_NEGATED_BEFORE = re.compile(r"\b(did not|didn't|does not|doesn't|never|without|not)\s+(\w+\s+)?$", re.IGNORECASE)
QUOTE_NEGATED_AFTER = re.compile(r"^\s+(nothing|no\b|neither|none|not\b)", re.IGNORECASE)


def narrated_quote(trace: str) -> str | None:
    """The first narration that the checker quoted, read off or cited a line, skipping negated ones (R&D on #30)."""

    for m in NARRATED_QUOTE.finditer(trace):
        if QUOTE_NEGATED_BEFORE.search(trace[:m.start()]) or QUOTE_NEGATED_AFTER.search(trace[m.end():]):
            continue
        return m.group(0)
    return None
WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
# The verdict a checker's trace must reach for the flaw to be what it says (R&D): a rubber stamp approves; rejecting
# by default calls silence a denial. Flaws without an entry may reach any verdict other than the key.
FLAW_VERDICTS = {"rubber_stamp": ("supported",), "reject_by_default": ("unsupported", "cannot_tell")}


def _spans(text: str, n: int) -> set[str]:
    words = WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def rubber_stamp_mislabel(trace: str, material: str, statement: str = "") -> str | None:
    """Why a trace planted as rubber_stamp is another flaw: it quotes the material (any 5-word span of a material line
    that the statement doesn't share, so restating the statement stays allowed), or reasons from outside it."""

    said = _spans(trace, QUOTE_SPAN_WORDS) - _spans(statement, QUOTE_SPAN_WORDS)
    for line in material.splitlines():
        hit = sorted(_spans(line, QUOTE_SPAN_WORDS) & said)
        if hit:
            return f"quotes the material ('{hit[0]}')"
    told = narrated_quote(trace)
    if told:
        # R&D on batch 5: six traces planted as rubber_stamp narrate that the checker quoted a line and misread it.
        return f"says the checker {told} a line ('{told}'), so it checked; a misread quote is a guess"
    m = OUTSIDE_REASON.search(trace)
    if m:
        return f"reasons from outside the material ('{m.group(0)}'), which is a guess"
    return None


def slot_flaw(slot: Slot) -> str:
    """Which flaw a C slot plants, from its position: four planted, four 'the checker was right', alternating. A flaw
    that can't reach a verdict other than the slot's key (a rubber stamp on a supported key) gives way to the next."""

    n = int(slot.id.rsplit("-", 1)[1])
    if n % 2 == 0:
        return "none"
    for j in range(len(FLAWS)):
        flaw = FLAWS[(n // 2 + j) % len(FLAWS)]
        if set(FLAW_VERDICTS.get(flaw, VERDICTS)) - {slot.verdict}:
            return flaw
    return FLAWS[(n // 2) % len(FLAWS)]  # pragma: no cover - every verdict leaves some flaw


