"""The runner: it suggests, it keeps the builder inside the role-set, and it keeps diagnostics.

One slot at a time per worker: material fixed, the writer asked for fields, the
row assembled, the mechanical truth checks, the gates' per-item checks, the
critic with the key hidden, then, under one lock, the exclusivity checks against
everything accepted so far. A refusal comes back as a plain-words suggestion with
the governing rule; after two replacements the slot is dropped. The run halts on
the drop-rate andon. Every event goes to status.json; every item gets a receipt.
Nothing is written to the output file until it has passed everything.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from role_set.assemble import AssemblyError, Material, assemble, slot_flaw
from role_set.critic import compare_answer, compare_flaw, compare_trait, compare_verdict, flaw_messages, trait_messages, verdict_messages
from role_set.gates import Gates
from role_set.plan import Slot
from role_set.prompts import kind_key, material_messages, parse_json_object, warmup_messages, writer_messages
from role_set.transport import TransportError

MAX_ATTEMPTS = 3  # the first draft, then two replacement requests
ANDON_DROP_SHARE = 0.20
ANDON_MIN_ATTEMPTS = 10  # the share means nothing on a handful of slots; the per-lesson cap covers a run's start
ANDON_LESSON_DROPS = 3  # ends that lesson; the run moves on to the next one
# The whole run stops only on a systemic signal: two lessons in a row halting on the same refusal code, or more
# than 20% of the last 10 writer calls failing to parse or reach the writer (the maintainer's rule for paid runs).
SYSTEMIC_WINDOW = 10
SYSTEMIC_RATE = 0.20
SYSTEMIC_CODES = ("PARSE", "TRANSPORT")
# check_against codes whose message names spent wording; a retry carries that wording as a do-not-use list.
SPENT_CODES = ("SPAN_REUSED", "B_PHRASE_REUSED", "OPENING_REUSED", "FRAMING_REUSED")
WARMUP_MISSES = 2

# Refusal code -> what the builder is told, with the governing rule. Pinned by a test.
SUGGESTIONS = {
    "ASSEMBLY": "{detail} (the kinds table and the schema).",
    "TRUTH_QUOTE": "The thinking quotes '{detail}', which is not in the material. Quote lines exactly or don't quote (rule: every thinking step is true).",
    "TRUTH_NUMBER": "The thinking states the number {detail}, which is in neither the material, the statement nor the computed facts. Use only given numbers (handoff rule 3).",
    "GATE": "{detail} (the validator refused it).",
    "ROLE_WORDS": "The prompt contains {detail}. User turns of C, D and verdict quizzes name no role, trait or pattern (amendment 3).",
    "VERDICT_MISMATCH": "A second reader, given only the material and the statement, reached a different verdict: {detail}. Make the statement settle cleanly on the slot's verdict (instant agreement).",
    "EVIDENCE_MISMATCH": "A second reader found different deciding lines: {detail}. Quote the lines that settle every clause of the statement, whole rows and headings included (amendment 1).",
    "FLAW_MISMATCH": "A second reader named a different flaw in the checker's trace: {detail}. Make the planted flaw the one obvious thing wrong (instant agreement).",
    "ANSWER_MISMATCH": "A second solver reached a different answer: {detail}. Close the puzzle so only one answer fits (rule: puzzles are closed).",
    "TRAIT_MISMATCH": "The item reads as another trait: {detail}. Make this trait, not general care, decide the named step (the trait lesson).",
    "VOID": "The second reader's answer was unusable ({detail}); it was asked again.",
    "AGAINST": "{detail}",
    "PARSE": "The reply was not one JSON object with the required fields: {detail}.",
    "TRANSPORT": "The writer could not be reached ({detail}); the slot ends.",
    "LESSON_COPY": "The text copies the trait lesson's wording ('{detail}'). Say it in your own words (handoff rule 11: never copy a sentence from the lessons).",
}


class RunHalted(RuntimeError):
    """The andon fired. The run stops and reports; the pod is the human's to shut down."""


@dataclass
class Status:
    started_at: float
    accepted: int = 0
    refused: dict = field(default_factory=dict)
    dropped: int = 0
    attempted: int = 0
    critic_agree: int = 0
    critic_asked: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    concurrency: int = 1
    actions: list = field(default_factory=list)
    halted: str | None = None  # a systemic stop: the whole run
    lesson_halts: dict = field(default_factory=dict)  # lesson -> {"reason", "kind", "rule"}: that lesson stopped, the run went on
    kinds_off: dict = field(default_factory=dict)  # kind -> rule: switched off after two lessons in a row halted on it
    skipped: dict = field(default_factory=dict)  # reason -> slots never started

    def as_dict(self, now: float) -> dict:
        share = self.dropped / self.attempted if self.attempted else 0.0
        return {
            "accepted": self.accepted,
            "refused_by_code": dict(sorted(self.refused.items())),
            "dropped": self.dropped,
            "attempted": self.attempted,
            "drop_share": round(share, 3),
            "andon_distance": round(ANDON_DROP_SHARE - share, 3),
            "critic_agreement": round(self.critic_agree / self.critic_asked, 3) if self.critic_asked else None,
            "tokens": {"prompt": self.tokens_prompt, "completion": self.tokens_completion},
            "elapsed_s": round(now - self.started_at, 1),
            "concurrency": self.concurrency,
            "halted": self.halted,
            "lesson_halts": self.lesson_halts,
            "kinds_off": self.kinds_off,
            "skipped": self.skipped,
            "actions": self.actions[-200:],
        }


@dataclass
class Outcome:
    slot_id: str
    accepted: bool
    attempts: int
    refusals: list = field(default_factory=list)  # [{"code","detail","suggestion"}]
    row: dict | None = None
    receipt: dict | None = None
    skipped: bool = False  # never started: the run, the lesson or the kind had stopped, or the time budget ran out
    skip_reason: str | None = None


def _quoted_spans(text: str) -> list[str]:
    spans = re.findall(r"\[([^\[\]]{12,})\]", text) + re.findall(r"[\"“]([^\"”]{12,})[\"”]", text)
    return [s.strip() for s in spans]


LESSON_SPAN_WORDS = 8  # the gates' span length


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def lesson_spans(lesson_text: str) -> set[str]:
    """Every 8-word span of a trait lesson, across line breaks, in the gates' tokenisation."""

    w = _words(lesson_text)
    return {" ".join(w[i:i + LESSON_SPAN_WORDS]) for i in range(len(w) - LESSON_SPAN_WORDS + 1)}


def lesson_copy(row: dict, spans: set[str]) -> str | None:
    """The first 8-word span (in sorted order) that an item's user turns, thinking or replies share with its trait
    lesson, outside material lines, or None. Handoff rule 11 (fresh wording) had no check behind it until the
    2026-10-11 rehearsal, where the writer copied a lesson sentence into every kind-A passage."""

    if not spans:
        return None
    content = {ln.strip() for ln in (row.get("material") or "").splitlines() if ln.strip()}
    found: set[str] = set()
    for t in row["turns"]:
        content |= {x.strip() for x in t.get("new_material", []) if x.strip()}
        for text in (t["user"], t["thinking"], t["reply"]):
            for line in text.splitlines():
                if line.strip() in content:
                    continue
                w = _words(line)
                found |= {" ".join(w[i:i + LESSON_SPAN_WORDS]) for i in range(len(w) - LESSON_SPAN_WORDS + 1)} & spans
    return sorted(found)[0] if found else None


def refusal_rule(refusal: dict) -> str:
    """The specific rule behind a refusal, for halt keys: a check_against refusal's gate code, a gate refusal's
    message with its line, ids, lesson and quoted text stripped, or the bare code for everything else."""

    code, detail = refusal.get("code", ""), refusal.get("detail") or ""
    if code == "AGAINST":
        return detail.split(":")[0]
    if code == "GATE":
        d = re.sub(r"^line \d+ \([^)]*\):\s*", "", detail)
        d = re.sub(r"(?<=[\s(])'[^']*'(?=[\s),:;.]|$)|\"[^\"]*\"", "''", d)  # quoted values, not apostrophes
        d = re.sub(r"\(T\d\d\)|\bT\d\d(-[A-E]-\d\d)?\b", "", d)
        return "GATE: " + " ".join(d.split())[:80]
    return code


def trait_words_note(words: list[str], spent: list[str]) -> str:
    """For kind A: the trait words stay required on every retry. Rehearsal 11e: told that a span holding the trait
    words was spent, the writer dropped the trait name and was refused for not naming it. When a spent span holds a
    trait word, the note says which words are required and that only the wording around them is spent."""

    if not words:
        return ""
    note = " Keep naming the trait with one of these exact words, which are required: " + ", ".join(f"'{w}'" for w in words) + "."
    held = [x for x in dict.fromkeys(spent) if any(w in x.lower() for w in words)]
    if held:
        note += (" In " + "; ".join(f"'{x}'" for x in held) + " only the wording around the trait words is spent:"
                 " reword that, and keep the trait words.")
    return note


def spent_wording(refusal: dict) -> list[str]:
    """The wording a refusal names as spent ("the wording 'x' is already in 2 items", or a copied lesson span), or []."""

    detail = refusal.get("detail") or ""
    if refusal.get("code") == "LESSON_COPY":
        return [detail] if detail else []
    if refusal.get("code") != "AGAINST" or not detail.startswith(SPENT_CODES):
        return []
    # The gates' messages put the wording between quotes before "is already" / "already opens"; the wording itself
    # may hold an apostrophe, so anchor on the message, not on the quote marks.
    m = re.search(r"'(.+)' (?:is already|already opens)", detail)
    return [m.group(1)] if m and m.group(1).strip() else []


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.:]*\d|\d", text))


# Typographic quotes in the writer's own prose become plain ones before assembly and gating, so a correct trait
# name ("don’t fix") is never refused over a glyph. Verbatim material lines (deciding, new material) are left alone.
GLYPHS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
VERBATIM_FIELDS = ("deciding", "turn2_deciding", "turn2_new_material")


def _plain(text: str) -> str:
    return text.translate(GLYPHS)


def normalize_glyphs(fields: dict) -> tuple[dict, int]:
    """(fields with typographic quotes made plain outside the verbatim fields, how many characters changed)."""

    count = 0

    def fix(v):
        nonlocal count
        if isinstance(v, str):
            out = _plain(v)
            count += sum(1 for a, b in zip(v, out) if a != b)
            return out
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v

    return {k: (v if k in VERBATIM_FIELDS else fix(v)) for k, v in fields.items()}, count


def truth_checks(row: dict, facts: dict | None) -> tuple[str, str] | None:
    """Every bracketed or quoted span in the thinking is verbatim in the material; every number is given."""

    material = row.get("material") or ""
    if not material:
        return None
    flat = _plain(" ".join(material.split()))
    statement = _plain("\n".join(t["user"] for t in row["turns"]))
    allowed = _numbers(material) | _numbers(statement) | _numbers(json.dumps(facts or {}))
    for t in row["turns"]:
        for span in _quoted_spans(t["thinking"]):
            span = _plain(span)
            if " ".join(span.split()) not in flat and " ".join(span.split()) not in " ".join(statement.split()):
                return ("TRUTH_QUOTE", span[:60])
        for n in _numbers(t["thinking"]):
            if n not in allowed and not re.fullmatch(r"[1-9]", n):  # step numbers 1-9 are not claims
                return ("TRUTH_NUMBER", n)
    return None


def _teaching_text(row: dict) -> str:
    parts = []
    for t in row["turns"]:
        parts.append(t.get("thinking", ""))
        parts.append(t.get("reply", ""))
    for k in ("key_points", "quiz", "variant"):
        if k in row:
            parts.append(json.dumps(row[k], ensure_ascii=False))
    return "\n".join(parts)


class Runner:
    def __init__(
        self,
        *,
        gates: Gates,
        pack: dict[str, str],
        trait_names: dict[str, str],
        writer,  # callable(messages, seed) -> Completion
        critic,  # callable(messages) -> Completion, or None to skip (tests and dry runs only)
        out_dir: Path,
        exemplars: dict[str, list[dict]] | None = None,
        excerpts: dict[str, list[Material]] | None = None,
        concurrency: int = 1,
        pins: dict | None = None,
        clock=time.time,
        flaw_gate: bool = False,  # the critic's flaw check refuses only once a control run shows it agrees
        stuck=None,  # callable(thinking, material, tier, tokens) -> the frozen stuck rule's measures, or None
    ) -> None:
        self.gates, self.pack, self.trait_names = gates, pack, trait_names
        self.writer, self.critic = writer, critic
        self.out_dir = out_dir
        self.exemplars = exemplars or {}
        self.excerpts = {k: list(v) for k, v in (excerpts or {}).items()}
        self.pins = pins or {}
        self.clock = clock
        self.status = Status(started_at=clock(), concurrency=max(1, concurrency))
        self.accepted: list[dict] = []
        self.lesson_drops: dict[str, int] = {}
        self.batch_check: str | None = None  # "clean" or "REFUSED: ..." once a run finishes unhalted
        self.lock = threading.RLock()  # re-entrant: logging runs both inside and outside the accept lock
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.avoid: list[str] = []
        self.flaw_gate = flaw_gate
        self._spans_by_lesson: dict[str, set[str]] = {}
        self.stuck = stuck
        self.lesson_codes: dict[str, Counter] = {}
        self.recent_calls: deque = deque(maxlen=SYSTEMIC_WINDOW)
        self.halt_order: list[tuple[str, tuple[str, str]]] = []
        self.lesson_drop_keys: dict[str, Counter] = {}
        self.plan_kinds: set[str] = set()
        self.deadline: float | None = None
        self.finished: list[Outcome] = []  # every outcome as it lands, so a halted run still reports its slots

    def _lesson_spans(self, lesson: str) -> set[str]:
        if lesson not in self._spans_by_lesson:
            self._spans_by_lesson[lesson] = lesson_spans(self.pack.get(f"traits/{lesson}.md", ""))
        return self._spans_by_lesson[lesson]

    # ---- status and log -------------------------------------------------------------------------------------
    def _log(self, action: str, **kw) -> None:
        entry = {"t": round(self.clock() - self.status.started_at, 1), "action": action, **kw}
        self.status.actions.append(entry)
        # The full log, every action and halt with its reason; status.json keeps only the latest 200.
        with (self.out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._write_status()

    def _append(self, name: str, record: dict) -> None:
        with self.lock, (self.out_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _note_refusal(self, slot: Slot, attempt: int, refusal: dict, *, draft: str) -> None:
        self.status.refused[refusal["code"]] = self.status.refused.get(refusal["code"], 0) + 1
        self.lesson_codes.setdefault(slot.lesson, Counter())[refusal["code"]] += 1
        self._log("refused", slot=slot.id, attempt=attempt, code=refusal["code"], detail=refusal["detail"][:200])
        # Every refused draft is kept, so a failure can be read afterwards (the third rehearsal's GATE refusals
        # could not be diagnosed because only accepted items were written).
        self._append("refusals.jsonl", {"slot": slot.id, "attempt": attempt, "code": refusal["code"], "detail": refusal["detail"], "draft": draft})

    def _record_call(self, slot: Slot, attempt: int, c, outcome: str, glyphs: int, material) -> None:
        """One line per writer call, on both paths: the raw draft, tokens, latency, the outcome, glyphs normalised, and
        the frozen stuck rule's measures on the draft's thinking. Then the systemic check over the last 10 calls."""
        text = c.text if c is not None else ""
        record = {"slot": slot.id, "lesson": slot.lesson, "kind": slot.kind, "attempt": attempt, "outcome": outcome,
                  "prompt_tokens": getattr(c, "prompt_tokens", None), "completion_tokens": getattr(c, "completion_tokens", None),
                  "seconds": round(getattr(c, "seconds", 0.0) or 0.0, 2), "glyphs_normalized": glyphs, "draft": text}
        if self.stuck is not None and text:
            try:
                thinking = str(parse_json_object(text).get("thinking") or "")
            except ValueError:
                thinking = text
            record["stuck"] = self.stuck(thinking, material.text if material else "", slot.tier, getattr(c, "completion_tokens", 0) or 0)
        self._append("drafts.jsonl", record)
        with self.lock:
            self.recent_calls.append(outcome)
            bad = sum(1 for x in self.recent_calls if x in SYSTEMIC_CODES)
            if len(self.recent_calls) == SYSTEMIC_WINDOW and bad > SYSTEMIC_RATE * SYSTEMIC_WINDOW and self.status.halted is None:
                self.status.halted = f"systemic: {bad} of the last {SYSTEMIC_WINDOW} writer calls failed to parse or reach the writer"
                self._log("halt", reason=self.status.halted)
        if self.status.halted is not None and self.status.halted.startswith("systemic: ") and "writer calls" in self.status.halted:
            raise RunHalted(self.status.halted)

    def _write_status(self) -> None:
        (self.out_dir / "status.json").write_text(json.dumps(self.status.as_dict(self.clock()), indent=1), encoding="utf-8")

    def _count_tokens(self, completion) -> None:
        self.status.tokens_prompt += completion.prompt_tokens or 0
        self.status.tokens_completion += completion.completion_tokens or 0

    # ---- warm-up ----------------------------------------------------------------------------------------------
    def warmup(self, lesson: str, sides: tuple[str, str]) -> bool:
        """The builder restates the trait before it writes. Two misses stop the trait."""

        lesson_text = self.pack.get(f"traits/{lesson}.md", "")
        must = {w for w in re.findall(r"[a-z]{4,}", (self.trait_names.get(lesson, "") + " " + " ".join(sides)).lower())}
        misses = 0
        for attempt in range(1, WARMUP_MISSES + 1):
            c = self.writer(warmup_messages(self.pack, lesson), seed=attempt)
            self._count_tokens(c)
            try:
                answer = parse_json_object(c.text)
                text = " ".join(str(answer.get(k, "")) for k in ("main_point", "counterweight", "deciding_case")).lower()
            except ValueError:
                text = ""
            hit = {w for w in must if w in text}
            if len(hit) >= max(1, len(must) // 3) and all(answer.get(k) for k in ("main_point", "counterweight", "deciding_case")):
                self._log("warmup_pass", lesson=lesson, attempt=attempt, terms=sorted(hit))
                (self.out_dir / f"warmup-{lesson}.json").write_text(json.dumps({"attempt": attempt, "answer": answer, "lesson_chars": len(lesson_text)}, indent=1), encoding="utf-8")
                return True
            misses += 1
            self._log("warmup_miss", lesson=lesson, attempt=attempt, missing=sorted(must - hit)[:6])
        return False

    # ---- one slot ---------------------------------------------------------------------------------------------
    def _material_for(self, slot: Slot) -> Material | None:
        if slot.kind in ("A", "B") or (slot.kind == "E" and slot.e_type == "written"):
            return None
        kind = slot.content_kind or "notice"
        if slot.real_source and self.excerpts.get(kind):
            return self.excerpts[kind].pop(0)
        needs = f"It must be the kind of text a verdict of '{slot.verdict}' can be reached about for one plain statement."
        c = self.writer(material_messages(slot, needs), seed=0)
        self._count_tokens(c)
        text = c.text.strip()
        if not text or len(text) < 20:
            raise AssemblyError("the drafted material came back empty")
        return Material(text + "\n", {"invented": f"{kind.replace('_', ' ')} material, drafted for this item and fixed before writing"})

    def _refusal(self, code: str, detail: str) -> dict:
        tmpl = SUGGESTIONS.get(code, "{detail}")
        return {"code": code, "detail": detail, "suggestion": tmpl.format(detail=detail)}

    def _critic_pass(self, slot: Slot, row: dict, fields: dict) -> dict | None:
        if self.critic is None:
            return None
        material = row.get("material") or ""
        verdicts = []
        if slot.kind in ("C", "D") or (slot.kind == "E" and slot.e_type == "verdict"):
            statement = fields["statement"]
            # Each verdict-giving turn is checked with the material available at that turn.
            turns: list[tuple[str, str, list[str]]] = []
            if slot.kind == "E":
                turns.append((material, row["quiz"]["verdict"], row["quiz"]["deciding"]))
            elif row.get("turn_keys"):
                first = row["turn_keys"]["1"]
                turns.append((material, first["verdict"], first["deciding"]))
                new_lines = row["turns"][-1].get("new_material") or []
                turns.append((material + "\n" + "\n".join(new_lines), row["verdict"], row["deciding"]))
            else:
                turns.append((material, row["verdict"], row["deciding"]))
            for seen_material, key_verdict, key_deciding in turns:
                for ask in range(2):
                    c = self.critic(verdict_messages(seen_material, statement))
                    self._count_tokens(c)
                    v = compare_verdict(self.gates, c.text, key_verdict, key_deciding, seen_material)
                    if v.code != "VOID":
                        break
                    self._log("critic_void", slot=slot.id, detail=v.detail)
                verdicts.append(v)
                if not v.ok:
                    break
            if slot.kind == "C":
                c = self.critic(flaw_messages(material, statement, fields["checker_trace"], row["checker_verdict"]))
                self._count_tokens(c)
                f = compare_flaw(c.text, row["planted_flaw"])
                if f.ok or self.flaw_gate:
                    verdicts.append(f)
                else:
                    # Advisory until a control run shows the critic agrees on flaws (Window A: 9 of 24): logged, counted, never a refusal.
                    self.status.critic_asked += 1
                    self._log("flaw_advisory", slot=slot.id, detail=f.detail)
        elif slot.kind == "B":
            c = self.critic([{"role": "system", "content": "Solve the puzzle. Answer with the answer only."}, {"role": "user", "content": row["turns"][0]["user"]}])
            self._count_tokens(c)
            verdicts.append(compare_answer(c.text, row["answer"]))
        c = self.critic(trait_messages(_teaching_text(row), self.trait_names))
        self._count_tokens(c)
        verdicts.append(compare_trait(c.text, slot.lesson))
        for v in verdicts:
            self.status.critic_asked += 1
            if v.ok:
                self.status.critic_agree += 1
            else:
                return self._refusal(v.code, v.detail)
        return None

    def run_slot(self, slot: Slot, coverage_notes: list[str] | None = None) -> Outcome:
        outcome = Outcome(slot_id=slot.id, accepted=False, attempts=0)
        trait = self.trait_names.get(slot.lesson, slot.lesson)
        try:
            material = self._material_for(slot)
        except AssemblyError as e:
            outcome.refusals.append(self._refusal("ASSEMBLY", str(e)))
            return outcome
        exemplars = [e for k, v in self.exemplars.items() if k != slot.lesson for e in v if e.get("kind") == slot.kind][:3]
        facts = {"material_line_count": len(material.text.splitlines())} if material else None
        messages = writer_messages(slot, self.pack, material_text=material.text if material else None, exemplars=exemplars, avoid=list(self.avoid[-60:]), facts=facts, coverage_notes=coverage_notes,
                                   trait_words=self.gates.trait_words(slot.lesson))
        spent: list[str] = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            outcome.attempts = attempt
            try:
                c = self.writer(messages, seed=attempt)
            except TransportError as e:
                # A writer that can't be reached ends this slot, never the run by itself; the systemic rule counts it.
                refusal = self._refusal("TRANSPORT", str(e)[:200])
                outcome.refusals.append(refusal)
                self._note_refusal(slot, attempt, refusal, draft="")
                self._record_call(slot, attempt, None, "TRANSPORT", 0, material)
                return outcome
            self._count_tokens(c)
            refusal = None
            row = None
            glyphs = 0
            try:
                fields, glyphs = normalize_glyphs(parse_json_object(c.text))
            except ValueError as e:
                refusal = self._refusal("PARSE", str(e))
            if refusal is None:
                if slot.kind == "C" and "planted_flaw" not in fields:
                    fields["planted_flaw"] = slot_flaw(slot)
                try:
                    row = assemble(slot, trait, fields, material)
                except AssemblyError as e:
                    refusal = self._refusal("ASSEMBLY", str(e))
            if refusal is None:
                t = truth_checks(row, facts)
                if t:
                    refusal = self._refusal(*t)
            if refusal is None:
                copied = lesson_copy(row, self._lesson_spans(slot.lesson))
                if copied:
                    refusal = self._refusal("LESSON_COPY", copied)
            if refusal is None:
                try:
                    self.gates.check_item(row)
                except self.gates.error_type() as e:
                    refusal = self._refusal("GATE", str(e))
            if refusal is None:
                words = self.gates.role_words_in_prompt(row)
                if words:
                    refusal = self._refusal("ROLE_WORDS", ", ".join(words))
            if refusal is None:
                refusal = self._critic_pass(slot, row, fields)
            if refusal is None:
                with self.lock:
                    try:
                        reasons = self.gates.check_against(row, self.accepted)
                    except self.gates.error_type() as e:
                        reasons = [("AGAINST", str(e))]
                    if reasons:
                        code, msg = reasons[0]
                        refusal = self._refusal("AGAINST", f"{code}: {msg}")
                    else:
                        self._accept(slot, row, c, attempt, glyphs=glyphs)
                        outcome.accepted, outcome.row = True, row
                        self._record_call(slot, attempt, c, "accepted", glyphs, material)
                        return outcome
            outcome.refusals.append(refusal)
            self._note_refusal(slot, attempt, refusal, draft=c.text)
            self._record_call(slot, attempt, c, refusal["code"], glyphs, material)
            spent += spent_wording(refusal)
            if spent:
                with self.lock:
                    self.avoid.extend(x for x in spent if x not in self.avoid)
            ask = "Refused: " + refusal["suggestion"]
            if spent:
                # The 8B rehearsal repeated the same span on every retry when told only the reason (Window B).
                ask += " Do not use any of these wordings, in any order or form: " + "; ".join(f"'{x}'" for x in dict.fromkeys(spent)) + "."
            ask += trait_words_note(self.gates.trait_words(slot.lesson) if slot.kind == "A" else [], spent)
            messages = messages + [
                {"role": "assistant", "content": c.text},
                {"role": "user", "content": ask + " Return a corrected JSON object with all fields."},
            ]
        return outcome

    def _accept(self, slot: Slot, row: dict, completion, attempt: int, glyphs: int = 0) -> None:
        self.accepted.append(row)
        self.status.accepted += 1
        framing = row["turns"][0]["user"].split("\n", 1)[0]
        if framing and slot.kind in ("C", "D", "E"):
            self.avoid.append(framing)
        receipt = {
            "slot": slot.as_dict(),
            "writer_model": completion.model,
            "attempt": attempt,
            "gates": {"version": self.gates.version, "sha256": self.gates.sha256},
            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest(),
            "glyphs_normalized": glyphs,
            **self.pins,
        }
        with (self.out_dir / "accepted.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (self.out_dir / "receipts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        self._log("accepted", slot=slot.id, attempt=attempt)

    # ---- the batch --------------------------------------------------------------------------------------------
    def _andon(self, slot: Slot, out: Outcome | None = None) -> None:
        """A drop counts toward its lesson. Three drops end that lesson (its remaining slots are skipped) and the run
        moves on. The lesson's halt key is the (kind, specific rule) most of its drops fell on."""

        self.status.dropped += 1
        self.lesson_drops[slot.lesson] = self.lesson_drops.get(slot.lesson, 0) + 1
        keys = self.lesson_drop_keys.setdefault(slot.lesson, Counter())
        for r in (out.refusals if out else []):
            keys[(slot.kind, refusal_rule(r))] += 1
        if slot.lesson not in self.status.lesson_halts and self.lesson_drops[slot.lesson] >= ANDON_LESSON_DROPS:
            key = keys.most_common(1)[0][0] if keys else (slot.kind, "NONE")
            self._halt_lesson(slot.lesson, f"{ANDON_LESSON_DROPS} dropped slots", key)

    def _halt_lesson(self, lesson: str, reason: str, key: tuple[str, str]) -> None:
        """Two lessons in a row halting on the same (kind, rule) switch that kind off for the remaining lessons; the
        other kinds carry on. The run stops only when every kind in the plan is off (or on the PARSE/TRANSPORT rate)."""

        kind, rule = key
        self.status.lesson_halts[lesson] = {"reason": reason, "kind": kind, "rule": rule}
        self.halt_order.append((lesson, key))
        self._log("lesson_halt", lesson=lesson, reason=reason, kind=kind, rule=rule)
        if len(self.halt_order) >= 2 and self.halt_order[-1][1] == self.halt_order[-2][1]:
            switched = sorted(self.plan_kinds) if kind == "*" else [kind]
            for k in switched:
                if k not in self.status.kinds_off:
                    self.status.kinds_off[k] = rule
                    self._log("kind_off", kind=k, rule=rule, lessons=[self.halt_order[-2][0], lesson])
            if self.plan_kinds and self.plan_kinds <= set(self.status.kinds_off) and self.status.halted is None:
                self.status.halted = "systemic: every kind is switched off (" + "; ".join(f"{k}: {r}" for k, r in sorted(self.status.kinds_off.items())) + ")"
                self._log("halt", reason=self.status.halted)
                raise RunHalted(self.status.halted)

    def _skip(self, slot: Slot) -> str | None:
        if self.status.halted is not None:
            return "run halted"
        if slot.kind in self.status.kinds_off:
            return f"kind {slot.kind} systemic: {self.status.kinds_off[slot.kind]}"
        if slot.lesson in self.status.lesson_halts:
            return f"lesson {slot.lesson} halted"
        if self.deadline is not None and self.clock() >= self.deadline:
            return "time budget"
        return None

    def run(self, slots: list[Slot], sides: dict[str, tuple[str, str]], *, stop_after_s: float | None = None) -> list[Outcome]:
        outcomes: list[Outcome] = []
        if stop_after_s is not None:
            self.deadline = self.clock() + stop_after_s  # no new slot starts after this; running ones finish
        self.plan_kinds = {s.kind for s in slots}
        lessons = []
        for s in slots:
            if s.lesson not in lessons:
                lessons.append(s.lesson)
        try:
            for lesson in lessons:
                if not self.warmup(lesson, sides.get(lesson, ("", ""))):
                    self._halt_lesson(lesson, "warm-up failed: the builder could not restate the trait", ("*", "WARMUP"))
        except RunHalted:
            self._finish(outcomes, lessons)
            raise

        def one(slot: Slot) -> Outcome:
            with self.lock:
                why = self._skip(slot)
                if why:
                    self.status.skipped[why] = self.status.skipped.get(why, 0) + 1
                    return Outcome(slot_id=slot.id, accepted=False, attempts=0, skipped=True, skip_reason=why)
                self.status.attempted += 1
            out = self.run_slot(slot)
            with self.lock:
                self.finished.append(out)
                if not out.accepted:
                    self._log("dropped", slot=slot.id, codes=[r["code"] for r in out.refusals])
                    self._andon(slot, out)
            return out

        try:
            if self.status.concurrency == 1:
                for s in slots:
                    outcomes.append(one(s))
            else:
                with ThreadPoolExecutor(max_workers=self.status.concurrency) as pool:
                    outcomes = list(pool.map(one, slots))
        finally:
            self._finish(outcomes or self.finished, lessons)
        return outcomes

    def _finish(self, outcomes: list[Outcome], lessons: list[str]) -> None:
        # The share-based gates (floors, pairs, multi-line share, phrases, formulas) need finished lessons, so they run
        # once here, over the lessons that were neither halted nor cut short. A refusal is reported, not retried.
        whole = [ls for ls in lessons if ls not in self.status.lesson_halts]
        rows = [r for r in self.accepted if r.get("lesson") in whole]
        if self.status.halted is None and not self.status.skipped and rows:
            try:
                self.gates.check_batch(rows)
                self.batch_check = "clean"
            except self.gates.error_type() as e:
                self.batch_check = f"REFUSED: {e}"
            self._log("batch_check", result=self.batch_check, lessons=whole)
        self._log("end", reminder="shut the pod down with offrig; the window closes 30 minutes after this run")
        report = {
            "batch_check": self.batch_check,
            "status": self.status.as_dict(self.clock()),
            "outcomes": [{"slot": o.slot_id, "accepted": o.accepted, "attempts": o.attempts, "refusals": o.refusals, "skipped": o.skipped, "skip_reason": o.skip_reason} for o in outcomes],
        }
        (self.out_dir / "batch-report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
