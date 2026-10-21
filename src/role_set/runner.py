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
from role_set.critic import (TRAIT_TEMPERATURE, TRAIT_VOTES, answer_messages, compare_answer, compare_flaw,
                             compare_trait_votes, compare_twin_verdict, compare_verdict, flaw_messages, teaching_text,
                             trait_messages, verdict_messages)
from role_set import solvefirst
from role_set.frames import check_frame
from role_set.solvefirst import SolveFirstError
from role_set.gates import Gates
from role_set.plan import Slot
from role_set.puzzle import PuzzleError, check_puzzle
from role_set.prompts import FIELDS, kind_key, material_messages, parse_json_object, warmup_messages, writer_messages
from role_set.trait_steps import TRAIT_STEPS
from role_set.transport import TransportError

MAX_ATTEMPTS = 3  # the first draft, then two replacement requests
# A B puzzle whose asked answer follows from its 'is' facts alone is a lookup, not a puzzle: one per lesson (R&D, PR H).
LOOKUP_CAP = 1
# A solve-first generator retries on any lookup, so a lookup on a locked puzzle is a defect: its cap is 0 (Argon §9.4 #7,
# R&D 2026-10-17). Freehand slots keep LOOKUP_CAP.
LOOKUP_CAP_SOLVE_FIRST = 0


def lookup_cap(slot) -> int:
    return LOOKUP_CAP_SOLVE_FIRST if getattr(slot, "b_path", None) == "solve_first" else LOOKUP_CAP
LOOKUP_FULL = "the asked answer follows from direct facts alone, a lookup, and this lesson already has its one; make the answer need elimination"
LOOKUP_NONE = "the asked answer follows from direct facts alone, a lookup, and a solve-first puzzle takes none; make the answer need elimination"


def lookup_full(slot) -> str:
    return LOOKUP_NONE if lookup_cap(slot) == 0 else LOOKUP_FULL


ANDON_DROP_SHARE = 0.20
ANDON_MIN_ATTEMPTS = 10  # the share means nothing on a handful of slots; the per-lesson cap covers a run's start
ANDON_CELL_DROPS = 3  # ends that (lesson, kind) cell; the lesson's other kinds and the run go on
# The whole run stops only on a systemic signal: every kind switched off (two cells of different lessons in a row
# halting on the same kind and rule switch that kind off), or more
# than 20% of the last 20 writer calls failing to parse or reach the writer (the maintainer's rule for paid runs).
# The window was 10 until pilot 2: at concurrency 8, three cut-off drafts from one kind filled it in seven minutes.
# A draft cut off at max_tokens is TRUNCATED, not PARSE: it is a defect of that cell's drafts, so it counts toward
# the cell's drops and never toward the systemic rate.
SYSTEMIC_WINDOW = 20
SYSTEMIC_RATE = 0.20
SYSTEMIC_CODES = ("PARSE", "TRANSPORT")
# check_against codes whose message names spent wording; a retry carries that wording as a do-not-use list.
SPENT_CODES = ("SPAN_REUSED", "B_PHRASE_REUSED", "OPENING_REUSED", "FRAMING_REUSED")
WARMUP_MISSES = 2
WARMUP_PARTS = ("main_point", "counterweight", "deciding_case")
# Words of a trait name that don't identify the trait ("knows your limits" is identified by "limits").
WARMUP_FILLER = {"you", "your", "are", "the", "and", "don't", "doesn't", "doesn", "does", "goes", "keeps", "takes", "knows",
                 "makes", "what", "things", "just", "about", "from", "into", "round", "take", "material"}


REPLY_OPENING_WORDS = 6
REPLY_OPENING_MAX = 2  # accepted items of a lesson that may share a reply's opening words, as the gates allow for C/D/E prompts


def reply_opening_reused(row: dict, accepted: list[dict]) -> str | None:
    """The first six words of a kind-A reply, when two accepted kind-A items of the lesson already open that way. The
    gates check the openings of C, D and E prompts; R&D found 21 of the pilot's 46 kind-A replies opening "As a verifier, I"."""

    def opening(r: dict) -> str:
        words = re.findall(r"[a-z0-9']+", (r.get("turns") or [{}])[0].get("reply", "").lower())
        return " ".join(words[:REPLY_OPENING_WORDS]) if len(words) >= REPLY_OPENING_WORDS else ""

    if row.get("kind") != "A":
        return None
    op = opening(row)
    if op and sum(1 for a in accepted if a.get("lesson") == row.get("lesson") and a.get("kind") == "A" and opening(a) == op) >= REPLY_OPENING_MAX:
        return op
    return None


def row_shaped(fields: dict, slot: Slot) -> bool:
    """The writer copied an exemplar's finished-row shape instead of returning the fields: 45 of the 235B pilot's kind-B
    drafts came back as {kind, answer, turns}."""

    wanted = FIELDS[kind_key(slot)]
    return any(k in fields for k in ("turns", "kind")) and "turns" not in wanted and not all(k in fields for k in wanted)


WARMUP_LESSON_TERMS = 2  # content words of the lesson's teaching paragraph a restatement must use, if it doesn't name the trait
# Words every lesson uses, which say nothing about which trait a restatement is about.
LESSON_COMMON = {"verifier", "verifiers", "statement", "statements", "material", "matters", "because", "something", "someone",
                 "people", "there", "their", "which", "where", "would", "could", "should", "every", "whole", "check", "checks",
                 "checking", "answer", "verdict", "about", "other", "thing", "things", "still", "never", "always", "these",
                 "those", "makes", "means", "within", "without", "being", "while", "rather"}


LESSON_TERM_SHARED_MAX = 2  # a term also in more than this many other lessons' paragraphs doesn't identify the trait


def _paragraph_words(lesson_text: str) -> set[str]:
    from role_set.prompts import lesson_sentences

    words = re.findall(r"[a-z']+", " ".join(lesson_sentences(lesson_text)).lower())
    return {w for w in words if len(w) >= 5 and "'" not in w and w not in LESSON_COMMON}  # contractions say nothing of the trait


def lesson_terms(lesson_text: str, other_lessons: list[str] | None = None) -> set[str]:
    """Distinctive content words of a lesson's teaching paragraph, the prose the writer is shown: five letters or more,
    not in LESSON_COMMON, and in at most two other lessons' paragraphs (R&D on #25), so a restatement can't pass on
    words the whole pack shares."""

    own = _paragraph_words(lesson_text)
    if not other_lessons:
        return own
    elsewhere: dict[str, int] = {}
    for text in other_lessons:
        for w in _paragraph_words(text) & own:
            elsewhere[w] = elsewhere.get(w, 0) + 1
    return {w for w in own if elsewhere.get(w, 0) <= LESSON_TERM_SHARED_MAX}


def warmup_retry(trait_name: str, empty: list[str]) -> str:
    ask = f"Your restatement must name this trait as the lesson does: '{trait_name}'." if trait_name else "Restate the trait as the lesson does."
    if empty:
        ask += " Answer every part: " + ", ".join(empty) + "."
    return ask + " Return the JSON object again, in your own words."


def warmup_terms(trait_name: str) -> set[str]:
    """The content words of a trait's name: the warm-up's restatement names the trait by at least one of them."""

    words = re.findall(r"[a-z']+", trait_name.lower())
    return {w for w in words if len(w) >= 3 and w not in WARMUP_FILLER}


# The dispatch order of kinds within each round of interleave(): B last, since it is the kind most likely to fail on
# output shape and, first in the plan, it once ended every lesson before any other kind ran.
KIND_ORDER = ("A", "C", "D", "E", "B")


def cell_key(lesson: str, kind: str) -> str:
    return f"{lesson}/{kind}"


def interleave(slots: list[Slot]) -> list[Slot]:
    """The plan's slots in dispatch order: round by round, one slot from every (lesson, kind) cell, kinds in KIND_ORDER
    and lessons in plan order. Each cell keeps its own order. A short budget then samples every lesson and kind."""

    lessons: list[str] = []
    position: dict[tuple[str, str], int] = {}
    keyed = []
    for s in slots:
        if s.lesson not in lessons:
            lessons.append(s.lesson)
        cell = (s.lesson, s.kind)
        position[cell] = position.get(cell, -1) + 1
        rank = KIND_ORDER.index(s.kind) if s.kind in KIND_ORDER else len(KIND_ORDER)
        keyed.append(((position[cell], rank, lessons.index(s.lesson)), s))
    return [s for _, s in sorted(keyed, key=lambda x: x[0])]


# Refusal code -> what the builder is told, with the governing rule. Pinned by a test.
SUGGESTIONS = {
    "ANSWER_EARLY": "{detail}. Work through the lines in order, citing each as 'line N', and reach the answer only at the end (BUILD-SPEC B3).",
    "FRAME": "The framing doesn't open in its frame: {detail}. Rewrite only the framing, in the slot's frame (BUILD-SPEC B2).",
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
    "TRUNCATED": "The reply ran past the length limit before its JSON closed ({detail}). Decide the item before writing it: the thinking states the path to the answer in a few steps and never searches or reconsiders.",
    "TRANSPORT": "The writer could not be reached ({detail}); the slot ends.",
    "PUZZLE": "The puzzle's model doesn't hold: {detail}.",
    "STUCK": "The thinking circles back on itself ({detail}). Walk the elimination once, forward, one line at a time, and stop at the answer.",
    "SHAPE": "The reply is a finished row (kind, turns, answer), the exemplars' shape. Return one flat object with exactly these fields instead: {detail}.",
    "REPLY_OPENING": "The reply opens like two accepted items of this trait ('{detail}'). Open it another way, in this item's own words.",
    "SPENT": "The user turn reuses a trained item's wording ('{detail}'). Write it fresh: an apply item's wording appears in no trained item.",
    "LESSON_COPY": "The text copies the trait lesson's wording ('{detail}'). Say it in your own words (handoff rule 11: never copy a sentence from the lessons).",
}


def models_line(row_id: str, structure: str | None, line: str, model: dict) -> dict:
    """One `.models.jsonl` line, as `puzzle-check --stamp` writes it: the row's id, structure and model, and the
    sha256 of the row's line exactly as written to the data file."""

    from role_set.puzzle_check import row_sha256  # noqa: PLC0415 - puzzle_check imports the runner's neighbours

    return {"id": row_id, "structure": structure, "model": model, "row_sha256": row_sha256(line)}


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
    lesson_halts: dict = field(default_factory=dict)  # lesson -> {"reason", "kind": "*", "rule": "WARMUP"}: its warm-up failed
    cell_halts: dict = field(default_factory=dict)  # "T09/B" -> {"reason", "kind", "rule"}: that lesson's kind stopped, the rest went on
    kinds_off: dict = field(default_factory=dict)  # kind -> rule: switched off after two lessons' cells in a row halted on it
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
            "cell_halts": self.cell_halts,
            "kinds_off": self.kinds_off,
            "skipped": self.skipped,
            "actions": self.actions[-200:],
        }


@dataclass
class SlotState:
    """One slot between attempts: the writer conversation so far, its material, and what it has spent. The inline
    path keeps it in memory; wave mode carries it in the drafts file between waves (BUILD-SPEC B1)."""

    slot: Slot
    messages: list
    material: Material | None
    facts: dict | None
    attempt: int = 0  # attempts used, across every wave: one budget of MAX_ATTEMPTS per slot
    spent: list = field(default_factory=list)
    refusals: list = field(default_factory=list)
    ended: bool = False  # the writer could not be reached: the slot ends, as inline


@dataclass
class Candidate:
    """A draft that passed every mechanical check up to and including ROLE_WORDS, waiting for the critic."""

    completion: object
    fields: dict
    row: dict
    glyphs: int
    lookup: bool


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


def _user_spans(row: dict) -> set[str]:
    """Every 8-word span of a row's user turns, outside its material lines."""

    content = {ln.strip() for ln in (row.get("material") or "").splitlines() if ln.strip()}
    out: set[str] = set()
    for t in row.get("turns") or []:
        for line in t.get("user", "").splitlines():
            if line.strip() and line.strip() not in content:
                w = _words(line)
                out |= {" ".join(w[i:i + LESSON_SPAN_WORDS]) for i in range(len(w) - LESSON_SPAN_WORDS + 1)}
    return out


def trained_spans(rows: list[dict]) -> set[str]:
    """The spent wording for apply items: every 8-word span of a trained item's user turns, outside material lines."""

    return set().union(*(_user_spans(r) for r in rows)) if rows else set()


def spent_span(row: dict, spent: set[str]) -> str | None:
    """The first 8-word span (sorted) of an apply item's user turn that a trained item's user turn holds, or None (the
    quiz's spent rule, aspire-si#127)."""

    found = _user_spans(row) & spent
    return sorted(found)[0] if found else None


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
    if code in ("ASSEMBLY", "PUZZLE"):
        # Rehearsal 3 keyed every assembly halt as bare ASSEMBLY, so two cells dropping for unrelated reasons could
        # switch a kind off. The key is the first problem's wording, with quoted values and numbers taken out.
        d = re.sub(r"^1\)\s*", "", detail).split("; ")[0]
        d = re.sub(r"(?<=[\s(])'[^']*'(?=[\s),:;.]|$)|\"[^\"]*\"", "''", d)
        d = re.sub(r"\d+", "N", d)
        return f"{code}: " + " ".join(d.split())[:80]
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
    if refusal.get("code") in ("LESSON_COPY", "SPENT"):  # a copied lesson span, or a trained item's span (apply items)
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


_teaching_text = teaching_text  # the trait check reads the same text in a run and in critic-control


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
        trait_gate: bool = False,  # likewise the trait check, until critic-control's trait leg meets R&D's bar
        answer_gate: bool = False,  # and the kind-B answer check, until the answer leg does
        stuck=None,  # callable(thinking, material, tier, tokens) -> the frozen stuck rule's measures, or None
        training: list[dict] | None = None,  # the whole training set, for apply items' exclusions (aspire-si#127)
        training_models: dict | None = None,  # its sidecar models, {id: model}
        prior=None,  # role_set.exclusion.State: every row earlier batches accepted (BUILD-SPEC B4)
        critique_scope: str = "all",  # "all" or "verdict" (design §11 F2: skip A, B, and written E when "verdict")
    ) -> None:
        self.gates, self.pack, self.trait_names = gates, pack, trait_names
        self.writer, self.critic = writer, critic
        self.out_dir = out_dir
        self.exemplars = exemplars or {}
        self.training = training or []
        self.training_models = training_models or {}
        self._spent: set[str] | None = None
        self.excerpts = {k: list(v) for k, v in (excerpts or {}).items()}
        self._slot_excerpt: dict[str, int] = {}
        self._used_excerpts: dict[str, set[int]] = {}
        self.critique_scope = critique_scope
        self.pins = pins or {}
        self.clock = clock
        self.status = Status(started_at=clock(), concurrency=max(1, concurrency))
        self.accepted: list[dict] = []
        self.models: dict[str, dict] = {}  # id -> sidecar model of each accepted kind B row, for the gates' puzzle_sig
        self.cell_drops: dict[tuple[str, str], int] = {}
        self.batch_check: str | None = None  # "clean" or "REFUSED: ..." once a run finishes unhalted
        self.lock = threading.RLock()  # re-entrant: logging runs both inside and outside the accept lock
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.avoid: list[str] = []
        # Earlier batches' rows (B4): the acceptance checks run against them as well as this run's, and their framings
        # seed the writer's avoid list.
        self.prior_rows: list[dict] = list(prior.rows) if prior is not None else []
        self.prior_models: dict = dict(prior.models) if prior is not None else {}
        if prior is not None:
            self.avoid.extend(f for f in prior.framings() if f not in self.avoid)
        self.flaw_gate = flaw_gate
        self.trait_gate, self.answer_gate = trait_gate, answer_gate
        self._spans_by_lesson: dict[str, set[str]] = {}
        self.stuck = stuck
        self.lesson_codes: dict[str, Counter] = {}
        self.recent_calls: deque = deque(maxlen=SYSTEMIC_WINDOW)
        self.halt_order: list[tuple[str, tuple[str, str]]] = []
        self.cell_drop_keys: dict[tuple[str, str], Counter] = {}
        self.receipts: dict[str, dict] = {}  # slot id -> its accepted receipt, for wave mode's drafts file
        self._locked: dict = {}  # slot id -> its solve-first puzzle (BUILD-SPEC B3), built once from the slot's pack and seed
        self._packs = None
        self.accept_order: dict[str, int] = {}  # slot id -> its place in self.accepted, set under the accept lock
        self.plan_kinds: set[str] = set()
        self.deadline: float | None = None
        self.lookups: dict[str, int] = {}  # lesson -> accepted B lookups
        self.finished: list[Outcome] = []  # every outcome as it lands, so a halted run still reports its slots
        self._critic_notes: dict[str, list] = {}  # slot id -> the critic checks of its current attempt, for drafts.jsonl

    def _spent_spans(self) -> set[str]:
        if self._spent is None:
            self._spent = trained_spans(self.training)
        return self._spent

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
        record = {"slot": slot.id, "attempt": attempt, "code": refusal["code"], "detail": refusal["detail"], "draft": draft}
        if refusal.get("critic_raw") is not None:
            record["critic_raw"] = refusal["critic_raw"]
        self._append("refusals.jsonl", record)

    def _record_call(self, slot: Slot, attempt: int, c, outcome: str, glyphs: int, material) -> None:
        """One line per writer call, on both paths: the raw draft, tokens, latency, the outcome, glyphs normalised, and
        the frozen stuck rule's measures on the draft's thinking. Then the systemic check over the last SYSTEMIC_WINDOW calls."""
        text = c.text if c is not None else ""
        record = {"slot": slot.id, "lesson": slot.lesson, "kind": slot.kind, "attempt": attempt, "outcome": outcome,
                  "prompt_tokens": getattr(c, "prompt_tokens", None), "completion_tokens": getattr(c, "completion_tokens", None),
                  "seconds": round(getattr(c, "seconds", 0.0) or 0.0, 2), "glyphs_normalized": glyphs, "draft": text}
        # Every critic check this draft went through, agreeing or not (gating or advisory), so the critic's behaviour
        # on a paid run can be read per draft. Absent when the draft never reached the critic.
        with self.lock:
            checks = self._critic_notes.pop(slot.id, None)
        if checks is not None:
            record["critic"] = checks
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
    def warmup(self, lesson: str) -> bool:
        """The builder restates the trait before it writes: all three parts answered, and the trait named by a content
        word of its own name. Two misses stop the trait. Every reply is kept in warmups.jsonl, passed or missed.

        The 235B pilot's check also demanded the planner's side labels ("quick on the obvious"), which the writer is
        never shown, and filler words: T07 and T10 could not pass at any seed, and 21 of 64 chunks lost their warm-up."""

        lesson_text = self.pack.get(f"traits/{lesson}.md", "")
        trait_name = self.trait_names.get(lesson, "")
        must = warmup_terms(trait_name)
        others = [v for k, v in self.pack.items() if k.startswith("traits/") and k != f"traits/{lesson}.md"]
        taught = lesson_terms(lesson_text, others)
        messages = warmup_messages(self.pack, lesson)
        for attempt in range(1, WARMUP_MISSES + 1):
            c = self.writer(messages, seed=attempt)
            self._count_tokens(c)
            answer: dict = {}
            try:
                answer = parse_json_object(c.text)
            except ValueError:
                pass
            text = " ".join(str(answer.get(k, "")) for k in WARMUP_PARTS).lower()
            hit = {w for w in must if re.search(rf"\b{re.escape(w)}", text)}
            taught_hit = {w for w in taught if re.search(rf"\b{re.escape(w)}", text)}
            empty = [k for k in WARMUP_PARTS if not str(answer.get(k, "")).strip()]
            # Named by its own name, or restated in the lesson's own terms (two of them). A restatement that is vague
            # or about another trait hits neither.
            passed = bool(hit or not must or len(taught_hit) >= WARMUP_LESSON_TERMS) and not empty
            self._append("warmups.jsonl", {"lesson": lesson, "attempt": attempt, "passed": passed, "terms": sorted(hit),
                                           "lesson_terms": sorted(taught_hit), "empty_parts": empty, "reply": c.text})
            if passed:
                self._log("warmup_pass", lesson=lesson, attempt=attempt, terms=sorted(hit), lesson_terms=sorted(taught_hit))
                (self.out_dir / f"warmup-{lesson}.json").write_text(json.dumps({"attempt": attempt, "answer": answer, "lesson_chars": len(lesson_text)}, indent=1), encoding="utf-8")
                return True
            self._log("warmup_miss", lesson=lesson, attempt=attempt, missing=sorted(must)[:6], empty_parts=empty)
            # The retry marks the missed point, as the design asks. The rehearsal's 8B gave the identical reply to an
            # unchanged prompt, seed or no seed.
            messages = messages + [{"role": "assistant", "content": c.text},
                                   {"role": "user", "content": warmup_retry(trait_name, empty)}]
        return False

    # ---- one slot ---------------------------------------------------------------------------------------------
    def _material_for(self, slot: Slot) -> Material | None:
        if slot.kind in ("A", "B") or (slot.kind == "E" and slot.e_type == "written"):
            return None
        kind = slot.content_kind or "notice"
        if slot.real_source:
            pool = self.excerpts.get(kind) or []
            if not pool:
                raise AssemblyError(f"real_source slot {slot.id} ({kind}) has no excerpt in the pool")
            with self.lock:
                if slot.id not in self._slot_excerpt:
                    used = self._used_excerpts.setdefault(kind, set())
                    start = int(hashlib.sha256(slot.id.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
                    idx = next(((start + k) % len(pool) for k in range(len(pool)) if (start + k) % len(pool) not in used), start)
                    self._slot_excerpt[slot.id] = idx
                    used.add(idx)
                return pool[self._slot_excerpt[slot.id]]
        needs = f"It must be the kind of text a verdict of '{slot.verdict}' can be reached about for one plain statement."
        c = self._writer_call(material_messages(slot, needs), 0, slot)
        self._count_tokens(c)
        text = c.text.strip()
        if not text or len(text) < 20:
            raise AssemblyError("the drafted material came back empty")
        return Material(text + "\n", {"invented": f"{kind.replace('_', ' ')} material, drafted for this item and fixed before writing"})

    def _writer_call(self, messages: list[dict], seed: int, slot: Slot):
        """The writer, with the slot's id for a mock that asks for it (wave mode's keyed mock, so its drafts depend on
        the slot and attempt, not the order of calls). A real writer is never told: the prompt carries no id."""

        if getattr(self.writer, "takes_slot_id", False):
            return self.writer(messages, seed=seed, slot_id=slot.id)
        return self.writer(messages, seed=seed)

    def _critic_call(self, messages: list[dict], *, check: str | None = None, **kwargs):
        if getattr(self.critic, "takes_check", False) and check is not None:
            return self.critic(messages, check=check, **kwargs)
        return self.critic(messages, **kwargs)

    def _refusal(self, code: str, detail: str) -> dict:
        tmpl = SUGGESTIONS.get(code, "{detail}")
        return {"code": code, "detail": detail, "suggestion": tmpl.format(detail=detail)}

    def _gated_or_advisory(self, verdicts: list, slot: Slot, v, raw: str, gate: bool, action: str) -> None:
        """A check that refuses only once critic-control shows the critic meets its bar. Until then a mismatch is
        recorded with the critic's words and counted, never a refusal (R&D, 2026-10-11: on reviewed items the critic
        named the trait right 40% of the time and solved 63% of the puzzles, so gating would refuse correct items)."""

        if v.ok or gate:
            verdicts.append((v, raw))
            return
        self.status.critic_asked += 1
        self._log(action, slot=slot.id, code=v.code, detail=v.detail, critic_raw=raw[:2000])

    def _note(self, slot: Slot, check: str, v, raw: str) -> None:
        with self.lock:
            self._critic_notes.setdefault(slot.id, []).append({"check": check, "code": v.code, "detail": v.detail, "raw": raw[:2000]})

    def _critic_pass(self, slot: Slot, row: dict, fields: dict) -> dict | None:
        if self.critic is None:
            return None
        if self.critique_scope == "verdict" and (slot.kind in ("A", "B") or (slot.kind == "E" and slot.e_type == "written")):
            return None
        with self.lock:
            self._critic_notes[slot.id] = []
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
                    c = self._critic_call(verdict_messages(seen_material, statement), check="verdict")
                    self._count_tokens(c)
                    v = compare_verdict(self.gates, c.text, key_verdict, key_deciding, seen_material)
                    if v.code != "VOID":
                        break
                    self._log("critic_void", slot=slot.id, detail=v.detail)
                self._note(slot, "verdict", v, c.text)
                verdicts.append((v, c.text))
                if not v.ok:
                    break
            twin = row.get("r10_twin")
            if isinstance(twin, dict) and twin.get("statement") and twin.get("verdict"):
                for ask in range(2):
                    c = self._critic_call(verdict_messages(material, twin["statement"]), check="twin_verdict")
                    self._count_tokens(c)
                    tv = compare_twin_verdict(self.gates, c.text, twin["verdict"])
                    if tv.code != "VOID":
                        break
                    self._log("critic_void", slot=slot.id, detail=tv.detail)
                self._note(slot, "twin_verdict", tv, c.text)
                verdicts.append((tv, c.text))
            if slot.kind == "C":
                c = self._critic_call(flaw_messages(material, statement, fields["checker_trace"], row["checker_verdict"]), check="flaw")
                self._count_tokens(c)
                f = compare_flaw(c.text, row["planted_flaw"])
                self._note(slot, "flaw", f, c.text)
                if f.ok or self.flaw_gate:
                    verdicts.append((f, c.text))
                else:
                    # Advisory until a control run shows the critic agrees on flaws (Window A: 9 of 24): logged, counted, never a refusal.
                    self.status.critic_asked += 1
                    self._log("flaw_advisory", slot=slot.id, detail=f.detail, critic_raw=c.text[:2000])
        elif slot.kind == "B":
            for ask in range(2):
                c = self._critic_call(answer_messages(row["turns"][0]["user"], row["answer"]), check="answer")
                self._count_tokens(c)
                v = compare_answer(c.text, row["answer"])
                if v.code != "VOID":
                    break
                self._log("critic_void", slot=slot.id, detail=v.detail)
            self._note(slot, "answer", v, c.text)
            self._gated_or_advisory(verdicts, slot, v, c.text, self.answer_gate, "answer_advisory")
        # The trait is a majority of three calls (R&D, 2026-10-11): one call was unstable on 5 of 20 items.
        votes = []
        for seed in range(TRAIT_VOTES):
            c = self._critic_call(trait_messages(_teaching_text(row), self.trait_names, TRAIT_STEPS),
                                  check="trait", temperature=TRAIT_TEMPERATURE, seed=seed)
            self._count_tokens(c)
            votes.append(c.text)
        trait = compare_trait_votes(votes, slot.lesson)
        self._note(slot, "trait", trait, json.dumps(votes, ensure_ascii=False))
        self._gated_or_advisory(verdicts, slot, trait, json.dumps(votes, ensure_ascii=False), self.trait_gate, "trait_advisory")
        for v, raw in verdicts:
            self.status.critic_asked += 1
            if v.ok:
                self.status.critic_agree += 1
            else:
                # The critic's own words are kept with the refusal, so a mismatch can be adjudicated afterwards.
                return {**self._refusal(v.code, v.detail), "critic_raw": raw}
        return None

    def open_slot(self, slot: Slot, coverage_notes: list[str] | None = None) -> SlotState | Outcome:
        """The slot's material and first writer prompt. A material that can't be drafted ends the slot here."""

        try:
            material = self._material_for(slot)
        except AssemblyError as e:
            return Outcome(slot_id=slot.id, accepted=False, attempts=0, refusals=[self._refusal("ASSEMBLY", str(e))])
        exemplars = [e for k, v in self.exemplars.items() if k != slot.lesson for e in v
                     if e.get("kind") == slot.kind and e.get("split") != "apply"][:3]  # apply rows are never shown
        facts = {"material_line_count": len(material.text.splitlines())} if material else None
        try:
            locked = self.locked_for(slot)
        except SolveFirstError as e:
            return Outcome(slot_id=slot.id, accepted=False, attempts=0, refusals=[self._refusal("PUZZLE", str(e))])
        messages = writer_messages(slot, self.pack, material_text=material.text if material else None, exemplars=exemplars, avoid=list(self.avoid[-60:]), facts=facts, coverage_notes=coverage_notes,
                                   trait_words=self.gates.trait_words(slot.lesson), closed_form=self.gates.closed_form,
                                   locked_brief=solvefirst.brief(locked) if locked else None)
        return SlotState(slot=slot, messages=messages, material=material, facts=facts)

    def locked_for(self, slot: Slot):
        """The slot's solve-first puzzle, built from its pack and seed, or None for a freehand slot. Deterministic, so a
        critique wave rebuilds the very model the write wave locked."""

        if slot.kind != "B" or slot.b_path != "solve_first":
            return None
        if slot.id not in self._locked:
            if self._packs is None:
                self._packs = solvefirst.load_packs()
            if slot.pack not in self._packs:
                raise SolveFirstError(f"REFUSED: {slot.id} names pack {slot.pack!r}, which b_packs.json doesn't hold")
            self._locked[slot.id] = solvefirst.generate(slot.structure, self._packs[slot.pack], slot.b_seed or 0)
        return self._locked[slot.id]

    def mechanical(self, st: SlotState, c) -> tuple[dict | None, Candidate | None, int]:
        """Every check up to and including ROLE_WORDS: parse, shape, puzzle, stuck, assemble, truth, lesson copy,
        spent, the gates' check_item and role words. A refusal or the candidate for the critic, and the glyphs normalised."""

        slot, material, facts = st.slot, st.material, st.facts
        trait = self.trait_names.get(slot.lesson, slot.lesson)
        refusal = None
        row = None
        glyphs = 0
        fields: dict = {}
        try:
            fields, glyphs = normalize_glyphs(parse_json_object(c.text))
        except ValueError as e:
            refusal = self._refusal("TRUNCATED" if getattr(c, "finish_reason", None) == "length" else "PARSE", str(e))
        if refusal is None and row_shaped(fields, slot):
            refusal = self._refusal("SHAPE", ", ".join(FIELDS[kind_key(slot)]))
        lookup = False
        locked = self.locked_for(slot) if refusal is None else None
        if locked is not None:
            # Solve-first: the writer never writes the model or the answer. They are injected, and the same check runs.
            fields["model"], fields["answer"] = locked.model, locked.key
        if refusal is None and slot.kind == "B":
            # Only the solver decides a puzzle's answer; the writer is never asked whether it is unique.
            try:
                lookup = check_puzzle(fields, slot.structure, strict=locked is not None).lookup
            except PuzzleError as e:
                refusal = self._refusal("PUZZLE", str(e))
            if refusal is None and locked is not None:
                early = solvefirst.answer_early(locked, str(fields.get("thinking") or ""))
                if early:
                    refusal = self._refusal("ANSWER_EARLY", early)
            if refusal is None and lookup and self.lookups.get(slot.lesson, 0) >= lookup_cap(slot):
                refusal = self._refusal("PUZZLE", lookup_full(slot))
            # A B trace that circles is refused, not just recorded (R&D on #29: pilot 2's T03-B-01 circled and passed).
            if refusal is None and self.stuck is not None:
                measures = self.stuck(str(fields.get("thinking") or ""), "", slot.tier, getattr(c, "completion_tokens", 0) or 0)
                if measures.get("stuck"):
                    refusal = self._refusal("STUCK", ", ".join(f"{k} {measures[k]}" for k in ("bursts", "hedges") if k in measures) or "stuck rule")
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
        if refusal is None and slot.frame:
            subject = [(slot.content_kind or "").replace("_", " "), (slot.domain or {}).get("title", "")]
            off = check_frame(slot.frame, row["turns"][0]["user"], material.text if material else "",
                              deciding=((row.get("turn_keys") or {}).get("1") or row).get("deciding") or (),
                              subject_words=subject)
            if off:
                refusal = self._refusal("FRAME", off)
        if refusal is None:
            copied = lesson_copy(row, self._lesson_spans(slot.lesson))
            if copied:
                refusal = self._refusal("LESSON_COPY", copied)
        if refusal is None and slot.split == "apply":
            spent = spent_span(row, self._spent_spans())
            if spent:
                refusal = self._refusal("SPENT", spent)
        if refusal is None:
            try:
                # An apply item is never trained, so check_item refuses it; it has its own check (gates 1.7.5).
                (self.gates.check_apply_item if slot.split == "apply" else self.gates.check_item)(row)
                if slot.kind == "C":
                    self.gates.check_trace_leaks_label(row)
            except self.gates.error_type() as e:
                refusal = self._refusal("GATE", str(e))
        if refusal is None:
            words = self.gates.role_words_in_prompt(row)
            if words:
                refusal = self._refusal("ROLE_WORDS", ", ".join(words))
        if refusal is not None:
            return refusal, None, glyphs
        return None, Candidate(completion=c, fields=fields, row=row, glyphs=glyphs, lookup=lookup), glyphs

    def write_step(self, st: SlotState) -> Candidate | None:
        """Writer calls until a draft passes the mechanical checks, inside the slot's attempt budget. A mechanical
        refusal is retried at once, while the writer is loaded. None when the budget is spent or the writer is gone."""

        while not st.ended and st.attempt < MAX_ATTEMPTS:
            st.attempt += 1
            try:
                c = self._writer_call(st.messages, st.attempt, st.slot)
            except TransportError as e:
                # A writer that can't be reached ends this slot, never the run by itself; the systemic rule counts it.
                refusal = self._refusal("TRANSPORT", str(e)[:200])
                st.refusals.append(refusal)
                self._note_refusal(st.slot, st.attempt, refusal, draft="")
                self._record_call(st.slot, st.attempt, None, "TRANSPORT", 0, st.material)
                st.ended = True
                return None
            self._count_tokens(c)
            refusal, cand, glyphs = self.mechanical(st, c)
            if cand is not None:
                return cand
            self.refuse(st, c, refusal, glyphs)
        return None

    def critique_step(self, st: SlotState, cand: Candidate) -> dict | None:
        """The critic with the key hidden, then the acceptance checks under the lock. None when the row was accepted;
        otherwise the refusal, not yet applied to the slot (refuse() does that)."""

        slot, row, fields, c = st.slot, cand.row, cand.fields, cand.completion
        refusal = self._critic_pass(slot, row, fields)
        if refusal is not None:
            return refusal
        with self.lock:
            try:
                models = {**self.prior_models, **self.models, **({row["id"]: fields["model"]} if slot.kind == "B" else {})}
                # An apply item may reuse no training item's material either (aspire-si#127): it is checked
                # against the whole training set as well as this run's rows.
                against = self.prior_rows + self.accepted + (self.training if slot.split == "apply" else [])
                if slot.split == "apply":
                    models = {**self.training_models, **models}
                reasons = self.gates.check_against(row, against, models=models)
            except self.gates.error_type() as e:
                reasons = [("AGAINST", str(e))]
            stamp = reply_opening_reused(row, self.prior_rows + self.accepted)
            if reasons:
                code, msg = reasons[0]
                return self._refusal("AGAINST", f"{code}: {msg}")
            if stamp:
                return self._refusal("REPLY_OPENING", stamp)
            if cand.lookup and self.lookups.get(slot.lesson, 0) >= lookup_cap(slot):
                # Read again under the lock: two in-flight B slots of one lesson both passed the first read.
                return self._refusal("PUZZLE", lookup_full(slot))
            if slot.kind == "B":
                self.models[row["id"]] = fields["model"]
            self._accept(slot, row, c, st.attempt, glyphs=cand.glyphs)
            self.accept_order[slot.id] = len(self.accepted) - 1
            if cand.lookup:
                self.lookups[slot.lesson] = self.lookups.get(slot.lesson, 0) + 1
            self._record_call(slot, st.attempt, c, "accepted", cand.glyphs, st.material)
            return None

    def refuse(self, st: SlotState, c, refusal: dict, glyphs: int) -> None:
        """A refused draft: kept, logged, and answered with the suggestion, so the next attempt continues the same
        conversation (the writer's draft, then "Refused: ...")."""

        slot = st.slot
        st.refusals.append(refusal)
        self._note_refusal(slot, st.attempt, refusal, draft=c.text)
        self._record_call(slot, st.attempt, c, refusal["code"], glyphs, st.material)
        st.spent += spent_wording(refusal)
        if st.spent:
            with self.lock:
                self.avoid.extend(x for x in st.spent if x not in self.avoid)
        ask = "Refused: " + refusal["suggestion"]
        if st.spent:
            # The 8B rehearsal repeated the same span on every retry when told only the reason (Window B).
            ask += " Do not use any of these wordings, in any order or form: " + "; ".join(f"'{x}'" for x in dict.fromkeys(st.spent)) + "."
        ask += trait_words_note(self.gates.trait_words(slot.lesson) if slot.kind == "A" else [], st.spent)
        st.messages = st.messages + [
            {"role": "assistant", "content": c.text},
            {"role": "user", "content": ask + " Return a corrected JSON object with all fields."},
        ]

    @staticmethod
    def outcome_of(st: SlotState, row: dict | None = None) -> Outcome:
        return Outcome(slot_id=st.slot.id, accepted=row is not None, attempts=st.attempt, refusals=list(st.refusals), row=row)

    def run_slot(self, slot: Slot, coverage_notes: list[str] | None = None) -> Outcome:
        """The inline path: write, critique, and rewrite on a refusal, all in one process."""

        st = self.open_slot(slot, coverage_notes)
        if isinstance(st, Outcome):
            return st
        while True:
            cand = self.write_step(st)
            if cand is None:
                return self.outcome_of(st)
            refusal = self.critique_step(st, cand)
            if refusal is None:
                return self.outcome_of(st, cand.row)
            self.refuse(st, cand.completion, refusal, cand.glyphs)

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
            "gates": {"version": self.gates.version, "sha256": self.gates.sha256, "closed_form": self.gates.closed_form},
            "row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode("utf-8")).hexdigest(),
            "glyphs_normalized": glyphs,
            **self.pins,
        }
        if slot.kind == "B":
            locked = self._locked.get(slot.id)
            receipt["b"] = locked.receipt() if locked is not None else {"b_path": slot.b_path or "freehand", "pack": slot.pack}
        self.receipts[slot.id] = receipt
        line = json.dumps(row, ensure_ascii=False)
        with (self.out_dir / "accepted.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if slot.kind == "B" and row.get("id") in self.models:
            # The gates' sidecar (Argon §9.4 #8): an apply batch's --training reads B models from it, each line's
            # row_sha256 the stamp of the row exactly as written above.
            with (self.out_dir / "accepted.models.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(models_line(row["id"], slot.structure, line, self.models[row["id"]]), ensure_ascii=False) + "\n")
        with (self.out_dir / "receipts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        # One file for the key-check: each accepted row beside the critic checks it passed through (R&D, pilot 2).
        # A review copy only; the store ingests accepted.jsonl and receipts.jsonl, never this.
        review = {"id": row.get("id"), "row": row, "critic": list(self._critic_notes.get(slot.id) or [])}
        with (self.out_dir / "review.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(review, ensure_ascii=False) + "\n")
        self._log("accepted", slot=slot.id, attempt=attempt)

    # ---- the batch --------------------------------------------------------------------------------------------
    def _andon(self, slot: Slot, out: Outcome | None = None) -> None:
        """A drop counts toward its (lesson, kind) cell. Three drops end that cell: its remaining slots are skipped and
        the lesson's other kinds go on. The cell's halt key is the specific rule most of its drops fell on.

        The 235B pilot's per-lesson andon put each lesson's first kind (B) in charge of the whole lesson: B failed on
        its output shape, every lesson ended inside B, and C, D, E and A were never reached."""

        self.status.dropped += 1
        cell = (slot.lesson, slot.kind)
        self.cell_drops[cell] = self.cell_drops.get(cell, 0) + 1
        keys = self.cell_drop_keys.setdefault(cell, Counter())
        for r in (out.refusals if out else []):
            keys[refusal_rule(r)] += 1
        if cell_key(*cell) not in self.status.cell_halts and self.cell_drops[cell] >= ANDON_CELL_DROPS:
            rule = keys.most_common(1)[0][0] if keys else "NONE"
            self._halt_cell(slot.lesson, slot.kind, f"{ANDON_CELL_DROPS} dropped slots", rule)

    def _halt_cell(self, lesson: str, kind: str, reason: str, rule: str) -> None:
        """Two cells of different lessons halting in a row on the same (kind, rule) switch that kind off for the rest of
        the run; the other kinds carry on. The run stops only when every kind in the plan is off (or on the
        PARSE/TRANSPORT rate). A failed warm-up is never part of this: it ends its own lesson and nothing else."""

        self.status.cell_halts[cell_key(lesson, kind)] = {"reason": reason, "kind": kind, "rule": rule}
        self._log("cell_halt", lesson=lesson, kind=kind, reason=reason, rule=rule)
        prev = self.halt_order[-1] if self.halt_order else None
        self.halt_order.append((lesson, (kind, rule)))
        if prev and prev[1] == (kind, rule) and prev[0] != lesson and kind not in self.status.kinds_off:
            self.status.kinds_off[kind] = rule
            self._log("kind_off", kind=kind, rule=rule, lessons=[prev[0], lesson])
            if self.plan_kinds and self.plan_kinds <= set(self.status.kinds_off) and self.status.halted is None:
                self.status.halted = "systemic: every kind is switched off (" + "; ".join(f"{k}: {r}" for k, r in sorted(self.status.kinds_off.items())) + ")"
                self._log("halt", reason=self.status.halted)
                raise RunHalted(self.status.halted)

    def _halt_lesson_warmup(self, lesson: str) -> None:
        self.status.lesson_halts[lesson] = {"reason": "warm-up failed: the builder could not restate the trait", "kind": "*", "rule": "WARMUP"}
        self._log("lesson_halt", lesson=lesson, reason=self.status.lesson_halts[lesson]["reason"], kind="*", rule="WARMUP")

    def _skip(self, slot: Slot) -> str | None:
        if self.status.halted is not None:
            return "run halted"
        if slot.kind in self.status.kinds_off:
            return f"kind {slot.kind} systemic: {self.status.kinds_off[slot.kind]}"
        if slot.lesson in self.status.lesson_halts:
            return f"lesson {slot.lesson} warm-up failed"
        if cell_key(slot.lesson, slot.kind) in self.status.cell_halts:
            return f"{cell_key(slot.lesson, slot.kind)} halted"
        if self.deadline is not None and self.clock() >= self.deadline:
            return "time budget"
        return None

    def run(self, slots: list[Slot], *, stop_after_s: float | None = None) -> list[Outcome]:
        outcomes: list[Outcome] = []
        if stop_after_s is not None:
            self.deadline = self.clock() + stop_after_s  # no new slot starts after this; running ones finish
        self.plan_kinds = {s.kind for s in slots}
        lessons = []
        for s in slots:
            if s.lesson not in lessons:
                lessons.append(s.lesson)
        for lesson in lessons:
            if not self.warmup(lesson):
                self._halt_lesson_warmup(lesson)
        # Every lesson and kind is reached early, so a short budget still samples every failure mode.
        slots = interleave(slots)

        def one(slot: Slot) -> Outcome:
            with self.lock:
                why = self._skip(slot)
                if why:
                    self.status.skipped[why] = self.status.skipped.get(why, 0) + 1
                    return Outcome(slot_id=slot.id, accepted=False, attempts=0, skipped=True, skip_reason=why)
                self.status.attempted += 1
            try:
                out = self.run_slot(slot)
            except RunHalted:
                # The halt fired on this slot's own call, after its row may already have been accepted. Its outcome
                # is still reported: pilot 2 lost four accepted slots this way, and its readout's kinds summed to 45 of 49.
                with self.lock:
                    done = any(r.get("id") == slot.id for r in self.accepted)
                    # Not accepted: the slot was cut short, not finished, so a relaunch (pilot-auto) runs it again.
                    self.finished.append(Outcome(slot_id=slot.id, accepted=done, attempts=0, skipped=not done,
                                                 skip_reason=None if done else "run halted during the slot"))
                raise
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
                self.gates.check_batch(rows, models=self.models)
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
