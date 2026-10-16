"""The domain table (BUILD-SPEC B2): every slot that carries material or a passage is set in a named domain.

Domains come from a public taxonomy, not a model: the US Census NAICS 2022 sector and subsector titles, verbatim
from one pinned source file (its census.gov URL and sha256), under rnd's licence registry entry `census-naics-2022`.
Ordinary slots draw 3-digit subsectors, distinct within a lesson. An ESCALATE slot draws from the codes R&D mapped to
its escalation domain (2026-10-16): the trades that hold the qualification. A mapped code may also appear in an
ordinary slot; a slot is ESCALATE by plan, never by domain.

`scripts/naics_table.py` builds the table from the pinned xlsx; this module loads and checks it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from role_set.plan import ESCALATE_DOMAINS

REGISTRY_ID = "census-naics-2022"
SOURCE_HOSTS = ("https://www.census.gov/", "https://www2.census.gov/")
CODE_RE = re.compile(r"\d{2,6}")
SHA_RE = re.compile(r"[0-9a-f]{64}")


class DomainTableError(ValueError):
    """A domain table that can't be trusted: no pinned source, a bad code, or an ESCALATE domain left empty."""


@dataclass(frozen=True)
class DomainTable:
    subsectors: tuple[dict, ...]  # {"code", "title"}, 3-digit
    escalate: dict  # domain -> ({"code", "title"}, ...)
    source: dict
    sha256: str  # of the table file itself

    def pin(self) -> dict:
        return {"domain_table_sha256": self.sha256, "domain_source_sha256": self.source["sha256"], "domain_registry_id": self.source["registry_id"]}


def _entries(items, where: str, digits: int | None = None) -> tuple[dict, ...]:
    out = []
    for e in items or []:
        code, title = str(e.get("code") or ""), str(e.get("title") or "").strip()
        if not CODE_RE.fullmatch(code) or (digits and len(code) != digits):
            raise DomainTableError(f"REFUSED: {where} has a bad NAICS code {code!r}")
        if not title:
            raise DomainTableError(f"REFUSED: {where} code {code} has no title")
        out.append({"code": code, "title": title})
    if len({e["code"] for e in out}) != len(out):
        raise DomainTableError(f"REFUSED: {where} repeats a code")
    return tuple(out)


def load(path: Path) -> DomainTable:
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    src = data.get("source") or {}
    if src.get("registry_id") != REGISTRY_ID:
        raise DomainTableError(f"REFUSED: the domain table's source must carry registry_id {REGISTRY_ID}")
    if not str(src.get("url") or "").startswith(SOURCE_HOSTS):
        raise DomainTableError("REFUSED: the domain table's source must be a census.gov URL")
    if not SHA_RE.fullmatch(str(src.get("sha256") or "")):
        raise DomainTableError("REFUSED: the domain table's source file must be pinned by sha256")
    subsectors = _entries(data.get("subsectors"), "subsectors", digits=3)
    if not subsectors:
        raise DomainTableError("REFUSED: the domain table has no subsectors")
    esc = data.get("escalate") or {}
    if set(esc) != set(ESCALATE_DOMAINS):
        raise DomainTableError(f"REFUSED: the ESCALATE mapping must cover exactly {', '.join(ESCALATE_DOMAINS)}")
    escalate = {d: _entries(esc[d], f"escalate {d}") for d in ESCALATE_DOMAINS}
    if any(not v for v in escalate.values()):
        raise DomainTableError("REFUSED: every ESCALATE domain needs at least one code")
    return DomainTable(subsectors=subsectors, escalate=escalate, source=src, sha256=hashlib.sha256(raw).hexdigest())


class Deck:
    """Subsectors dealt across a whole batch: shuffled, dealt in turn, reshuffled when spent, so a batch uses each
    domain about as often as any other. A lesson never gets the same code twice."""

    def __init__(self, table: DomainTable, rng):
        self.table, self.rng = table, rng
        self.cards: list[dict] = []

    def deal(self, used: set[str]) -> dict:
        for _ in range(2):
            for i, card in enumerate(self.cards):
                if card["code"] not in used:
                    return self.cards.pop(i)
            fresh = list(self.table.subsectors)
            self.rng.shuffle(fresh)
            self.cards += fresh
        raise DomainTableError(f"REFUSED: a lesson needs more distinct domains than the table's {len(self.table.subsectors)} subsectors")

    def escalate(self, domain: str, used: set[str]) -> dict:
        choices = [c for c in self.table.escalate[domain] if c["code"] not in used] or list(self.table.escalate[domain])
        return dict(self.rng.choice(choices))


def bears_domain(slot) -> bool:
    """A slot that carries material or a passage: A, C, D, a verdict-type E, and the C and D apply slots. Kind B
    takes its domains from the puzzle packs (B3); a written E has no material."""

    return slot.kind in ("A", "C", "D") or (slot.kind == "E" and slot.e_type == "verdict")


def _escalate_domain(escalate: str) -> str | None:
    m = re.match(r"yes \((\w+)", escalate or "")
    return m.group(1) if m else None


def assign(slots: list, table: DomainTable, *, seed: int) -> None:
    """A domain for every slot that bears one and a frame for every framed slot, by seed, with no model call.
    Codes are distinct within a lesson; an ESCALATE slot draws from its escalation domain's codes; frames are
    balanced within a lesson (counts differ by at most one) and rotate across lessons."""

    import random

    from role_set import frames

    rng = random.Random(f"domains:{seed}")
    deck = Deck(table, rng)
    lessons: list[str] = []
    for s in slots:
        if s.lesson not in lessons:
            lessons.append(s.lesson)
    for li, lesson in enumerate(lessons):
        mine = [s for s in slots if s.lesson == lesson]
        used: set[str] = set()
        for s in mine:
            dom = _escalate_domain(s.escalate)
            if bears_domain(s) and dom:
                pick = deck.escalate(dom, used)
                s.domain = {**pick, "escalate": dom}
                used.add(pick["code"])
        for s in mine:
            if bears_domain(s) and s.domain is None:
                pick = deck.deal(used)
                s.domain = dict(pick)
                used.add(pick["code"])
        framed = [s for s in mine if frames.framed(s.kind, s.e_type)]
        for s, f in zip(framed, frames.assign(len(framed), li, random.Random(f"frames:{seed}:{lesson}"))):
            s.frame = f
    check(slots)


def check(slots: list) -> None:
    """Refuse a plan whose domains repeat within a lesson, or whose frames are out of balance."""

    from collections import Counter

    from role_set import frames

    lessons = {s.lesson for s in slots}
    for lesson in lessons:
        mine = [s for s in slots if s.lesson == lesson]
        codes = [s.domain["code"] for s in mine if s.domain]
        if len(set(codes)) != len(codes):
            raise DomainTableError(f"REFUSED: {lesson} repeats a domain")
        if any(bears_domain(s) and not s.domain for s in mine):
            raise DomainTableError(f"REFUSED: {lesson} has a slot with material and no domain")
        counts = Counter(s.frame for s in mine if frames.framed(s.kind, s.e_type))
        if None in counts:
            raise DomainTableError(f"REFUSED: {lesson} has a framed slot with no frame")
        if counts and max(counts.values()) - min(counts.get(f, 0) for f in frames.FRAMES) > 1:
            raise DomainTableError(f"REFUSED: {lesson}'s frames are out of balance: {dict(counts)}")
    total = Counter(s.frame for s in slots if s.frame)
    n = sum(total.values())
    share = n / len(frames.FRAMES) if n else 0
    off = {f: total.get(f, 0) for f in frames.FRAMES if abs(total.get(f, 0) - share) > max(1.0, 0.1 * share)}
    if off:
        raise DomainTableError(f"REFUSED: across the batch, frames stray from their share of {share:.1f}: {off}")
