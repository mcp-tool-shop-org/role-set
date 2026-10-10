"""The critic's known-answer control: run it over reviewed items whose keys are known, and over
planted errors, before it is trusted to refuse anything in a paid run.

Measured: false refusals (the critic disagrees with a correct, reviewed key) and
catches (the critic disagrees with a key deliberately broken). A false-refusal
rate over the cap means the model isn't usable as the critic. Nothing here
writes to the store or the data; it writes one report.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from role_set.critic import compare_flaw, compare_verdict, flaw_messages, verdict_messages
from role_set.gates import Gates

FALSE_REFUSAL_CAP = 0.10
STATEMENT = re.compile(r"^Statement:\s*(.+?)\s*$", re.M)


@dataclass
class Case:
    item_id: str
    kind: str
    material: str
    statement: str
    verdict: str
    deciding: list[str]
    planted: bool = False
    flaw: str | None = None
    checker_trace: str | None = None
    checker_verdict: str | None = None


@dataclass
class ControlReport:
    items: int = 0
    verdict_asks: int = 0
    agree: int = 0
    false_refusals: list = field(default_factory=list)
    planted: int = 0
    catches: int = 0
    voids: int = 0
    flaw_asks: int = 0
    flaw_agree: int = 0
    flaw_misses: list = field(default_factory=list)

    def as_dict(self) -> dict:
        reviewed = self.verdict_asks - self.planted
        rate = len(self.false_refusals) / reviewed if reviewed else 0.0
        return {
            "items": self.items,
            "verdict_asks": self.verdict_asks,
            "reviewed_asks": reviewed,
            "agree": self.agree,
            "false_refusals": len(self.false_refusals),
            "false_refusal_rate": round(rate, 3),
            "usable_as_critic": rate <= FALSE_REFUSAL_CAP,
            "planted": self.planted,
            "catches": self.catches,
            "catch_rate": round(self.catches / self.planted, 3) if self.planted else None,
            "voids": self.voids,
            "flaw_asks": self.flaw_asks,
            "flaw_agree": self.flaw_agree,
            "flaw_misses": self.flaw_misses,
            "false_refusal_detail": self.false_refusals,
        }


def _statement(user: str) -> str:
    m = STATEMENT.search(user)
    return m.group(1).strip().strip('"') if m else ""


def _trace(user: str) -> str:
    """What the other checker wrote, from the C user turn: everything after the statement line."""

    m = STATEMENT.search(user)
    rest = user[m.end():] if m else user
    rest = re.sub(r"^Checker verdict:.*$", "", rest, flags=re.M)
    return rest.strip()


def cases_from_rows(rows: list[dict]) -> list[Case]:
    """Verdict-bearing items only: C, D (first turn, base material) and verdict-type E."""

    out: list[Case] = []
    for r in rows:
        kind = r.get("kind")
        material = r.get("material") or ""
        if not material:
            continue
        user = r["turns"][0]["user"]
        statement = _statement(user)
        if not statement:
            continue
        if kind == "C":
            out.append(Case(r["id"], "C", material, statement, r["verdict"], list(r.get("deciding") or []),
                            flaw=r.get("planted_flaw"), checker_trace=_trace(user), checker_verdict=r.get("checker_verdict")))
        elif kind == "D":
            keys = (r.get("turn_keys") or {}).get("1") or {"verdict": r["verdict"], "deciding": r.get("deciding") or []}
            out.append(Case(r["id"], "D", material, statement, keys["verdict"], list(keys["deciding"])))
        elif kind == "E" and (r.get("quiz") or {}).get("type") == "verdict":
            q = r["quiz"]
            out.append(Case(r["id"], "E", material, statement, q["verdict"], list(q.get("deciding") or [])))
    return out


def plant_errors(cases: list[Case], n: int, *, seed: int) -> list[Case]:
    """Copies of n cases with the key deliberately broken: a different verdict, and lines to match."""

    rng = random.Random(seed)
    pool = [c for c in cases if not c.planted]
    chosen = rng.sample(pool, min(n, len(pool)))
    planted = []
    for c in chosen:
        if c.verdict == "cannot_tell":
            first = next((line for line in c.material.splitlines() if line.strip()), c.material.strip())
            planted.append(Case(c.item_id + "+planted", c.kind, c.material, c.statement, "supported", [first], planted=True))
        else:
            planted.append(Case(c.item_id + "+planted", c.kind, c.material, c.statement, "cannot_tell", [], planted=True))
    return planted


def run_control(gates: Gates, critic, cases: list[Case], *, planted: list[Case] | None = None, log=None) -> ControlReport:
    report = ControlReport(items=len(cases))
    for case in list(cases) + list(planted or []):
        report.verdict_asks += 1
        if case.planted:
            report.planted += 1
        v = None
        for ask in range(2):
            c = critic(verdict_messages(case.material, case.statement))
            v = compare_verdict(gates, c.text, case.verdict, case.deciding, case.material)
            if v.code != "VOID":
                break
            report.voids += 1
        if case.planted:
            if not v.ok:
                report.catches += 1
            elif log:
                log({"missed_plant": case.item_id, "critic": v.critic_verdict})
            continue
        if v.ok:
            report.agree += 1
        else:
            report.false_refusals.append({"item": case.item_id, "code": v.code, "detail": v.detail, "critic_verdict": v.critic_verdict, "critic_deciding": v.critic_deciding})
        if case.kind == "C" and case.flaw and case.checker_trace and case.checker_verdict:
            report.flaw_asks += 1
            c = critic(flaw_messages(case.material, case.statement, case.checker_trace, case.checker_verdict))
            f = compare_flaw(c.text, case.flaw)
            if f.ok:
                report.flaw_agree += 1
            else:
                report.flaw_misses.append({"item": case.item_id, "detail": f.detail})
    return report


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
