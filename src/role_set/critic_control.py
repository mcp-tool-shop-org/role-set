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
    # Every case's outcome, in order: run-to-run agreement is read from two runs' lists (R&D, 2026-10-17).
    per_item: list = field(default_factory=list)
    seed: int = 0

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
            "seed": self.seed,
            "per_item": self.per_item,
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


def _ask(gates: Gates, critic, case: Case, seed: int) -> dict:
    """One case through the critic: the verdict (asked again once on a VOID), and the flaw for a kind-C item."""

    out: dict = {"voids": 0, "flaw": None}
    v = None
    for ask in range(2):
        # Each run's seeds are spaced apart, so run 0's re-ask never shares run 1's first seed (R&D on #85).
        c = critic(verdict_messages(case.material, case.statement), seed=seed * 1000 + ask)
        v = compare_verdict(gates, c.text, case.verdict, case.deciding, case.material)
        if v.code != "VOID":
            break
        out["voids"] += 1
    out["v"] = v
    if not case.planted and case.kind == "C" and case.flaw and case.checker_trace and case.checker_verdict:
        c = critic(flaw_messages(case.material, case.statement, case.checker_trace, case.checker_verdict), seed=seed * 1000 + 500)
        out["flaw"] = compare_flaw(c.text, case.flaw)
    return out


def run_control(gates: Gates, critic, cases: list[Case], *, planted: list[Case] | None = None, log=None, seed: int = 0,
                concurrency: int = 1) -> ControlReport:
    """Every case through the critic, at `seed` (a second run at another seed measures run-to-run agreement for a
    sampling critic). Cases may run concurrently; the report is assembled in case order either way."""

    from concurrent.futures import ThreadPoolExecutor

    report = ControlReport(items=len(cases), seed=seed)
    allcases = list(cases) + list(planted or [])
    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            outcomes = list(pool.map(lambda c: _ask(gates, critic, c, seed), allcases))
    else:
        outcomes = [_ask(gates, critic, c, seed) for c in allcases]
    for case, o in zip(allcases, outcomes):
        v = o["v"]
        report.verdict_asks += 1
        report.voids += o["voids"]
        report.per_item.append({"item": case.item_id, "kind": case.kind, "planted": case.planted, "key": case.verdict,
                                "critic_verdict": v.critic_verdict, "code": v.code, "ok": v.ok,
                                "flaw_ok": o["flaw"].ok if o["flaw"] is not None else None})
        if case.planted:
            report.planted += 1
            if not v.ok:
                report.catches += 1
            elif log:
                log({"missed_plant": case.item_id, "critic": v.critic_verdict})
            continue
        if v.ok:
            report.agree += 1
        else:
            report.false_refusals.append({"item": case.item_id, "code": v.code, "detail": v.detail, "critic_verdict": v.critic_verdict, "critic_deciding": v.critic_deciding})
        if o["flaw"] is not None:
            report.flaw_asks += 1
            if o["flaw"].ok:
                report.flaw_agree += 1
            else:
                report.flaw_misses.append({"item": case.item_id, "detail": o["flaw"].detail})
    return report


def run_agreement(a: dict, b: dict) -> dict:
    """Two runs' per-item lists: how often the critic's verdict differs between them, item by item, with the unplanted
    items (where R&D's bar sits) and the planted ones reported apart."""

    left = {r["item"]: r for r in a["per_item"]}
    right = {r["item"]: r for r in b["per_item"]}
    shared = sorted(set(left) & set(right))

    def part(ids):
        differ = [i for i in ids if left[i]["critic_verdict"] != right[i]["critic_verdict"]]
        return {"items": len(ids), "differ": len(differ), "rate": round(len(differ) / len(ids), 3) if ids else None, "differing": differ}

    return {"unplanted": part([i for i in shared if not left[i]["planted"]]), "planted": part([i for i in shared if left[i]["planted"]]),
            "seeds": [a.get("seed"), b.get("seed")]}


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---- the trait leg ---------------------------------------------------------------------------------------------
# R&D's bar (2026-10-11), on reviewed items whose trait is known (the slot's lesson), majority of three calls: overall
# agreement at least 85%, every trait at least 70%, and the three calls unanimous on at least 90% of items. If the
# critic misses it, its prompt changes before the trait check gates again; the gate is never relaxed to fit the critic.
TRAIT_BAR = {"overall": 0.85, "per_trait": 0.70, "stability": 0.90}
# Declared by R&D before dev iteration 2 was scored (2026-10-11): sibling pairs one step apart in meaning. Agreement
# within a family is reported beside exact agreement and never gates by itself. T02/T14 and T05/T02 are deliberately
# not families: the pilot's failure was T02 items decided by a silence call, and the check must keep catching that.
TRAIT_FAMILIES = (frozenset({"T05", "T13"}), frozenset({"T03", "T16"}), frozenset({"T01", "T11"}), frozenset({"T04", "T14"}),
                  frozenset({"T10", "T16"}))


def same_family(named: str | None, known: str) -> bool:
    return named == known or any({named, known} <= f for f in TRAIT_FAMILIES)


def trait_leg(critic, rows: list[dict], trait_names: dict[str, str], steps: dict[str, str] | None = None, *,
              concurrency: int = 1, log=None) -> dict:
    """Ask the critic, three times each, which trait every reviewed item teaches, and score it against the known trait."""

    from concurrent.futures import ThreadPoolExecutor

    from role_set.critic import TRAIT_TEMPERATURE, TRAIT_VOTES, majority, teaching_text, trait_messages, trait_named

    items = [r for r in rows if r.get("lesson") in trait_names and r.get("turns")]
    seconds: list[float] = []  # per call, for sizing the critic queue in a paid run

    def ask(r: dict) -> tuple[str, list[str]]:
        msgs = trait_messages(teaching_text(r), trait_names, steps)
        votes = []
        for s in range(TRAIT_VOTES):
            c = critic(msgs, temperature=TRAIT_TEMPERATURE, seed=s)
            seconds.append(float(getattr(c, "seconds", 0.0) or 0.0))
            votes.append(trait_named(c.text))
        return r["lesson"], votes

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            answers = list(pool.map(ask, items))
    else:
        answers = [ask(r) for r in items]
    per: dict[str, dict] = {}
    confusion: dict[str, dict] = {}
    agree = unanimous = with_majority = family = 0
    for r, (known, votes) in zip(items, answers):
        won = majority(votes)
        cell = per.setdefault(known, {"items": 0, "agree": 0, "family_agree": 0})
        cell["items"] += 1
        if same_family(won, known):
            family += 1
            cell["family_agree"] += 1
        named = won or "split"
        confusion.setdefault(known, {})[named] = confusion.setdefault(known, {}).get(named, 0) + 1
        # Unanimous only on a trait id: three unreadable replies agree on nothing (R&D on #23).
        unanimous += len(set(votes)) == 1 and bool(votes[0])
        with_majority += won is not None
        if won == known:
            agree += 1
            cell["agree"] += 1
        elif log:
            log({"item": r.get("id"), "known": known, "votes": votes})
    n = len(items)
    for cell in per.values():
        cell["rate"] = round(cell["agree"] / cell["items"], 3)
        cell["thin"] = cell["items"] < THIN_ITEMS
    overall = round(agree / n, 3) if n else None
    stability = round(unanimous / n, 3) if n else None
    worst = min((c["rate"] for c in per.values()), default=None)
    passes = bool(n) and overall >= TRAIT_BAR["overall"] and worst >= TRAIT_BAR["per_trait"] and stability >= TRAIT_BAR["stability"]
    return {"items": n, "agree": agree, "overall": overall, "per_trait": dict(sorted(per.items())), "worst_trait": worst,
            "family_agreement": round(family / n, 3) if n else None,  # reported only; never part of `passes`
            "families": [sorted(f) for f in TRAIT_FAMILIES],
            "stability_unanimous": stability, "with_majority": round(with_majority / n, 3) if n else None,
            "seconds_per_call": _timing(seconds),
            "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())}, "bar": TRAIT_BAR, "passes": passes}


THIN_ITEMS = 10  # a per-trait rate on fewer items than this is marked thin
ANSWER_BAR = 0.90  # R&D (2026-10-11): the critic agrees with at least 90% of reviewed B keys


def _timing(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    return {"calls": len(ordered), "p50": round(ordered[len(ordered) // 2], 2) if ordered else None,
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2) if ordered else None,
            "total": round(sum(ordered), 1)}


def answer_leg(critic, rows: list[dict]) -> dict:
    """The kind-B answer check on reviewed puzzles: the critic solves each one with the closed-answer prompt, and its
    ANSWER is compared with the reviewed key. Two subsets, per R&D's ruling: keys closed as written, and older keys whose
    head before a dash or colon is exactly a verdict label (the label is the answer). Every other key is listed and left
    out; every disagreement is listed. The bar applies to the combined figure."""

    from role_set.critic import ANSWER_LINE, answer_messages, compare_answer

    from role_set.assemble import names_not_in_puzzle

    puzzles = [r for r in rows if r.get("kind") == "B" and r.get("turns")]
    keyed = [(r, *leg_key(str(r.get("answer", "")))) for r in puzzles]
    # A reviewed key in descriptive form ("the red lamp" where the puzzle names colours) can't be compared closed; it is
    # listed and left out until R&D restates it in the puzzle's own names (R&D, 2026-10-11).
    descriptive = [r.get("id") for r, key, subset in keyed if subset == "as_written" and names_not_in_puzzle(key, r["turns"][0]["user"])]
    seconds: list[float] = []
    voids, disagreements = 0, []
    tally = {"as_written": [0, 0], "label_head": [0, 0]}  # subset -> [items, agree]
    for r, key, subset in keyed:
        if key is None or r.get("id") in descriptive:
            continue
        v, c = None, None
        for ask in range(2):
            c = critic(answer_messages(r["turns"][0]["user"], key))
            seconds.append(float(getattr(c, "seconds", 0.0) or 0.0))
            v = compare_answer(c.text, key)
            if v.code != "VOID":
                break
            voids += 1
        tally[subset][0] += 1
        if v.ok:
            tally[subset][1] += 1
        else:
            said = ANSWER_LINE.findall(c.text)
            disagreements.append({"item": r.get("id"), "subset": subset, "key": r["answer"], "compared_as": key, "code": v.code,
                                  "critic_answer": said[-1] if said else None, "critic_raw": c.text[:2000]})

    def part(items: int, agree: int) -> dict:
        return {"items": items, "agree": agree, "rate": round(agree / items, 3) if items else None}

    subsets = {k: part(*v) for k, v in tally.items()}
    combined = part(sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values()))
    written = subsets["as_written"]
    return {"puzzles": len(puzzles), "keys_left_out": [r.get("id") for r, key, _ in keyed if key is None],
            "descriptive_keys_left_out": descriptive, "subsets": subsets,
            "combined": combined, "voids": voids, "seconds_per_call": _timing(seconds), "bar": ANSWER_BAR,
            "passes": bool(combined["items"]) and combined["rate"] >= ANSWER_BAR,
            "as_written_below_bar": bool(written["items"]) and written["rate"] < ANSWER_BAR, "disagreements": disagreements}


LABEL_HEAD = re.compile(r"\s[—–-]\s|:")


def leg_key(answer: str) -> tuple[str | None, str | None]:
    """The key the answer leg compares against, and its subset: the answer itself when it is closed as written; the bare
    label when an older 'label — reason' key's head is exactly a verdict label; otherwise (None, None)."""

    from role_set.assemble import VERDICTS, closed_answer

    if closed_answer(answer) is not None:
        return answer, "as_written"
    head = LABEL_HEAD.split(answer, maxsplit=1)[0].strip().strip(".*").lower()
    if head in VERDICTS:
        return head, "label_head"
    return None, None
