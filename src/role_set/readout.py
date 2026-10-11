"""`role-set readout`: one run's results, readable whether it passed or failed, beside earlier runs.

Reads only the run's own output directory (batch-report.json, drafts.jsonl, refusals.jsonl, receipts.jsonl); runs
from before drafts.jsonl existed fall back to the batch report. Per lesson and per kind: the accept rate; a refusal
histogram (check_against refusals by their gate code); the retry recovery rate (slots refused at least once that
were accepted later); glyphs normalised; tokens and seconds per writer call; the frozen stuck rule's flags on the
drafts; and the critic's agreement. Nothing here reads the store or anything held out.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def summarize(run_dir: Path) -> dict:
    report = json.loads((run_dir / "batch-report.json").read_text(encoding="utf-8"))
    status, outcomes = report.get("status", {}), report.get("outcomes", [])
    if not outcomes:  # a run that halted before the runner kept its outcomes: rebuild them from the action log
        tried: dict[str, dict] = {}
        for a in status.get("actions", []):
            if a.get("action") == "refused":
                tried.setdefault(a["slot"], {"slot": a["slot"], "accepted": False, "refusals": []})["refusals"].append({"code": a["code"]})
            elif a.get("action") in ("accepted", "dropped"):
                tried.setdefault(a["slot"], {"slot": a["slot"], "accepted": False, "refusals": []})["accepted"] = a["action"] == "accepted"
        outcomes = list(tried.values())
    calls = _lines(run_dir / "drafts.jsonl")
    refusals = _lines(run_dir / "refusals.jsonl")
    receipts = _lines(run_dir / "receipts.jsonl")

    per: dict[str, dict] = {"lesson": defaultdict(Counter), "kind": defaultdict(Counter)}
    reasons: dict[str, Counter] = {"lesson": defaultdict(Counter), "kind": defaultdict(Counter)}
    recovered = refused_slots = 0
    for o in outcomes:
        lesson, kind = (o["slot"].split("-") + ["?", "?"])[:2]
        state = "skipped" if o.get("skipped") else ("accepted" if o["accepted"] else "dropped")
        for axis, key in (("lesson", lesson), ("kind", kind)):
            per[axis][key][state] += 1
            if state == "skipped":
                reasons[axis][key][o.get("skip_reason") or "unrecorded"] += 1
        if o.get("refusals"):
            refused_slots += 1
            recovered += bool(o["accepted"])

    def table(axis: str) -> dict:
        out = {}
        for key, c in sorted(per[axis].items()):
            tried = c["accepted"] + c["dropped"]
            out[key] = {"accepted": c["accepted"], "dropped": c["dropped"], "skipped": c["skipped"], "accept_rate": _rate(c["accepted"], tried),
                        "skipped_by_reason": dict(reasons[axis][key].most_common())}
        return out

    codes = Counter()
    for r in refusals:
        codes[r["detail"].split(":")[0] if r["code"] == "AGAINST" else r["code"]] += 1
    if not refusals:  # a run from before refusals.jsonl
        codes.update(status.get("refused_by_code", {}))

    seconds = [c["seconds"] for c in calls if c.get("seconds")]
    stuck = [c["stuck"] for c in calls if isinstance(c.get("stuck"), dict)]
    glyphs = sum(r.get("glyphs_normalized", 0) for r in receipts) + sum(c.get("glyphs_normalized", 0) for c in calls if c.get("outcome") != "accepted")
    tokens = status.get("tokens", {})
    n_calls = len(calls) or sum(1 for a in status.get("actions", []) if a.get("action") in ("accepted", "refused")) or None
    return {
        "run": "/".join(run_dir.parts[-3:]) if run_dir.name in ("rehearsal", "mock") else run_dir.name,
        "halted": status.get("halted"),
        "lesson_halts": status.get("lesson_halts", {}),
        "kinds_off": status.get("kinds_off", {}),
        "skipped": status.get("skipped", {}),
        "batch_check": report.get("batch_check"),
        "accepted": status.get("accepted"), "dropped": status.get("dropped"), "attempted": status.get("attempted"),
        "accept_rate": _rate(status.get("accepted", 0), status.get("attempted", 0)),
        "per_lesson": table("lesson"), "per_kind": table("kind"),
        "refusals": dict(codes.most_common()),
        "retry_recovery": {"refused_slots": refused_slots, "recovered": recovered, "rate": _rate(recovered, refused_slots)},
        "glyphs_normalized": glyphs,
        "calls": len(calls) or sum(1 for a in status.get("actions", []) if a.get("action") in ("accepted", "refused")),
        "tokens_per_call": {"prompt": round(tokens.get("prompt", 0) / n_calls) if n_calls else None,
                            "completion": round(tokens.get("completion", 0) / n_calls) if n_calls else None},
        "seconds_per_call": {"mean": round(statistics.mean(seconds), 2) if seconds else None, "p50": _pct(seconds, 0.5), "p95": _pct(seconds, 0.95)},
        "stuck": {"drafts_measured": len(stuck), "stuck": sum(1 for s in stuck if s.get("stuck")),
                  "with_burst": sum(1 for s in stuck if s.get("bursts")), "hedges_mean": round(statistics.mean(s.get("hedges", 0) for s in stuck), 2) if stuck else None},
        "critic_agreement": status.get("critic_agreement"),
        "elapsed_s": status.get("elapsed_s"),
    }


def markdown(main: dict, others: list[dict]) -> str:
    lines = [f"# Readout: {main['run']}", ""]
    lines.append(f"Accepted {main['accepted']} of {main['attempted']} attempted (accept rate {main['accept_rate']}); "
                 f"halted: {main['halted'] or 'no'}; lesson halts: {main['lesson_halts'] or 'none'}; skipped: {main['skipped'] or 'none'}; "
                 f"batch check: {main['batch_check']}.")
    def why(v: dict) -> str:
        return "; ".join(f"{r}: {n}" for r, n in v.get("skipped_by_reason", {}).items()) or "-"

    if main.get("kinds_off"):
        lines += ["", "Kinds switched off: " + "; ".join(f"{k} ({r})" for k, r in main["kinds_off"].items()) + "."]
    lines += ["", "| Lesson | Accepted | Dropped | Skipped | Accept rate | Skipped by reason |", "|---|---|---|---|---|---|"]
    lines += [f"| {k} | {v['accepted']} | {v['dropped']} | {v['skipped']} | {v['accept_rate']} | {why(v)} |" for k, v in main["per_lesson"].items()]
    lines += ["", "| Kind | Accepted | Dropped | Skipped | Accept rate |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['accepted']} | {v['dropped']} | {v['skipped']} | {v['accept_rate']} |" for k, v in main["per_kind"].items()]
    lines += ["", "| Refusal | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in main["refusals"].items()]
    rr = main["retry_recovery"]
    lines += ["", f"Retry recovery: {rr['recovered']} of {rr['refused_slots']} refused slots accepted later ({rr['rate']}). "
              f"Glyphs normalised: {main['glyphs_normalized']}. Critic agreement: {main['critic_agreement']}.",
              f"Writer calls: {main['calls']}; tokens per call {main['tokens_per_call']}; seconds per call {main['seconds_per_call']}; "
              f"stuck rule on drafts {main['stuck']}."]
    if others:
        cols = [main] + others
        lines += ["", "| Run | Accept rate | Calls | Prompt tok/call | Completion tok/call | s/call (p50) | Top refusal |", "|---|---|---|---|---|---|---|"]
        for c in cols:
            top = next(iter(c["refusals"].items()), ("-", 0))
            lines.append(f"| {c['run']} | {c['accept_rate']} | {c['calls']} | {c['tokens_per_call']['prompt']} | {c['tokens_per_call']['completion']} | "
                         f"{c['seconds_per_call']['p50']} | {top[0]} ({top[1]}) |")
    return "\n".join(lines) + "\n"


def readout(run_dir: Path, compare: list[Path] | None = None) -> tuple[dict, str]:
    main = summarize(run_dir)
    others = [summarize(p) for p in (compare or [])]
    return {"run": main, "compare": others}, markdown(main, others)
