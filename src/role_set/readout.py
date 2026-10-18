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

from role_set.defects import open_classes
from role_set.runner import refusal_rule


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


# Older runs carry the Ollama digests; runs on the proven list carry the GGUF and llama-server pins (BUILD-SPEC B6).
INPUT_PINS = ("partition", "wave", "writer_model", "writer_family", "writer_repo", "writer_revision", "writer_gguf_sha256", "writer_digest",
              "critic_model", "critic_family", "critic_gguf_sha256", "critic_digest", "llama_server_commit", "llama_server_sha256",
              "pack_commit", "pack_sha256", "plan_sha256", "exemplar_commit", "stuck_rule_sha256", "grant", "rehearsal", "mock")


def _inputs(receipts: list[dict]) -> dict:
    """What the run was pinned to, from its receipts (every receipt carries the same pins). Exemplars are drawn from
    other traits only: the runner never shows a slot an approved item of its own trait."""

    if not receipts:
        return {}
    first = receipts[0]
    out = {k: first.get(k) for k in INPUT_PINS if first.get(k) is not None}
    if "exemplar_commit" in out:
        out["exemplars"] = "other traits only (a slot never sees approved items of its own trait)"
    return out


def keyed_letters(rows: list[dict]) -> dict:
    """Per lesson, how often each letter is keyed in accepted closed-form written variants: the varied-letter rule is
    otherwise unchecked (R&D on aspire-si #91)."""

    out: dict[str, Counter] = {}
    for r in rows:
        key = (r.get("variant") or {}).get("key")
        if (r.get("quiz") or {}).get("type") == "written" and isinstance(key, dict) and "answer" in key:
            out.setdefault(r.get("lesson", "?"), Counter())[key["answer"]] += 1
    return {lesson: dict(c) for lesson, c in sorted(out.items())}


def keyed_longest(rows: list[dict]) -> dict:
    """Per lesson, of the accepted closed-form written variants, how many have the keyed option as the strictly longest:
    a length cue a model could pick up instead of the trait (R&D on #27)."""

    from role_set.assemble import OPTION_LABEL

    out: dict[str, list[int]] = {}
    for r in rows:
        v = r.get("variant") or {}
        key = v.get("key")
        if (r.get("quiz") or {}).get("type") != "written" or not isinstance(key, dict) or "answer" not in key:
            continue
        opts = {line[0]: OPTION_LABEL.sub("", line) for line in str(v.get("user", "")).splitlines() if OPTION_LABEL.match(line)}
        if key["answer"] not in opts:
            continue
        mine = len(opts[key["answer"]])
        longest = all(mine > len(t) for k, t in opts.items() if k != key["answer"])
        cell = out.setdefault(r.get("lesson", "?"), [0, 0])
        cell[0] += longest
        cell[1] += 1
    return {lesson: {"keyed_longest": a, "items": n} for lesson, (a, n) in sorted(out.items())}


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
        # The halts' (kind, rule) keys, so GATE splits by the gate rule that fired.
        kind = (r["slot"].split("-") + ["?", "?"])[1]
        codes[f"{kind} / {refusal_rule(r)}"] += 1
    if not refusals:  # a run from before refusals.jsonl
        codes.update(status.get("refused_by_code", {}))

    seconds = [c["seconds"] for c in calls if c.get("seconds")]
    stuck = [c["stuck"] for c in calls if isinstance(c.get("stuck"), dict)]
    glyphs = sum(r.get("glyphs_normalized", 0) for r in receipts) + sum(c.get("glyphs_normalized", 0) for c in calls if c.get("outcome") != "accepted")
    tokens = status.get("tokens", {})
    n_calls = len(calls) or sum(1 for a in status.get("actions", []) if a.get("action") in ("accepted", "refused")) or None
    is_stub = bool(report.get("stub") or ((report.get("pins") or {}).get("machine") or {}).get("stub")
                   or (receipts and (receipts[0].get("machine") or {}).get("stub")))
    return {
        "run": "/".join(run_dir.parts[-3:]) if run_dir.name in ("rehearsal", "mock") else run_dir.name,
        **({"stub": True} if is_stub else {}),
        "halted": status.get("halted"),
        "lesson_halts": status.get("lesson_halts", {}),
        "cell_halts": status.get("cell_halts", {}),
        "warmups": dict(Counter("passed" if w.get("passed") else "missed" for w in _lines(run_dir / "warmups.jsonl"))),
        "keyed_letters": keyed_letters(_lines(run_dir / "accepted.jsonl")),
        "keyed_longest": keyed_longest(_lines(run_dir / "accepted.jsonl")),
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
        "inputs": _inputs(receipts),
    }


def markdown(main: dict, others: list[dict]) -> str:
    head = f"# Readout (STUB): {main['run']}" if main.get("stub") else f"# Readout: {main['run']}"
    lines = [head, ""]
    if main.get("inputs"):
        lines += ["Inputs: " + "; ".join(f"{k} {v}" for k, v in main["inputs"].items()) + ".", ""]
    lines.append(f"Accepted {main['accepted']} of {main['attempted']} attempted (accept rate {main['accept_rate']}); "
                 f"halted: {main['halted'] or 'no'}; lesson halts: {main['lesson_halts'] or 'none'}; skipped: {main['skipped'] or 'none'}; "
                 f"batch check: {main['batch_check']}.")
    def why(v: dict) -> str:
        return "; ".join(f"{r}: {n}" for r, n in v.get("skipped_by_reason", {}).items()) or "-"

    if main.get("cell_halts"):
        lines += ["", "Lesson-kind cells halted: " + "; ".join(f"{c} ({h['rule']})" for c, h in sorted(main["cell_halts"].items())) + "."]
    if main.get("warmups"):
        lines += ["", f"Warm-ups: {main['warmups'].get('passed', 0)} passed, {main['warmups'].get('missed', 0)} missed."]
    if main.get("keyed_letters"):
        lines += ["", "Closed-form keyed letters (accepted written E): " + "; ".join(
            f"{lesson} " + " ".join(f"{k}{n}" for k, n in sorted(c.items())) for lesson, c in sorted(main["keyed_letters"].items())) + "."]
    if main.get("keyed_longest"):
        lines += ["", "Keyed option is the longest: " + "; ".join(
            f"{lesson} {c['keyed_longest']} of {c['items']}" for lesson, c in sorted(main["keyed_longest"].items())) + "."]
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


def open_defects_markdown(rows: list[dict]) -> str:
    """The catalog's `none` rows: defect classes that no check covers yet, listed on every readout."""

    if not rows:
        return "\nOpen defect classes (no check yet): none.\n"
    return "\nOpen defect classes (no check yet, from roles/verifier/defects.jsonl):\n" + "".join(
        f"- {r['name']} ({r['id']}): {r['description']} Candidate: {r['anchor']}\n" for r in rows)


def readout(run_dir: Path, compare: list[Path] | None = None, catalog: list[dict] | None = None) -> tuple[dict, str]:
    main = summarize(run_dir)
    others = [summarize(p) for p in (compare or [])]
    data, md = {"run": main, "compare": others}, markdown(main, others)
    if catalog is not None:
        gaps = open_classes(catalog)
        data["open_defects"] = [r["name"] for r in gaps]
        md += open_defects_markdown(gaps)
    return data, md
