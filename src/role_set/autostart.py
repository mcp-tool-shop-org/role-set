"""One paste starts a paid pilot's write: wait for the lane, write, and relaunch after a lost-call stop.

Pilot 2 lost about $1.70 of pod time waiting for a paste, and a relaunch the pod still had time for never ran,
because it needed a second paste. The launcher is approved once, before the pod serves. It polls the lane's
/v1/models until it answers, then runs scripts/pilot-write.sh in <work>/attempt-1. When an attempt ends on the
parse/transport rule (lost calls, not bad data) with more than RELAUNCH_MIN_WRITING minutes of writing left, it
starts the next attempt in its own directory, on the slots no earlier attempt reached, with the earlier attempts'
cell, kind and lesson halts carried over. Any other ending is final: budget, queue, kill, every kind off. The lane
must still answer before a relaunch, so a pod that is gone ends the run instead of feeding the transport rule.

launcher.json in the work dir records every attempt and, at the end, done: true. The session watching it reports
the end to the Publisher and Switch, who shut the pod down. Standards: the runner keeps its andon and pins;
pilot-write.sh keeps the time box and the critic unload; this module adds only the wait and the relaunch.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

POLL_S = 10.0
SHUTDOWN_MARGIN_MIN = 8  # pilot-write.sh's budget is the minutes to the deadline minus 8
WRITE_MIN = 5  # pilot-write.sh refuses a budget under 5 minutes
RELAUNCH_MIN_WRITING = 10  # the Publisher's rule: relaunch only while more than 10 minutes of writing remain
MAX_ATTEMPTS = 5


def deadline_epoch(hhmm: str, now: float) -> float:
    """Today's UTC date at HH:MM, as pilot-write.sh reads its deadline."""

    day = datetime.fromtimestamp(now, tz=timezone.utc).date()
    h, m = (int(x) for x in hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, tzinfo=timezone.utc).timestamp()


def writing_minutes_left(deadline: float, now: float) -> float:
    return (deadline - now) / 60 - SHUTDOWN_MARGIN_MIN


def _open(url: str, timeout: float) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - loopback only (the lane's tunnel)
        return resp.status


def lane_serves(port: int, *, opener=_open) -> bool:
    try:
        return opener(f"http://127.0.0.1:{port}/v1/models", 5.0) == 200
    except OSError:
        return False


def wait_for_lane(port: int, deadline: float, *, opener=_open, sleep=time.sleep, now=time.time) -> bool:
    """Poll the lane until it answers, while a write could still start; False when the window closes first."""

    while writing_minutes_left(deadline, now()) >= WRITE_MIN:
        if lane_serves(port, opener=opener):
            return True
        sleep(POLL_S)
    return False


def _status(attempt: Path) -> dict:
    reports = sorted((attempt / "out").rglob("batch-report.json"))
    return json.loads(reports[0].read_text(encoding="utf-8")).get("status", {}) if reports else {}


def remainder(plan: dict, attempts: list[Path]) -> tuple[dict, dict]:
    """The plan's slots that no earlier attempt reached, minus the cells, kinds and lessons its andon ended."""

    reached: set[str] = set()
    cells: set[str] = set()
    kinds: set[str] = set()
    lessons: set[str] = set()
    for a in attempts:
        for f in (a / "out").rglob("drafts.jsonl"):
            reached.update(json.loads(x)["slot"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip())
        st = _status(a)
        cells.update(st.get("cell_halts") or {})
        kinds.update(st.get("kinds_off") or {})
        lessons.update(k for k, v in (st.get("lesson_halts") or {}).items() if v.get("kind") == "*")
    keep, why = [], {"reached": 0, "cell halted": 0, "kind off": 0, "lesson halted": 0}
    for s in plan["slots"]:
        if s["id"] in reached:
            why["reached"] += 1
        elif s["lesson"] in lessons:
            why["lesson halted"] += 1
        elif s["kind"] in kinds:
            why["kind off"] += 1
        elif f"{s['lesson']}/{s['kind']}" in cells:
            why["cell halted"] += 1
        else:
            keep.append(s)
    return {**plan, "slots": keep}, why


def lost_call_stop(budget: dict) -> bool:
    """The parse/transport rule, not a data stop: the attempt lost calls, and its accepted items stand."""

    return budget.get("ended_by") == "systemic stop" and "writer calls" in str(budget.get("halted") or "")


def _pilot_write(port: int, grant: str, hhmm: str, work: Path) -> int:
    return subprocess.call(["sh", "scripts/pilot-write.sh", str(port), grant, hhmm, str(work)])


def run(port: int, grant: str, hhmm: str, work: Path, *, write=_pilot_write, opener=_open, sleep=time.sleep, now=time.time) -> dict:
    deadline = deadline_epoch(hhmm, now())
    state: dict = {"port": port, "grant": grant, "deadline_utc": hhmm, "attempts": [], "done": False}
    record = work / "launcher.json"

    def save() -> None:
        record.write_text(json.dumps(state, indent=1), encoding="utf-8")

    save()
    plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))
    dirs: list[Path] = []
    ended = "lane never served"
    for n in range(1, MAX_ATTEMPTS + 1):
        if not wait_for_lane(port, deadline, opener=opener, sleep=sleep, now=now):
            ended = "lane never served" if n == 1 else "lane gone before a relaunch"
            break
        d = work / f"attempt-{n}"
        d.mkdir(exist_ok=True)
        attempt_plan, carried = (plan, {}) if n == 1 else remainder(plan, dirs)
        if not attempt_plan["slots"]:
            ended = "no slots left"
            break
        (d / "plan.json").write_text(json.dumps(attempt_plan, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copytree(work / "pack", d / "pack", dirs_exist_ok=True)
        shutil.copy2(work / "pool.jsonl", d / "pool.jsonl")
        state["lane_served_at"] = state.get("lane_served_at") or round(now())
        rc = write(port, grant, hhmm, d)
        dirs.append(d)
        budget_file = d / "budget.json"
        budget = json.loads(budget_file.read_text(encoding="utf-8")) if budget_file.exists() else {}
        st = _status(d)
        state["attempts"].append({"dir": d.name, "slots": len(attempt_plan["slots"]), "carried_over": carried, "exit_code": rc,
                                  "ended_by": budget.get("ended_by") or "refused", "halted": budget.get("halted"),
                                  "accepted": st.get("accepted"), "attempted": st.get("attempted"),
                                  "critic_unloaded": budget.get("critic_unloaded"), "ollama_ps_after": budget.get("ollama_ps_after")})
        save()
        ended = budget.get("ended_by") or "refused"
        if not lost_call_stop(budget):
            break
        left = writing_minutes_left(deadline, now())
        if left <= RELAUNCH_MIN_WRITING:
            ended = f"systemic stop; no relaunch with {left:.0f} minutes of writing left"
            break
    else:
        ended = f"systemic stop; {MAX_ATTEMPTS} attempts used"
    last = state["attempts"][-1] if state["attempts"] else {}
    state.update(done=True, ended=ended, accepted=sum(a.get("accepted") or 0 for a in state["attempts"]),
                 card_free=bool(last.get("critic_unloaded", True)), finished_at=round(now()))
    save()
    return state
