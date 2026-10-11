"""The one-paste launcher: wait for the lane, write, relaunch after a lost-call stop, and record every attempt."""

import json
from datetime import datetime, timezone
from pathlib import Path

from role_set import autostart
from role_set.cli import main

T0 = datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc).timestamp()


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _work(tmp_path: Path, n_slots=6) -> Path:
    w = tmp_path / "w"
    (w / "pack").mkdir(parents=True)
    (w / "pack" / "manifest.json").write_text("{}")
    (w / "pool.jsonl").write_text("")
    slots = [{"id": f"T0{i % 2 + 1}-{k}-0{i}", "lesson": f"T0{i % 2 + 1}", "kind": k} for i in range(n_slots) for k in ("A", "B")]
    (w / "plan.json").write_text(json.dumps({"slots": slots, "figures": {}}))
    return w


def _fake_write(clock, endings, minutes=5):
    """Each attempt reaches its plan's first two slots, accepts one, and ends as `endings` says."""

    seen = []

    def write(port, grant, hhmm, d):
        plan = json.loads((d / "plan.json").read_text())
        seen.append([s["id"] for s in plan["slots"]])
        ended, halted, status = endings[len(seen) - 1]
        out = d / "out"
        out.mkdir()
        reached = plan["slots"][:2]
        (out / "drafts.jsonl").write_text("".join(json.dumps({"slot": s["id"]}) + "\n" for s in reached))
        (out / "batch-report.json").write_text(json.dumps({"status": {"accepted": 1, "attempted": 2, **status}}))
        (d / "budget.json").write_text(json.dumps({"ended_by": ended, "halted": halted, "critic_unloaded": True, "ollama_ps_after": "empty"}))
        clock.t += minutes * 60
        return 2 if ended == "systemic stop" else 0

    return write, seen


PARSE_STOP = ("systemic stop", "systemic: 5 of the last 20 writer calls failed to parse or reach the writer", {})


def test_it_waits_for_the_lane_then_relaunches_on_the_remaining_slots(tmp_path: Path):
    clock = Clock()
    w = _work(tmp_path)
    answers = iter([OSError("refused"), 503, 200, 200, 200])

    def opener(url, timeout):
        assert url == "http://127.0.0.1:11506/v1/models"
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    write, seen = _fake_write(clock, [PARSE_STOP, ("systemic stop", PARSE_STOP[1], {"cell_halts": {"T01/B": {}}}), ("queue", None, {})])
    state = autostart.run(11506, "G", "09:48", w, write=write, opener=opener, sleep=clock.sleep, now=clock)
    assert state["lane_served_at"] == T0 + 20  # two polls of 10 s before the lane answered
    assert [a["dir"] for a in state["attempts"]] == ["attempt-1", "attempt-2", "attempt-3"]
    assert len(seen[0]) == 12 and seen[1] == seen[0][2:] and not set(seen[0][:2]) & set(seen[1])
    # The second attempt's halted cell carries over: no T01 B slot in the third attempt.
    assert not any(s.startswith("T01-B") for s in seen[2]) and state["attempts"][2]["carried_over"]["cell halted"] > 0
    assert state["done"] and state["ended"] == "queue" and state["accepted"] == 3 and state["card_free"]
    assert json.loads((w / "launcher.json").read_text())["done"] is True
    assert (w / "attempt-2" / "pack" / "manifest.json").exists() and (w / "attempt-2" / "pool.jsonl").exists()


def test_a_data_stop_is_final(tmp_path: Path):
    clock = Clock()
    write, seen = _fake_write(clock, [("systemic stop", "systemic: every kind is off", {})])
    state = autostart.run(11506, "G", "09:48", _work(tmp_path), write=write, opener=lambda u, t: 200, sleep=clock.sleep, now=clock)
    assert len(seen) == 1 and state["ended"] == "systemic stop"


def test_no_relaunch_with_ten_minutes_of_writing_or_less(tmp_path: Path):
    clock = Clock()
    write, seen = _fake_write(clock, [PARSE_STOP, ("queue", None, {})], minutes=25)  # 48 - 25 - 8 = 15 minutes left
    state = autostart.run(11506, "G", "09:48", _work(tmp_path), write=write, opener=lambda u, t: 200, sleep=clock.sleep, now=clock)
    assert len(seen) == 2  # 15 minutes left: relaunched
    clock2 = Clock()
    write2, seen2 = _fake_write(clock2, [PARSE_STOP], minutes=31)  # 48 - 31 - 8 = 9 minutes left
    state = autostart.run(11506, "G", "09:48", _work(tmp_path / "b"), write=write2, opener=lambda u, t: 200, sleep=clock2.sleep, now=clock2)
    assert len(seen2) == 1 and state["ended"] == "systemic stop; no relaunch with 9 minutes of writing left"


def test_a_lane_that_never_serves_or_goes_away_ends_the_run(tmp_path: Path):
    clock = Clock()
    write, seen = _fake_write(clock, [])
    state = autostart.run(11506, "G", "09:48", _work(tmp_path), write=write, opener=lambda u, t: 503, sleep=clock.sleep, now=clock)
    assert seen == [] and state["ended"] == "lane never served" and state["attempts"] == []
    assert clock.t >= T0 + (48 - 8 - 5) * 60  # it polled until a write could no longer start

    clock = Clock()
    answers = iter([200, 503] + [503] * 1000)
    write, seen = _fake_write(clock, [PARSE_STOP])
    state = autostart.run(11506, "G", "09:48", _work(tmp_path / "b"), write=write, opener=lambda u, t: next(answers), sleep=clock.sleep, now=clock)
    assert len(seen) == 1 and state["ended"] == "lane gone before a relaunch"


def test_attempts_are_capped_and_an_empty_remainder_ends_it(tmp_path: Path):
    clock = Clock()
    write, seen = _fake_write(clock, [PARSE_STOP] * 10, minutes=1)
    state = autostart.run(11506, "G", "09:48", _work(tmp_path, n_slots=50), write=write, opener=lambda u, t: 200, sleep=clock.sleep, now=clock)
    assert len(seen) == autostart.MAX_ATTEMPTS and state["ended"] == f"systemic stop; {autostart.MAX_ATTEMPTS} attempts used"

    clock = Clock()
    write, seen = _fake_write(clock, [PARSE_STOP] * 10, minutes=1)
    state = autostart.run(11506, "G", "09:48", _work(tmp_path / "b", n_slots=1), write=write, opener=lambda u, t: 200, sleep=clock.sleep, now=clock)
    assert len(seen) == 1 and state["ended"] == "no slots left"


def test_a_refused_write_and_the_lane_probe(tmp_path: Path):
    clock = Clock()
    state = autostart.run(11506, "G", "09:48", _work(tmp_path), write=lambda *a: 3, opener=lambda u, t: 200, sleep=clock.sleep, now=clock)
    assert state["ended"] == "refused" and state["attempts"][0]["ended_by"] == "refused"
    assert autostart.lane_serves(1, opener=lambda u, t: (_ for _ in ()).throw(OSError("x"))) is False
    assert autostart.lost_call_stop({"ended_by": "systemic stop", "halted": "systemic: every kind is off"}) is False


def test_lesson_and_kind_halts_carry_over(tmp_path: Path):
    a = tmp_path / "attempt-1"
    (a / "out").mkdir(parents=True)
    (a / "out" / "batch-report.json").write_text(json.dumps({"status": {"lesson_halts": {"T01": {"kind": "*"}}, "kinds_off": {"B": {}}}}))
    plan = {"slots": [{"id": "T01-A-01", "lesson": "T01", "kind": "A"}, {"id": "T02-B-01", "lesson": "T02", "kind": "B"},
                      {"id": "T02-A-01", "lesson": "T02", "kind": "A"}]}
    left, why = autostart.remainder(plan, [a])
    assert [s["id"] for s in left["slots"]] == ["T02-A-01"] and why["lesson halted"] == 1 and why["kind off"] == 1


def test_the_cli_runs_the_launcher(tmp_path: Path, monkeypatch, capsys):
    seen = {}

    def fake_run(port, grant, hhmm, work):
        seen.update(port=port, grant=grant, hhmm=hhmm, work=work)
        return {"done": True, "ended": "queue", "attempts": [{"dir": "attempt-1", "ended_by": "queue", "accepted": 3, "attempted": 4, "halted": None}]}

    monkeypatch.setattr(autostart, "run", fake_run)
    assert main(["pilot-auto", "11506", "G", "09:48", str(tmp_path)]) == 0
    assert seen == {"port": 11506, "grant": "G", "hhmm": "09:48", "work": tmp_path}
    assert "attempt-1: queue; accepted 3 of 4" in capsys.readouterr().out
    monkeypatch.setattr(autostart, "run", lambda *a: {"done": True, "attempts": []})
    assert main(["pilot-auto", "11506", "G", "09:48", str(tmp_path)]) == 3
