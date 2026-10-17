"""The Colab window script's own rules that run without a VM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import pin_window as pw  # noqa: E402


def _w0(dropped: int) -> dict:
    return {"exit": 0, "result": {"dropped": dropped, "halted": None, "lesson_halts": {}}}


def test_the_funnel_scales_to_the_slot_count():
    # 30% of the slots: 6 of the pilot's 20, 3 of the screen's 8 (R&D, 2026-10-17).
    assert pw.funnel_stop(_w0(5), 20) is None
    assert "threshold 6" in pw.funnel_stop(_w0(6), 20)
    assert pw.funnel_stop(_w0(2), 8) is None
    assert "3 of 8 slots dropped at w0 (threshold 3)" == pw.funnel_stop(_w0(3), 8)
    assert pw.funnel_stop(_w0(0), 1) is None and pw.funnel_stop(_w0(1), 1)


def test_the_funnel_still_stops_on_a_halt_or_a_dead_wave():
    assert pw.funnel_stop({"exit": 0, "result": {"dropped": 0, "halted": "systemic"}}, 8) == "halted at w0: systemic"
    assert pw.funnel_stop({"exit": 1}, 8) == "w0 exited 1 with no result"


# ---- lazy setup and a critic per arm (R&D, 2026-10-17, from Argon's budget-150 reply)

import json  # noqa: E402

import pytest  # noqa: E402


class _Reclaimed(BaseException):
    """Colab taking the VM back: nothing in the window catches it."""


def _fake_window(monkeypatch, tmp_path, *, die_after_write_of=None):
    events = []

    def fetch_model(spec, built, keep_snapshot=False):
        events.append(("fetch", spec["served_as"]))
        return {"served_as": spec["served_as"], "gguf_sha256": spec.get("expect_sha256") or "x" * 64, "path": "/dev/null"}

    class Served:
        def __init__(self, built, m, pinned):
            self.m, self.starts = m, []

        def __enter__(self):
            events.append(("serve", self.m["served_as"]))
            return self

        def __exit__(self, *exc):
            events.append(("stop", self.m["served_as"]))

    def write_wave(rs, common, wave, out, *, writer=None, critic=None, **kw):
        who = (writer or critic)["served_as"]
        events.append((wave, out.parent.name, who))
        if wave == "write" and out.parent.name == die_after_write_of:
            raise _Reclaimed()
        return {"exit": 0, "result": {"dropped": 0, "states": {"awaiting_critique": 0, "needs_rewrite": 0}}}

    monkeypatch.setattr(pw, "fetch_model", fetch_model)
    monkeypatch.setattr(pw, "Served", Served)
    monkeypatch.setattr(pw, "choose_quant", lambda built, spec, m, policy, result: (m, {"policy": policy}))
    monkeypatch.setattr(pw, "j1_card", lambda m, cards, out: (tmp_path / "c.json", m["served_as"], {}))
    monkeypatch.setattr(pw, "write_wave", write_wave)
    monkeypatch.setattr(pw, "run_dir", lambda out: out)
    monkeypatch.setattr(pw, "checkpoint", lambda *a: None)
    monkeypatch.setattr(pw, "PILOT_WAVES", 2)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"slots": [{"id": f"s{i}"} for i in range(8)]}), encoding="utf-8")
    pool = pw.Pool({}, {}, {}, tmp_path / "cards.json", tmp_path / "out")
    return events, plan, pool


def test_each_writer_is_set_up_just_before_its_own_write_and_the_critic_last(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    A, B, M = "qwen3.6-35b-a3b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    fetches = [e for e in events if e[0] == "fetch"]
    assert fetches == [("fetch", A), ("fetch", B), ("fetch", M)]
    order = [e for e in events if e[0] in ("fetch", "write", "critique")]
    assert order == [("fetch", A), ("write", A, A), ("fetch", B), ("write", B, B),
                     ("fetch", M), ("critique", A, M), ("critique", B, M)]
    assert events.count(("serve", M)) == 1  # one critic start reads both arms


def test_a_reclaim_after_the_first_arms_write_costs_only_that_arms_setup(monkeypatch, tmp_path):
    A, B, M = "qwen3.6-35b-a3b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    events, plan, pool = _fake_window(monkeypatch, tmp_path, die_after_write_of=A)
    with pytest.raises(_Reclaimed):
        pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    assert [e for e in events if e[0] == "fetch"] == [("fetch", A)]  # B and the critic were never set up


def test_two_arms_critique_each_other_with_each_model_fetched_once(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    L, M = "nemotron-3.5-lightning-30b-a3b", "muse-glimmer-30b"
    pw.pilot_batch({}, [], plan, tmp_path / "pilot", [L, M], {L: M, M: L}, pool)
    assert [e for e in events if e[0] == "fetch"] == [("fetch", L), ("fetch", M)]
    assert [e for e in events if e[0] == "critique"] == [("critique", L, M), ("critique", M, L)]


def test_critics_for_takes_the_default_and_the_per_arm_settings(monkeypatch):
    monkeypatch.setenv("PIN_PILOT_CRITIC", "muse-glimmer-30b")
    monkeypatch.setenv("PIN_PILOT_CRITIC_FOR", "muse-glimmer-30b=nemotron-3.5-lightning-30b-a3b")
    got = pw.critics_for(["nemotron-3.5-lightning-30b-a3b", "muse-glimmer-30b"])
    assert got == {"nemotron-3.5-lightning-30b-a3b": "muse-glimmer-30b", "muse-glimmer-30b": "nemotron-3.5-lightning-30b-a3b"}
    monkeypatch.setenv("PIN_PILOT_CRITIC_FOR", "muse-glimmer-30b=muse-glimmer-30b")
    with pytest.raises(RuntimeError, match="critique itself"):
        pw.critics_for(["muse-glimmer-30b"])
    monkeypatch.delenv("PIN_PILOT_CRITIC")
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR")
    with pytest.raises(RuntimeError, match="no critic"):
        pw.critics_for(["granite-4.2-30b"])
