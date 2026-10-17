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
