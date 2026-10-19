"""The VM health gate's own checks (scripts/colab/vm_gate.py; Mike and R&D, 2026-10-18), without a VM."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))
import vm_gate as g  # noqa: E402

GOOD = {"gpu_name": "NVIDIA A100-SXM4-40GB", "vram_mib": 40960.0, "vram_used_mib": 0.0, "gpu_processes": 0,
        "disk_free_gb": 204.2, "ram_available_gb": 83.4, "hf_mbps": 250.0}


@pytest.fixture
def lim(monkeypatch):
    for k in list(g.os.environ):
        if k.startswith("PIN_GATE_"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("PIN_GATE_MIN_DISK_GB", "184")
    return g.limits()


def test_night_ones_a100_passes_and_the_hf_floor_is_half_the_measured_median(lim):
    assert g.verdict(GOOD, lim) == []
    assert lim["min_hf_mbps"] == 129.0 and lim["min_vram_mib"] == 39000 and lim["max_used_mib"] == 500


@pytest.mark.parametrize("change, why", [
    ({"gpu_name": "NVIDIA L4"}, "is not A100"),
    ({"vram_mib": 23034.0}, "VRAM"),
    ({"vram_used_mib": 1200.0}, "card in use"),
    ({"gpu_processes": 1}, "card in use"),
    ({"disk_free_gb": 180.0}, "disk 180.0 GB free < 184"),
    ({"ram_available_gb": 40.0}, "RAM"),
    ({"hf_mbps": 90.0}, "Hugging Face 90.0 MB/s"),
])
def test_each_bad_reading_rejects_the_vm_with_its_reason(lim, change, why):
    got = g.verdict({**GOOD, **change}, lim)
    assert len(got) == 1 and why in got[0], got


def test_a_cpu_window_checks_no_card(lim, monkeypatch):
    monkeypatch.setenv("PIN_GATE_GPU", "0")
    cpu = {k: v for k, v in GOOD.items() if not k.startswith(("gpu", "vram"))}
    assert g.verdict(cpu, g.limits()) == []


def _run_main(monkeypatch, capsys, reads):
    monkeypatch.setattr(g, "gpu_readings", lambda: {k: GOOD[k] for k in ("gpu_name", "vram_mib", "vram_used_mib", "gpu_processes")})
    monkeypatch.setattr(g, "host_readings", lambda: {"disk_free_gb": 204.2, "ram_available_gb": 83.4})
    it = iter(reads)
    monkeypatch.setattr(g, "hf_read", lambda url, n: next(it))
    g.main()
    line = capsys.readouterr().out.strip()
    assert line.startswith("VM_GATE ")
    return json.loads(line.removeprefix("VM_GATE "))


def test_one_short_read_takes_a_second_before_rejecting(lim, monkeypatch, capsys):
    r = _run_main(monkeypatch, capsys, [80.0, 240.0])
    assert r["ok"] is True and r["hf_reads_mbps"] == [80.0, 240.0]
    r = _run_main(monkeypatch, capsys, [80.0, 90.0])
    assert r["ok"] is False and "Hugging Face 90.0" in r["why"][0]
    r = _run_main(monkeypatch, capsys, [250.0])
    assert r["hf_reads_mbps"] == [250.0]  # a good first read decides


def test_a_vm_the_gate_cannot_read_is_rejected_with_the_reason(lim, monkeypatch, capsys):
    def boom():
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(g, "gpu_readings", boom)
    g.main()
    r = json.loads(capsys.readouterr().out.strip().removeprefix("VM_GATE "))
    assert r["ok"] is False and "could not read the VM" in r["why"][0]


# The floor scaled to the window (R&D, 2026-10-19): bytes / (bytes / median + budget), budget max(300 s, 15% of the
# window's projected seconds), never under 20 MB/s; the override wins; with no window bytes, the old half-median.
@pytest.mark.parametrize("window_bytes, want", [(1e9, 20.0), (32e9, 75.5), (300e9, 205.1)])
def test_the_hf_floor_scales_with_the_bytes_the_window_downloads(lim, monkeypatch, window_bytes, want):
    monkeypatch.setenv("PIN_GATE_HF_BYTES_WINDOW", str(int(window_bytes)))
    got = g.limits()
    assert got["min_hf_mbps"] == want and got["hf_floor_from"] == "computed" and got["hf_budget_s"] == 300.0


def test_a_small_window_keeps_the_broken_pipe_floor(lim, monkeypatch):
    monkeypatch.setenv("PIN_GATE_HF_BYTES_WINDOW", str(10**6))
    assert g.limits()["min_hf_mbps"] == g.HF_HARD_FLOOR_MBPS == 20.0
    assert g.verdict({**GOOD, "hf_mbps": 19.9}, g.limits()) and not g.verdict({**GOOD, "hf_mbps": 43.2}, g.limits())


def test_without_window_bytes_the_floor_stays_half_the_median(lim, monkeypatch):
    assert lim["min_hf_mbps"] == 129.0 and lim["hf_floor_from"] == "default"
    monkeypatch.setenv("PIN_GATE_PROJECTED_S", "7200")  # a projection alone has no bytes to scale
    assert g.limits()["min_hf_mbps"] == 129.0 and g.limits()["hf_floor_from"] == "default"


def test_the_explicit_floor_wins_over_the_computed_one(lim, monkeypatch):
    monkeypatch.setenv("PIN_GATE_HF_BYTES_WINDOW", str(int(32e9)))
    monkeypatch.setenv("PIN_GATE_MIN_HF_MBPS", "60")
    got = g.limits()
    assert got["min_hf_mbps"] == 60.0 and got["hf_floor_from"] == "override"


def test_a_long_window_gets_a_bigger_budget_and_a_lower_floor(lim, monkeypatch):
    monkeypatch.setenv("PIN_GATE_HF_BYTES_WINDOW", str(int(300e9)))
    monkeypatch.setenv("PIN_GATE_PROJECTED_S", str(4 * 3600))  # 15% of 4 h is 2,160 s, over the 300 s minimum
    got = g.limits()
    assert got["hf_budget_s"] == 2160.0 and got["min_hf_mbps"] == round(300e9 / (300e9 / 258e6 + 2160) / 1e6, 1) == 90.3


def test_the_gate_line_records_the_floor_and_its_source(lim, monkeypatch, capsys):
    monkeypatch.setenv("PIN_GATE_HF_BYTES_WINDOW", str(int(32e9)))
    r = _run_main(monkeypatch, capsys, [80.0])
    assert r["ok"] is True and r["limits"]["min_hf_mbps"] == 75.5 and r["limits"]["hf_floor_from"] == "computed"
