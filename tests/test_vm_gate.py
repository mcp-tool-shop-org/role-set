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
