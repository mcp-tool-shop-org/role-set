"""scripts/colab/smoke_window.sh against a fake colab CLI (tests/fake_colab/colab), with the real mock_job.py as the
job. Linux only (bash, flock, setsid); on the rig run it in WSL: `python3 tests/test_window_script.py`.

Cases (R&D, 2026-10-17): the first exec lost and the retry runs the job; the first exec late, so two arrive and only
one job runs; an exec that never starts, twice, stalls; and the plain cases the watcher already handled."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import pytest
except ImportError:  # run as a plain script in WSL
    pytest = None

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "fake_colab"
pytestmark = pytest and pytest.mark.skipif(os.name == "nt" or not shutil.which("bash") or not shutil.which("setsid"),
                                reason="needs bash, flock and setsid (Linux)")


def _window(mode: str, *, extra: dict | None = None) -> tuple[str, str, Path]:
    tmp = Path(tempfile.mkdtemp(prefix=f"fakewin-{mode}-"))
    binfile = tmp / "bin.dat"
    binfile.write_bytes(b"x" * 1000)
    env = {**os.environ, "PATH": f"{FAKE}:{os.environ['PATH']}", "FAKE_ST": str(tmp / "st"), "FAKE_MODE": mode,
           "PIN_WATCH_S": "1", "PIN_STALL_FIRST_S": "6", "PIN_STALL_S": "8", "PIN_WALL_S": "120", "PIN_SESSION": "fake",
           "PIN_CAP_UNITS": "99", "PIN_CAP_MARGIN": "0.01", "PIN_DOWNLOADS": "pin-result.json", "PIN_SKIP_BINARY": "1",
           "PIN_JOB_SCRIPT": "mock_job.py", "PIN_MOCK_S": "8", "PIN_MOCK_EVERY_S": "1", "PIN_MOCK_SILENT": "1", **(extra or {})}
    out = tmp / "out"
    r = subprocess.run(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(binfile), str(out)], env=env,
                       capture_output=True, text=True, timeout=180)
    job = (out / "job.log").read_text(encoding="utf-8") if (out / "job.log").exists() else ""
    return r.stdout + r.stderr, job, out


def test_a_lost_first_exec_is_retried_once_and_the_job_runs():
    rig, job, out = _window("lostfirst")
    assert rig.count("EXEC NEVER STARTED") == 1, rig
    assert "STALL" not in rig and job.count("PIN_RESULT") == 1, (rig, job)
    assert (out / "pin-result.json").exists()


def test_a_late_first_exec_and_its_retry_run_one_job():
    rig, job, _ = _window("late", extra={"FAKE_LATE_S": "4"})
    assert rig.count("EXEC NEVER STARTED") == 1, rig
    assert job.count("JOB ALREADY RUNNING") == 1, job  # the late one found the lock taken
    assert job.count("mock step 1\n") == 1 and job.count("PIN_RESULT") == 1, job


def test_an_exec_that_never_starts_twice_stalls():
    rig, job, _ = _window("neverstarts")
    assert rig.count("EXEC NEVER STARTED") == 1, rig  # one retry, no more
    assert "STALL (fake): job.log empty" in rig, rig


def test_a_normal_exec_is_not_retried():
    rig, job, out = _window("normal")
    assert "EXEC NEVER STARTED" not in rig and "STALL" not in rig, rig
    assert job.count("PIN_RESULT") == 1 and (out / "pin-result.json").exists()


if __name__ == "__main__":  # WSL on the rig: no pytest needed
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {str(e)[:1500]}")
    sys.exit(1 if failed else 0)
