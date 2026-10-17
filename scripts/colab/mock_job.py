"""A stand-in job for testing the window script on a cheap runtime (R&D, 2026-10-17, after mistakes #31).

It does what pin_window.py does for the rig's watcher, with no model: it appends a line to /content/pin/job.log every
PIN_MOCK_EVERY_S seconds for PIN_MOCK_S seconds, then writes pin-result.json and its PIN_RESULT line. With
PIN_MOCK_SILENT=1 it prints nothing to the exec's stream, so the rig can only follow the job through job.log. With
PIN_MOCK_HANG_S it then waits that long before returning, as an exec that never comes back: the rig must end the window
from the PIN_RESULT line in job.log.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

WORK = Path(os.environ.get("PIN_WORK") or "/content/pin")  # PIN_WORK: the fake CLI's VM directory, in tests
_JOB_LOCK = None


def take_job_lock(work: Path):
    """The same one-job-per-VM lock as pin_window.take_job_lock."""

    import fcntl  # noqa: PLC0415

    work.mkdir(parents=True, exist_ok=True)
    fh = (work / "job.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        with (work / "job.log").open("a", encoding="utf-8") as f:
            f.write(f"JOB ALREADY RUNNING {time.strftime('%H:%M:%S')}: this exec stops; the first job carries on\n")
        return None
    return fh


def main() -> int:
    global _JOB_LOCK
    WORK.mkdir(parents=True, exist_ok=True)
    _JOB_LOCK = take_job_lock(WORK)
    if _JOB_LOCK is None:
        return 0
    log = WORK / "job.log"
    silent = os.environ.get("PIN_MOCK_SILENT") == "1"
    every, total = float(os.environ.get("PIN_MOCK_EVERY_S") or 30), float(os.environ.get("PIN_MOCK_S") or 900)
    started = time.time()
    n = 0

    def line(text: str) -> None:
        with log.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        if not silent:
            print(text, flush=True)

    while time.time() - started < total:
        n += 1
        line(f"[{time.strftime('%H:%M:%S')}] mock step {n}")
        time.sleep(every)
    result = {"ok": True, "mock": True, "steps": n, "seconds": round(time.time() - started), "silent": silent}
    (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
    line("PIN_RESULT " + json.dumps(result))
    time.sleep(float(os.environ.get("PIN_MOCK_HANG_S") or 0))
    return 0


if __name__ == "__main__":  # pragma: no cover - runs on the VM
    main()
