"""A stand-in job for testing the window script on a cheap runtime (R&D, 2026-10-17, after mistakes #31).

It does what pin_window.py does for the rig's watcher, with no model: it appends a line to /content/pin/job.log every
PIN_MOCK_EVERY_S seconds for PIN_MOCK_S seconds, then writes pin-result.json and its PIN_RESULT line. With
PIN_MOCK_SILENT=1 it prints nothing to the exec's stream, so the rig can only follow the job through job.log. With
PIN_MOCK_HANG_S it then waits that long before returning, as an exec that never comes back: the rig must end the window
from the PIN_RESULT line in job.log. With PIN_MOCK_DEADLINE=1 it stops at PIN_DEADLINE as pin_window does, writing a
result that says it was cut there. With PIN_MOCK_DIE_AFTER_S it dies that many seconds in, with no result, as a kernel
killed for memory would.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("PIN_WORK") or "/content/pin")  # PIN_WORK: the fake CLI's VM directory, in tests
CONTENT = Path(os.environ.get("PIN_CONTENT") or str(WORK.parent))


def take_token_file() -> None:
    """As pin_window.take_token: the private repo's token file, read once and deleted; only its length is logged."""

    p = CONTENT.parent / "dev" / "shm" / "hf.token"  # the fake CLI's VM root is PIN_CONTENT's parent
    if p.exists():
        n = len(p.read_text(encoding="utf-8").strip())
        p.unlink()
        WORK.mkdir(parents=True, exist_ok=True)
        with (WORK / "job.log").open("a", encoding="utf-8") as f:
            f.write(f"TOKEN FILE READ ({n} chars) and deleted\n")
# The same window-scoped lock as pin_window.take_job_lock (copied: the mock runs alone on the VM).
def take_job_lock(work: Path):
    """One job per VM, per window (R&D, 2026-10-17; fixed after Browser Argon's v2 re-review).

    The window re-runs an exec that never reached the kernel, so a late first exec can still arrive. The job takes an
    exclusive flock on <work>/job.lock for its own life and releases it when it ends (main's finally). A second job:
    - finds this window's result already written (pin-result.json carrying this window's PIN_JOB_TOKEN): it stops;
    - finds the lock held: it waits for the first job to end, then stops if that job wrote this window's result, or runs
      if it didn't (the first died without one).
    A pin-result.json from another run, with another token, never makes a real job skip itself. The lock used to live in a
    module global held for the kernel's life; `colab exec -f` re-runs the whole file in the kernel's globals, which
    rebound it to None and released the lock (Argon). Returns the open lock file, or None when this job must stop."""

    import fcntl  # noqa: PLC0415 - Linux, where the window runs

    work.mkdir(parents=True, exist_ok=True)

    def done_here() -> bool:
        token = os.environ.get("PIN_JOB_TOKEN")
        try:
            return bool(token) and json.loads((work / "pin-result.json").read_text(encoding="utf-8")).get("job_token") == token
        except (OSError, ValueError):
            return False

    def note(text: str) -> None:
        with (work / "job.log").open("a", encoding="utf-8") as f:
            f.write(f"{text} {time.strftime('%H:%M:%S')}\n")

    if done_here():
        note("JOB ALREADY DONE: this window's result is written; this exec stops")
        return None
    fh = (work / "job.lock").open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        note("JOB ALREADY RUNNING: this exec waits for it")
        fcntl.flock(fh, fcntl.LOCK_EX)  # until the first job ends
        if done_here():
            fh.close()
            note("JOB ALREADY DONE: the first job finished this window; this exec stops")
            return None
        note("JOB ALREADY RUNNING ended with no result for this window: this exec runs the job")
    return fh


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    lock = take_job_lock(WORK)
    if lock is None:
        return 0
    take_token_file()
    try:
        return _run()
    finally:
        lock.close()


def _run() -> int:
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

    extra_res: dict = {}
    try:
        if os.environ.get("PIN_PILOT_WRITERS") or os.environ.get("PIN_RESUME_WAVE") or (CONTENT / "resume.tar").exists():
            here = Path(__file__).resolve().parent
            if str(here) not in sys.path:
                sys.path.insert(0, str(here))
            import pin_window as pw  # noqa: PLC0415

            pw.CONTENT, pw.WORK, pw.RESUME_TAR = CONTENT, WORK, CONTENT / "resume.tar"
            if os.environ.get("PIN_PILOT_WRITERS"):
                writers = [x.strip() for x in os.environ["PIN_PILOT_WRITERS"].split(",") if x.strip()]
                policies = dict(x.split("=", 1) for x in os.environ.get("PIN_PILOT_QUANT", "").split(",") if "=" in x)
                listed = json.loads((here.parents[1] / "writers.json").read_text(encoding="utf-8"))
                pw.check_pins(writers, listed, policies)
            if (CONTENT / "resume.tar").exists() or os.environ.get("PIN_RESUME_WAVE"):
                unpacked = pw.unpack_resume()
                start_wave, resumed = pw.resume_from()
                extra_res["resume_unpacked"] = str(unpacked or "")
                extra_res["resumed_from"] = {"start_wave": start_wave, "arms": sorted(resumed)}
        deadline = os.environ.get("PIN_DEADLINE") if os.environ.get("PIN_MOCK_DEADLINE") == "1" else None
        die_after = float(os.environ.get("PIN_MOCK_DIE_AFTER_S") or 0)
        while time.time() - started < total:
            if die_after and time.time() - started >= die_after:
                line(f"[{time.strftime('%H:%M:%S')}] mock kernel killed")
                os._exit(137)  # no finally, no result: the kernel is gone
            if deadline and time.time() >= float(deadline):
                extra_res["cut_at_deadline"] = {"step": n}
                raise RuntimeError("DeadlineReached: the mock wave was cut at the window's deadline")
            n += 1
            line(f"[{time.strftime('%H:%M:%S')}] mock step {n}")
            time.sleep(every)
        result = {"ok": True, "mock": True, "job_token": os.environ.get("PIN_JOB_TOKEN"), "steps": n, "seconds": round(time.time() - started), "silent": silent, **extra_res}
    except Exception as e:  # noqa: BLE001 - reported in pin-result.json before any download
        result = {"ok": False, "mock": True, "job_token": os.environ.get("PIN_JOB_TOKEN"), "steps": n, "seconds": round(time.time() - started), "silent": silent,
                  "error": f"{type(e).__name__}: {e}", **extra_res}
    (WORK / "pin-result.json").write_text(json.dumps(result, indent=1))
    line("PIN_RESULT " + json.dumps(result))
    time.sleep(float(os.environ.get("PIN_MOCK_HANG_S") or 0))
    return 0 if result["ok"] else 1


if __name__ == "__main__":  # pragma: no cover - runs on the VM
    main()
