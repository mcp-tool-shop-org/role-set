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
           "PIN_OPEN_LOCK": str(tmp / "open.lock"), "PIN_GATE_MIN_DISK_GB": "0",
           "PIN_WATCH_S": "1", "PIN_STALL_FIRST_S": "6", "PIN_STALL_S": "8", "PIN_WALL_S": "120", "PIN_SESSION": "fake",
           "PIN_CAP_UNITS": "99", "PIN_CAP_MARGIN": "0.01", "PIN_DOWNLOADS": "pin-result.json", "PIN_SKIP_BINARY": "1",
           "PIN_JOB_SCRIPT": "mock_job.py", "PIN_MOCK_S": "8", "PIN_MOCK_EVERY_S": "1", "PIN_MOCK_SILENT": "1",
           # The fake card is an A100 at 5.30/hr: a margin of 0.01 is ~6.8 s, so the grace and the fetch allowance fit it.
           "PIN_DEADLINE_GRACE_S": "2", "PIN_FETCH_ALLOW_S": "2", **(extra or {})}
    out = tmp / "out"
    env.setdefault("PIN_OPEN_LOCK", str(tmp / "open.lock"))  # each case its own lock: no case waits on another's
    stdout, stderr = _run_window(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(binfile), str(out)], env, 180)
    job = (out / "job.log").read_text(encoding="utf-8") if (out / "job.log").exists() else ""
    return stdout + stderr, job, out


def _run_window(cmd: list, env: dict, timeout: float) -> tuple[str, str]:
    """The window under test in its own session. On timeout the whole process group goes, so a watcher subshell left
    holding the capture pipe can't hang the read (CI run on 96da7d6 hung 11 minutes at 97%)."""

    import signal

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        return proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        raise AssertionError(f"the window under test ran past {timeout}s:\n{out[-3000:]}\n{err[-2000:]}") from None


def test_a_lost_first_exec_is_retried_once_and_the_job_runs():
    rig, job, out = _window("lostfirst")
    assert rig.count("EXEC NEVER STARTED") == 1, rig
    assert "STALL" not in rig and job.count("PIN_RESULT") == 1, (rig, job)
    assert (out / "pin-result.json").exists()
    assert (out.parent / "st" / "stopped_under_lock").exists()


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
    assert (out.parent / "st" / "stopped_under_lock").exists()


def test_a_neighbour_closing_between_rate_reads_falls_back_to_gpu_rate():
    rig, job, out = _window("neighbourclose", extra={"PIN_MOCK_S": "2"})
    assert "account rate 15.90/hr" in rig, rig
    assert "own rate 5.3/hr" in rig, rig
    assert (out / "pin-result.json").exists()
    assert (out.parent / "st" / "stopped_under_lock").exists()


def test_a_window_whose_pin_check_fails_stops_before_any_download():
    import json

    rig, job, out = _window("normal", extra={"PIN_PILOT_WRITERS": "granite-4.2-30b", "PIN_PILOT_QUANT": "granite-4.2-30b=q6k"})
    res = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is False and res["steps"] == 0, res
    assert "pins don't agree, nothing downloaded" in res["error"], res
    assert "mock step 1" not in job, job


def test_a_resume_tar_upload_reaches_resume_from():
    import json
    import tarfile

    tmp = Path(tempfile.mkdtemp(prefix="fakewin-resume-in-"))
    w0 = tmp / "pilot-w0"
    w0.mkdir()
    (w0 / "granite-4.2-30b.jsonl").write_text('{"type": "header"}\n', encoding="utf-8")
    (w0 / "granite-4.2-30b.jsonl.sha256").write_text("x  granite-4.2-30b.jsonl\n", encoding="utf-8")
    tar = tmp / "resume.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(w0, arcname="pilot-w0")
    rig, job, out = _window("normal", extra={"PIN_UPLOADS": str(tar), "PIN_RESUME_WAVE": "0",
                                             "PIN_RESUME_DIR": "/content/resume/pilot-w0", "PIN_MOCK_S": "2"})
    assert "uploaded resume.tar" in rig, rig
    res = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is True, res
    assert res["resumed_from"] == {"start_wave": 1, "arms": ["granite-4.2-30b"]}, res


def test_smoke_window_refuses_pin_stub_on_any_gpu_runtime():
    tmp = Path(tempfile.mkdtemp(prefix="fakewin-stubgpu-"))
    binfile = tmp / "bin.dat"
    binfile.write_bytes(b"x" * 100)
    for gpu in ("A100", "L4"):
        out = tmp / f"out-{gpu}"
        env = {**os.environ, "PATH": f"{FAKE}:{os.environ['PATH']}", "FAKE_ST": str(tmp / f"st-{gpu}"),
               "FAKE_MODE": "normal", "PIN_STUB": "1", "PIN_GPU": gpu}
        r = subprocess.run(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(binfile), str(out)],
                           env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 2, (gpu, r.returncode, r.stdout, r.stderr)
        assert "REFUSED: PIN_STUB=1 may only run on PIN_GPU=CPU" in r.stderr, r.stderr
        assert not (tmp / f"st-{gpu}" / "open").exists()


def test_a_late_first_exec_that_takes_the_lock_first_runs_the_only_job():
    # Argon's race: exec #1 arrives after the IDLE read and before the retry's job; the retry must wait, not stop the VM.
    rig, job, out = _window("latefirstwins", extra={"FAKE_LATE_S": "1.5", "FAKE_SECOND_DELAY_S": "2"})
    assert rig.count("EXEC NEVER STARTED") == 1, rig
    assert job.count("mock step 1\n") == 1 and job.count("PIN_RESULT") == 1, job
    assert "JOB ALREADY RUNNING: this exec waits" in job, job
    assert (out / "pin-result.json").exists() and "STALL" not in rig, rig


# ---- the VM health gate (Mike and R&D, 2026-10-18): before any upload, a bad VM is stopped and a fresh one asked for

def _gate_rows(out: Path) -> list[dict]:
    import json

    p = out / "gate.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def test_a_healthy_vm_passes_the_gate_once_and_the_job_runs():
    rig, job, out = _window("normal", extra={"FAKE_GATE": "good"})
    assert "VM gate passed (try 1)" in rig and job.count("PIN_RESULT") == 1, rig
    rows = _gate_rows(out)
    assert [r["try"] for r in rows] == [1] and rows[0]["gate"]["ok"] is True, rows


def test_a_bad_vm_and_a_dead_one_are_stopped_and_the_third_runs_the_job():
    rig, job, out = _window("normal", extra={"FAKE_GATE": "bad,dead,good", "PIN_GATE_TIMEOUT_S": "3"})
    assert "VM gate rejected try 1" in rig and "card in use" in rig, rig
    assert "VM gate rejected try 2" in rig and "died or timed out" in rig, rig
    assert "VM gate passed (try 3)" in rig and job.count("PIN_RESULT") == 1, rig
    assert [r["gate"]["ok"] for r in _gate_rows(out)] == [False, False, True]
    assert rig.count("Session terminated") >= 3  # each rejected VM stopped, then the window's own stop


def test_no_healthy_vm_in_three_tries_ends_the_window_before_any_upload_or_job():
    rig, job, out = _window("normal", extra={"FAKE_GATE": "bad,bad,bad", "PIN_UPLOADS": str(Path(__file__))})
    assert "VM GATE: no healthy A100 in 3 tries" in rig, rig
    st = out.parent / "st"
    assert not (st / "execs").exists() and not job, rig  # no job exec ever ran
    assert not (st / "vm" / "content" / Path(__file__).name).exists()  # and nothing was uploaded
    assert not (st / "open").exists()  # the last VM was stopped too
    assert len(_gate_rows(out)) == 3


def test_the_gate_gets_its_settings_and_a_cpu_window_skips_the_card_check():
    rig, job, out = _window("normal", extra={"PIN_GPU": "CPU", "PIN_GATE_MIN_DISK_GB": "224", "PIN_GATE_TRIES": "2"})
    env = (out.parent / "st" / "gate-env-1").read_text(encoding="utf-8").split()
    assert "PIN_GATE_GPU=0" in env and "PIN_GATE_MIN_DISK_GB=224" in env, env
    assert not any(e.startswith(("PIN_GATE_TRIES=", "PIN_GATE_TIMEOUT_S=")) for e in env), env  # the rig's own settings
    assert _gate_rows(out)[0]["gpu"] == "CPU"


def test_pin_gate_0_skips_the_gate():
    rig, job, out = _window("normal", extra={"PIN_GATE": "0"})
    assert not _gate_rows(out) and "VM gate" not in rig and job.count("PIN_RESULT") == 1, rig


# ---- the private GGUF repo's token (R&D, 2026-10-18): a file, read once on the VM, in nothing the window keeps

SENTINEL = "hf_" + "SENTINELneverPERSIST9f3c2a71abcdEFGHijk"  # a real token's shape, split so the source never holds it
TOKEN_SHAPE = rb"hf_[A-Za-z0-9]{30,}"  # the identity scan's pattern too (R&D, from Argon)


def test_the_token_reaches_the_job_as_a_file_and_appears_in_nothing_the_window_keeps():
    rig, job, out = _window("normal", extra={"PIN_ROLE": "pilot", "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_READ",
                                             "HF_TOKEN_GGUF_READ": SENTINEL})
    assert f"TOKEN FILE READ ({len(SENTINEL)} chars) and deleted" in job, job
    assert "uploaded the token file" in rig and job.count("PIN_RESULT") == 1, rig
    st = out.parent / "st"
    assert not (st / "vm" / "dev" / "shm" / "hf.token").exists()  # deleted on the VM by the job
    # Every file the window keeps or sends back: job.log, pin-result.json, gate.jsonl, smoke-window.log, and the fake
    # CLI's record of each exec's command line and env (job-<n>.sh, gate-env-<n>).
    import re

    leaks = [str(p) for p in out.parent.rglob("*") if p.is_file()
             and (SENTINEL.encode() in p.read_bytes() or re.search(TOKEN_SHAPE, p.read_bytes()))]
    assert not leaks, leaks
    assert SENTINEL not in rig


def test_a_window_naming_an_empty_token_variable_stops_before_the_job():
    rig, job, out = _window("normal", extra={"PIN_ROLE": "pilot", "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_READ",
                                             "HF_TOKEN_GGUF_READ": ""})
    assert "names HF_TOKEN_GGUF_READ, which is empty on the rig" in rig and not job, rig
    assert not (out.parent / "st" / "news").exists()  # before any VM was asked for


def test_each_role_sends_only_its_own_scoped_token_and_never_hf_token():
    for role, var in (("pilot", "HF_TOKEN_GGUF_WRITE"), ("prebuild", "HF_TOKEN_GGUF_READ"), ("pilot", "HF_TOKEN"),
                      ("smoke", "HF_TOKEN_GGUF_READ")):
        rig, job, out = _window("normal", extra={"PIN_ROLE": role, "PIN_HF_TOKEN_FROM": var, var: SENTINEL})
        assert f"may not send {var}" in rig and not job, (role, var, rig)
        assert not (out.parent / "st" / "news").exists()  # refused before any VM was asked for


def test_the_token_goes_only_to_the_vm_that_passed_the_gate():
    rig, job, out = _window("normal", extra={"FAKE_GATE": "bad,good", "PIN_ROLE": "pilot",
                                             "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_READ", "HF_TOKEN_GGUF_READ": SENTINEL})
    uploads = (out.parent / "st" / "uploads.log").read_text(encoding="utf-8").splitlines()
    assert [u for u in uploads if u.endswith("/dev/shm/hf.token")] == ["vm2 /dev/shm/hf.token"], uploads
    assert "VM gate rejected try 1" in rig and "VM gate passed (try 2)" in rig, rig
    assert not [str(p) for p in out.parent.rglob("*") if p.is_file() and SENTINEL.encode() in p.read_bytes()]


def test_a_gpu_window_with_the_gate_on_must_name_its_disk_floor():
    rig, job, out = _window("normal", extra={"PIN_GATE_MIN_DISK_GB": ""})
    assert "needs PIN_GATE_MIN_DISK_GB" in rig and not job and not (out.parent / "st" / "news").exists(), rig


def test_pin_high_mem_asks_for_the_high_memory_shape_and_stays_off_the_vm():
    rig, job, out = _window("normal", extra={"PIN_GPU": "CPU", "PIN_HIGH_MEM": "1"})
    st = out.parent / "st"
    assert "--high-mem" in (st / "new-args.log").read_text(encoding="utf-8") and job.count("PIN_RESULT") == 1, rig
    assert not any("PIN_HIGH_MEM" in p.read_text(encoding="utf-8") for p in st.glob("job-*.sh"))
    rig, job, out = _window("normal", extra={"PIN_GPU": "CPU"})
    assert "--high-mem" not in (out.parent / "st" / "new-args.log").read_text(encoding="utf-8")


# ---- the deadline, a dead kernel, and the outcome draft (R&D, 2026-10-18, after Argon's golden-path and W-A replies) ----
# The fake account rate is 5.30/hr: a cap of 0.03 with 0.0256 kept back puts the deadline ~3 s after the session opens,
# and the cap's ceiling ~20 s after it.
DEADLINE = {"PIN_CAP_UNITS": "0.03", "PIN_CAP_MARGIN": "0.0256", "PIN_MOCK_S": "30"}


def test_a_job_that_stops_at_its_deadline_returns_its_results_before_the_cap():
    import json

    rig, job, out = _window("normal", extra={**DEADLINE, "PIN_MOCK_DEADLINE": "1"})
    # The job's own result ends it: the watcher's PIN_RESULT read, or the exec returning first.
    assert "deadline " in rig and ("JOB DONE" in rig or "end=exec" in rig), rig
    assert "CAP REACHED" not in rig and "CAP CEILING" not in rig, rig
    r = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert r["ok"] is False and "cut_at_deadline" in r and "DeadlineReached" in r["error"], r
    assert any(e.startswith("PIN_DEADLINE=") for e in (out.parent / "st" / "job-1.sh").read_text(encoding="utf-8")
               .replace("export ", "").split()), "PIN_DEADLINE never reached the job"


def test_a_job_that_ignores_its_deadline_is_stopped_after_the_grace():
    import re
    from datetime import datetime

    rig, job, out = _window("normal", extra={**DEADLINE, "PIN_DEADLINE_GRACE_S": "3"})  # 3 + 2 fit 17 s
    assert "CAP REACHED" in rig and "CAP CEILING" not in rig and "end=cap" in rig, rig
    # The kill waited out the grace. (The DEADLINE note is printed only if a watcher check falls inside the grace; under
    # a loaded suite the first check can come after it.)
    t = lambda pat: datetime.strptime(re.search(pat, rig).group(1), "%H:%M:%S")  # noqa: E731
    assert (t(r"CAP REACHED .* (\d\d:\d\d:\d\d)") - t(r"deadline (\d\d:\d\d:\d\d)")).total_seconds() >= 3, rig
    assert job.count("PIN_RESULT") == 0, job
    assert "PIN_DEADLINE_GRACE_S" not in (out.parent / "st" / "job-1.sh").read_text(encoding="utf-8")


def test_a_kernel_that_dies_without_a_result_ends_the_window_before_the_stall():
    import json

    rig, job, out = _window("kerneldies", extra={"PIN_MOCK_S": "30", "PIN_MOCK_DIE_AFTER_S": "3", "PIN_STALL_S": "30"})
    assert "JOB DIED (fake): kernel IDLE at two checks" in rig and "STALL" not in rig, rig
    assert "mock kernel killed" in job and "PIN_RESULT" not in job, job
    assert (out / "died.txt").exists()
    assert json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))["end"] == "died"


def test_one_idle_read_with_the_job_still_running_does_not_end_the_window():
    rig, job, out = _window("normal", extra={"PIN_MOCK_S": "4"})
    assert "JOB DIED" not in rig and job.count("PIN_RESULT") == 1, rig


def test_every_window_writes_an_outcome_draft_with_its_date_and_commit():
    import json
    import time

    rig, job, out = _window("normal", extra={"PIN_ROLESET_COMMIT": "c" * 40, "PIN_MOCK_S": "2"})
    d = json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert d["date"] == time.strftime("%Y-%m-%d", time.gmtime()) and d["commit"] == "c" * 40, d
    assert d["window"] == "out" and d["basis"] == "rate_x_runtime" and d["end"] in ("done", "exec"), d
    assert d["outcome"] is None and d["why"] is None  # the class and its why stay a person's call


def test_a_margin_that_cant_hold_the_grace_and_the_fetch_refuses_before_any_vm():
    rig, job, out = _window("normal", extra={"PIN_CAP_MARGIN": "0.3", "PIN_DEADLINE_GRACE_S": "180", "PIN_FETCH_ALLOW_S": "90"})
    assert "REFUSED: grace 180s + fetch 90s don't fit the margin: 0.3 units at 5.3/hr is 203s" in rig, rig
    assert not (out.parent / "st" / "news").exists() and not job  # no VM asked for
    rig, job, out = _window("normal", extra={"PIN_CAP_MARGIN": "0.45", "PIN_DEADLINE_GRACE_S": "180", "PIN_FETCH_ALLOW_S": "90",
                                             "PIN_MOCK_S": "2"})
    assert "REFUSED" not in rig and job.count("PIN_RESULT") == 1, rig  # 0.45 at 5.3/hr is 305 s


def test_the_stub_call_knob_never_runs_on_a_gpu_window():
    rig, job, out = _window("normal", extra={"PIN_STUB_CALL_S": "30"})  # the fake's default card: A100
    assert "REFUSED: PIN_STUB_CALL_S is a stub rung's setting; never on A100" in rig, rig
    assert not (out.parent / "st" / "news").exists() and not job

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
