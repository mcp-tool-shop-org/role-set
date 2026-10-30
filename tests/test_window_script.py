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


def _window(mode: str, *, extra: dict | None = None, out_name: str = "out") -> tuple[str, str, Path]:
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
    out = tmp / out_name
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
    for gpu in ("A100", "L4", "G4"):
        out = tmp / f"out-{gpu}"
        env = {**os.environ, "PATH": f"{FAKE}:{os.environ['PATH']}", "FAKE_ST": str(tmp / f"st-{gpu}"),
               "FAKE_MODE": "normal", "PIN_STUB": "1", "PIN_GPU": gpu}
        r = subprocess.run(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(binfile), str(out)],
                           env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 2, (gpu, r.returncode, r.stdout, r.stderr)
        assert "REFUSED: PIN_STUB=1 may only run on PIN_GPU=CPU" in r.stderr, r.stderr
        assert not (tmp / f"st-{gpu}" / "open").exists()
    # Unknown PIN_GPU value is refused before colab new can silently fall back to A100.
    out_bad = tmp / "out-badgpu"
    env_bad = {**os.environ, "PATH": f"{FAKE}:{os.environ['PATH']}", "FAKE_ST": str(tmp / "st-badgpu"),
               "FAKE_MODE": "normal", "PIN_GPU": "RTX_PRO_6000"}
    r_bad = subprocess.run(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(binfile), str(out_bad)],
                           env=env_bad, capture_output=True, text=True, timeout=30)
    assert r_bad.returncode == 2
    assert "REFUSED: PIN_GPU=RTX_PRO_6000 must be A100, H100, G4, L4, T4 or CPU" in r_bad.stderr
    assert not (tmp / "st-badgpu" / "open").exists()


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
                      ("smoke", "HF_TOKEN_GGUF_READ"), ("offrig-bench", "HF_TOKEN_GGUF_READ"),
                      ("offrig-bench", "HF_TOKEN_GGUF_WRITE"), ("offrig-bench", "HF_TOKEN"),
                      ("pilot", "HF_TOKEN_OLLAMA_PINS_READ"), ("critic-eval", "HF_TOKEN_OLLAMA_PINS_READ"),
                      ("prebuild", "HF_TOKEN_OLLAMA_PINS_READ"), ("smoke", "HF_TOKEN_OLLAMA_PINS_READ")):
        rig, job, out = _window("normal", extra={"PIN_ROLE": role, "PIN_HF_TOKEN_FROM": var, var: SENTINEL})
        assert f"may not send {var}" in rig and not job, (role, var, rig)
        assert not (out.parent / "st" / "news").exists()  # refused before any VM was asked for
    rig_ok, job_ok, out_ok = _window(
        "normal",
        extra={
            "PIN_ROLE": "offrig-bench",
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_OLLAMA_PINS_READ",
            "HF_TOKEN_OLLAMA_PINS_READ": SENTINEL,
            "PIN_MOCK_S": "1",
        },
    )
    assert "uploaded the token file" in rig_ok and job_ok.count("PIN_RESULT") == 1, rig_ok
    assert not (out_ok.parent / "st" / "vm" / "dev" / "shm" / "hf.token").exists()
    assert not [str(p) for p in out_ok.parent.rglob("*") if p.is_file() and SENTINEL.encode() in p.read_bytes()]


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


# ---- the deadline and the outcome draft (R&D, 2026-10-18, after Argon's golden-path and W-A replies) ----
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


def test_every_window_writes_an_outcome_draft_with_its_date_and_commit():
    import json
    import time

    rig, job, out = _window("normal", extra={"PIN_ROLESET_COMMIT": "c" * 40, "PIN_MOCK_S": "2"})
    d = json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert d["date"] == time.strftime("%Y-%m-%d", time.gmtime()) and d["commit"] == "c" * 40, d
    assert d["window"] == "out" and d["basis"] == "rate_x_runtime" and d["end"] in ("done", "exec"), d
    assert d["outcome"] is None and d["why"] is None  # the class and its why stay a person's call


def test_a_margin_that_cant_hold_the_grace_and_the_fetch_refuses_before_any_vm():
    # The defaults (an empty value takes them): grace 60 s + fetch 120 s against the margin at the fake A100's 5.3/hr.
    defaults = {"PIN_DEADLINE_GRACE_S": "", "PIN_FETCH_ALLOW_S": ""}
    rig, job, out = _window("normal", extra={**defaults, "PIN_CAP_MARGIN": "0.2"})
    assert "REFUSED: grace 60s + fetch 120s don't fit the margin: 0.2 units at 5.3/hr is 135s" in rig, rig
    assert not (out.parent / "st" / "news").exists() and not job  # no VM asked for
    rig, job, out = _window("normal", extra={**defaults, "PIN_CAP_MARGIN": "0.3", "PIN_MOCK_S": "2"})
    assert "REFUSED" not in rig and job.count("PIN_RESULT") == 1, rig  # the default margin: 0.3 at 5.3/hr is 203 s


def test_card_rates_refusals_and_g4_margin_check():
    tmp = Path(tempfile.mkdtemp(prefix="fakewin-cardrates-"))
    binfile = tmp / "bin.dat"
    binfile.write_bytes(b"x" * 100)
    script = str(ROOT / "scripts/colab/smoke_window.sh")

    # 1. Each known card:high_mem combination has the same measured rate in both CARD_RATE (margin check) and GPU_RATE (neighbourclose fallback)
    for gpu, high_mem, rate_str, bad_margin, bad_budget_s in (
        ("G4", "0", "8.90", "0.3", "121"),
        ("G4", "1", "8.90", "0.3", "121"),
        ("A100", "1", "6.77", "0.3", "159"),
        ("A100", "0", "5.3", "0.2", "135"),
        ("CPU", "1", "0.26", "0.01", "138"),
        ("CPU", "0", "0.08", "0.003", "135"),
    ):
        out_m = tmp / f"out-m-{gpu}-{high_mem}"
        env_m = {
            **os.environ,
            "PATH": f"{FAKE}:{os.environ['PATH']}",
            "FAKE_ST": str(tmp / f"st-m-{gpu}-{high_mem}"),
            "FAKE_MODE": "normal",
            "PIN_OPEN_LOCK": str(tmp / f"lock-m-{gpu}-{high_mem}"),
            "PIN_GATE_MIN_DISK_GB": "0",
            "PIN_GPU": gpu,
            "PIN_HIGH_MEM": high_mem,
            "PIN_CAP_MARGIN": bad_margin,
        }
        r_m = subprocess.run(["bash", script, str(binfile), str(out_m)], env=env_m, capture_output=True, text=True, timeout=30, check=False)
        assert r_m.returncode == 2, (gpu, high_mem, r_m.returncode, r_m.stdout, r_m.stderr)
        assert f"REFUSED: grace 60s + fetch 120s don't fit the margin: {bad_margin} units at {rate_str}/hr is {bad_budget_s}s" in r_m.stderr, (
            gpu,
            high_mem,
            r_m.stderr,
        )
        assert not (tmp / f"st-m-{gpu}-{high_mem}" / "news").exists()

        rig_fb, _job_fb, out_fb = _window(
            "neighbourclose",
            extra={"PIN_GPU": gpu, "PIN_HIGH_MEM": high_mem, "PIN_MOCK_S": "1", "PIN_GATE": "0"},
        )
        assert f"own rate {rate_str}/hr" in rig_fb, (gpu, high_mem, rig_fb)
        assert (out_fb / "pin-result.json").exists()

    # 2. G4 margin check refuses at 0.3 (0.3 / 8.9 h = 121 s < 180 s) above and passes at 0.5 (0.5 / 8.9 h = 202 s >= 180 s)
    rig_g4_pass, job_g4_pass, out_g4_pass = _window(
        "normal",
        extra={
            "PIN_DEADLINE_GRACE_S": "",
            "PIN_FETCH_ALLOW_S": "",
            "PIN_GPU": "G4",
            "PIN_CAP_MARGIN": "0.5",
            "PIN_MOCK_S": "1",
            "PIN_GATE": "0",
        },
    )
    assert "REFUSED" not in rig_g4_pass and job_g4_pass.count("PIN_RESULT") == 1, rig_g4_pass
    assert (out_g4_pass / "pin-result.json").exists()

    # 3. Cards accepted on line 12 with no measured rate (H100, L4, T4) are refused at start before asking for any VM
    for unmeasured in ("H100", "L4", "T4"):
        out_u = tmp / f"out-u-{unmeasured}"
        env_u = {
            **os.environ,
            "PATH": f"{FAKE}:{os.environ['PATH']}",
            "FAKE_ST": str(tmp / f"st-u-{unmeasured}"),
            "FAKE_MODE": "normal",
            "PIN_OPEN_LOCK": str(tmp / f"lock-u-{unmeasured}"),
            "PIN_GATE_MIN_DISK_GB": "0",
            "PIN_GPU": unmeasured,
        }
        r_u = subprocess.run(["bash", script, str(binfile), str(out_u)], env=env_u, capture_output=True, text=True, timeout=30, check=False)
        assert r_u.returncode == 2, (unmeasured, r_u.returncode, r_u.stdout, r_u.stderr)
        assert f"REFUSED: PIN_GPU={unmeasured} (PIN_HIGH_MEM=0) has no measured rate yet; refuse until one is measured" in r_u.stderr, (
            unmeasured,
            r_u.stderr,
        )
        assert not (tmp / f"st-u-{unmeasured}" / "news").exists()


def test_the_stub_call_knob_never_runs_on_a_gpu_window():
    rig, job, out = _window("normal", extra={"PIN_STUB_CALL_S": "30"})  # the fake's default card: A100
    assert "REFUSED: PIN_STUB_CALL_S is a stub rung's setting; never on A100" in rig, rig
    assert not (out.parent / "st" / "news").exists() and not job


def test_smoke_window_checks_pin_server_sha256_and_binary_sha_before_spend():
    import hashlib

    tmp = Path(tempfile.mkdtemp(prefix="fakewin-bincheck-"))
    binfile = tmp / "bin.dat"
    bin_bytes = b"fake-llama-server-binary-bytes"
    binfile.write_bytes(bin_bytes)
    bin_sha = hashlib.sha256(bin_bytes).hexdigest()
    default_sha = "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"
    g4_sha = "39742d8a0354d4577da8dcc9cb3a5b9f03f02593d106eff41297e952d6ebfc95"
    script = str(ROOT / "scripts/colab/smoke_window.sh")

    # 1. Refusal: malformed PIN_SERVER_SHA256 is refused before `colab new` (even when PIN_SKIP_BINARY=1)
    for idx, bad in enumerate(("", "a" * 63, "a" * 65, g4_sha.upper(), "g" + "a" * 63)):
        out_bad = tmp / f"out-bad-{idx}"
        env_bad = {
            **os.environ,
            "PATH": f"{FAKE}:{os.environ['PATH']}",
            "FAKE_ST": str(tmp / f"st-bad-{idx}"),
            "FAKE_MODE": "normal",
            "PIN_OPEN_LOCK": str(tmp / f"lock-bad-{idx}"),
            "PIN_GATE_MIN_DISK_GB": "0",
            "PIN_SKIP_BINARY": "1",
            "PIN_SERVER_SHA256": bad,
        }
        r_bad = subprocess.run(["bash", script, str(binfile), str(out_bad)], env=env_bad, capture_output=True, text=True, timeout=30, check=False)
        assert r_bad.returncode == 2, (bad, r_bad.returncode, r_bad.stdout, r_bad.stderr)
        assert f"REFUSED: PIN_SERVER_SHA256={bad} must be 64 lowercase hex characters" in r_bad.stderr, r_bad.stderr
        assert not (tmp / f"st-bad-{idx}" / "news").exists()

    # 2. Refusal: when PIN_SKIP_BINARY is not 1, binary sha256 mismatch is refused before `colab new` (both default and overridden)
    for tag, env_sha, want_sha in (("default", {}, default_sha), ("override", {"PIN_SERVER_SHA256": g4_sha}, g4_sha)):
        out_mism = tmp / f"out-mism-{tag}"
        env_mism = {
            **os.environ,
            "PATH": f"{FAKE}:{os.environ['PATH']}",
            "FAKE_ST": str(tmp / f"st-mism-{tag}"),
            "FAKE_MODE": "normal",
            "PIN_OPEN_LOCK": str(tmp / f"lock-mism-{tag}"),
            "PIN_GATE_MIN_DISK_GB": "0",
            "PIN_SKIP_BINARY": "0",
            **env_sha,
        }
        r_mism = subprocess.run(["bash", script, str(binfile), str(out_mism)], env=env_mism, capture_output=True, text=True, timeout=30, check=False)
        assert r_mism.returncode == 2, (tag, r_mism.returncode, r_mism.stdout, r_mism.stderr)
        assert f"REFUSED: llama-server {binfile} sha256 {bin_sha} != expected {want_sha}" in r_mism.stderr, r_mism.stderr
        assert not (tmp / f"st-mism-{tag}" / "news").exists()
        assert not (out_mism / "upload").exists()

    # 3. Pass: when sha256sum "$BIN" equals PIN_SERVER_SHA256 (with PIN_SKIP_BINARY=0), the window opens, uploads the binary, and runs the job
    default_window_bin_sha = hashlib.sha256(b"x" * 1000).hexdigest()
    rig_pass, job_pass, out_pass = _window(
        "normal",
        extra={
            "PIN_SKIP_BINARY": "0",
            "PIN_SERVER_SHA256": default_window_bin_sha,
            "PIN_MOCK_S": "1",
            "PIN_GATE": "0",
        },
    )
    assert "REFUSED" not in rig_pass and "uploaded llama-server.gz.part00" in rig_pass, rig_pass
    assert job_pass.count("PIN_RESULT") == 1 and (out_pass / "pin-result.json").exists(), job_pass


def test_offrig_bench_stub_window_uploads_binary_and_gold_and_downloads_out_tar_and_ps():
    import hashlib
    import json
    import socket
    import tarfile

    for p in (ROOT / "src", ROOT / "tests"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from stub_llama import FAKE_OFFRIG_PY

    tmp = Path(tempfile.mkdtemp(prefix="fakewin-offrig-bench-"))
    offrig_bin = tmp / "offrig"
    offrig_bin.write_text(FAKE_OFFRIG_PY, encoding="utf-8", newline="\n")
    offrig_bin.chmod(0o755)
    bin_sha = hashlib.sha256(offrig_bin.read_bytes()).hexdigest()

    gold = tmp / "bench30.jsonl"
    gold.write_text(
        '{"id": "c-01", "claim": "West pier opens at 06:00.", "label": "unsupported", "split": "tune"}\n',
        encoding="utf-8",
        newline="\n",
    )
    gold_sha = hashlib.sha256(gold.read_bytes()).hexdigest()

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    out = tmp / "out"
    env = {
        **os.environ,
        "PATH": f"{FAKE}:{os.environ['PATH']}",
        "FAKE_ST": str(tmp / "st"),
        "FAKE_MODE": "normal",
        "PIN_OPEN_LOCK": str(tmp / "open.lock"),
        "PIN_GPU": "CPU",
        "PIN_STUB": "1",
        "PIN_ROLE": "offrig-bench",
        "PIN_JOB_SCRIPT": "pin_window.py",
        "PIN_UPLOADS": str(gold),
        "PIN_OFFRIG_SHA256": bin_sha,
        "PIN_OFFRIG_GOLD_SHA256": gold_sha,
        "PIN_OLLAMA_PORT": str(port),
        "PIN_WATCH_S": "1",
        "PIN_STALL_FIRST_S": "15",
        "PIN_STALL_S": "30",
        "PIN_WALL_S": "120",
        "PIN_SESSION": "fake",
        "PIN_CAP_UNITS": "99",
        "PIN_CAP_MARGIN": "0.01",
        "PIN_DEADLINE_GRACE_S": "2",
        "PIN_FETCH_ALLOW_S": "2",
    }
    stdout, stderr = _run_window(["bash", str(ROOT / "scripts/colab/smoke_window.sh"), str(offrig_bin), str(out)], env, 120)
    rig = stdout + stderr
    assert "uploaded offrig" in rig and "uploaded bench30.jsonl" in rig, rig
    assert not list((out / "upload").glob("llama-server.gz*"))
    assert (out / "pin-result.json").is_file(), rig
    assert (out / "offrig-bench-out.tar").is_file(), rig
    assert (out / "ps.json").is_file(), rig

    res = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is True and res["stub"] is True and res["role"] == "offrig-bench", res
    assert res["offrig_bench"]["verdicts_count"] == 1
    ps = json.loads((out / "ps.json").read_text(encoding="utf-8"))
    assert len(ps["models"]) == 1 and ps["models"][0]["name"] == "gemma4:31b"
    with tarfile.open(out / "offrig-bench-out.tar") as tf:
        names = set(tf.getnames())
    assert "offrig-bench/out/verdicts.jsonl" in names and "offrig-bench/smi.csv" in names


def test_update_banner_is_silenced_and_stripped_so_idle_retry_fires():
    import json

    tmp = Path(tempfile.mkdtemp(prefix="fakewin-banner-"))
    settings_path = tmp / "colab-cli" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text('{"default_gpu": "A100", "enable_update_check": true}\n', encoding="utf-8")

    # Even when a banner is printed on status (`FAKE_BANNER=always`), the banner lines are stripped before `cut -c1-300`
    # so `Status: IDLE` is kept in status.log and the `EXEC NEVER STARTED` retry fires and completes the job.
    rig, job, out = _window(
        "lostfirst",
        extra={
            "COLAB_SETTINGS_FILE": str(settings_path),
            "FAKE_BANNER": "always",
            "PIN_MOCK_S": "3",
        },
    )
    saved_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved_settings["enable_update_check"] is False and saved_settings["default_gpu"] == "A100", saved_settings
    status_lines = [ln for ln in (out / "status.log").read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert status_lines and all("Status:" in ln for ln in status_lines), status_lines
    assert all("[colab] A new version" not in ln for ln in status_lines), status_lines
    assert rig.count("EXEC NEVER STARTED") == 1, rig
    assert "STALL" not in rig and job.count("PIN_RESULT") == 1, (rig, job)


def test_timed_out_or_nonzero_upload_and_new_fail_under_pipefail():
    tmp = Path(tempfile.mkdtemp(prefix="fakewin-pipefail-"))
    binfile = tmp / "bin.dat"
    binfile.write_bytes(b"x" * 1000)
    upload_in = tmp / "input.txt"
    upload_in.write_text("payload\n", encoding="utf-8")

    # 1. A timed-out upload (exit 124) whose last line holds neither "fail" nor "error" must fail, never run the job.
    rig, job, out = _window(
        "normal",
        extra={
            "PIN_UPLOADS": str(upload_in),
            "FAKE_UPLOAD_MODE": "timeout_silent",
            "PIN_UPLOAD_RETRY_SLEEP_S": "1",
            "PIN_GATE": "0",
        },
    )
    assert "UPLOAD FAILED: input.txt" in rig, rig
    assert not (out.parent / "st" / "execs").exists() and not job, (rig, job)

    # 2. A failing `colab new` whose output is piped to `tail -2` must exit 2 with `no A100 assigned`.
    rig_new, job_new, out_new = _window(
        "normal",
        extra={"FAKE_NEW_FAIL": "1"},
    )
    assert "no A100 assigned" in rig_new, rig_new
    assert not (out_new.parent / "st" / "execs").exists() and not job_new, (rig_new, job_new)


def test_scrub_log_redacts_proxy_tokens_access_tokens_and_jwts_across_all_logs():
    # Split token pieces so no literal token pattern sits in the test source.
    jwt_hdr = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ"
    jwt_pay = "eyJzdWIiOiJ3aW5kb3cxOGEiLCJpYXQiOjE3MDAwMDAwMDB9"
    jwt_sig = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    fake_jwt = f"{jwt_hdr}.{jwt_pay}.{jwt_sig}"
    fake_access = "ya29.a0AfH6SMB_SECRET_ACCESS_TOKEN_998877"
    leak_line = (
        "Server error '500 Internal Server Error' for url "
        "'https://8080-gpu-a100-s-kkb-usc1b2-1a2b.prod.colab.dev/api/contents/content/input.txt"
        f"?colab-runtime-proxy-token={fake_jwt}&access_token={fake_access}'"
    )
    tmp = Path(tempfile.mkdtemp(prefix="fakewin-scrub-"))
    upload_in = tmp / "input.txt"
    upload_in.write_text("hello\n", encoding="utf-8")

    rig, job, out = _window(
        "normal",
        extra={
            "PIN_UPLOADS": str(upload_in),
            "FAKE_LEAK_URL": leak_line,
            "FAKE_UPLOAD_FAIL_ONCE": "1",
            "FAKE_DOWNLOAD_LEAK_ONCE": "1",
            "PIN_UPLOAD_RETRY_SLEEP_S": "1",
            "PIN_MOCK_S": "3",
        },
    )
    assert job.count("PIN_RESULT") == 1, (rig, job)
    assert "colab-runtime-proxy-token=[REDACTED]" in rig, rig
    assert "access_token=[REDACTED]" in rig, rig
    status_text = (out / "status.log").read_text(encoding="utf-8")
    exec_text = (out / "smoke-window.log").read_text(encoding="utf-8")
    assert "colab-runtime-proxy-token=[REDACTED]" in status_text, status_text
    assert "colab-runtime-proxy-token=[REDACTED]" in exec_text, exec_text
    for secret in (fake_jwt, jwt_pay, jwt_sig, fake_access):
        assert secret not in rig, (secret, rig)
        for p in out.rglob("*"):
            if p.is_file():
                assert secret.encode("utf-8") not in p.read_bytes(), (secret, p)


def test_rejected_gate_vms_are_counted_in_running_spend_and_deadline():
    import json
    import re

    # 1 rejected gate VM (try 1 = bad, try 2 = good): try 1 runs `sleep 15` + `vm_gate` + `colab stop` + `sleep 5` (>= 20 s).
    # Without counting rejected gate VMs, `spent` only counts the ~2 s after try 2 passed (~0.01 units at 5.30/hr).
    # With rejected gate VMs counted, the ~20 s from try 1 are booked into T_OPEN, PIN_DEADLINE, and `spent` (>= 0.03 units).
    rig, job, out = _window(
        "normal",
        extra={
            "FAKE_GATE": "bad,good",
            "PIN_MOCK_S": "2",
        },
    )
    assert "VM gate rejected try 1" in rig and "VM gate passed (try 2)" in rig, rig
    m_rej = re.search(r"rejected gate VMs: (\d+)s", rig)
    assert m_rej is not None and int(m_rej.group(1)) >= 20, rig
    rejected_s = int(m_rej.group(1))
    draft = json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))
    # 20 s of rejected VM #1 + >= 2 s of accepted VM #2 at 5.30/hr is >= 22 * 5.30 / 3600 = 0.032 units -> >= 0.03 in draft.
    assert draft["spent"] >= round(5.30 * rejected_s / 3600, 2), (draft, rejected_s)


def test_open_lock_is_released_before_vm_gate_runs_and_preserves_neighbour_rate():
    rig, job, out = _window(
        "neighbourclose",
        extra={
            "FAKE_GATE": "bad,good",
            "PIN_MOCK_S": "2",
        },
    )
    st = out.parent / "st"
    new_locks = (st / "new-lock.log").read_text(encoding="utf-8").splitlines()
    gate_locks = (st / "gate-lock.log").read_text(encoding="utf-8").splitlines()
    stop_locks = (st / "stop-lock.log").read_text(encoding="utf-8").splitlines()
    assert new_locks == ["locked", "locked"], new_locks
    assert gate_locks == ["unlocked", "unlocked"], gate_locks
    assert stop_locks == ["locked", "locked"], stop_locks
    assert "own rate 5.3/hr" in rig and job.count("PIN_RESULT") == 1, rig


def test_gate_jsonl_records_vm_name_zone_and_pool_from_colab_sessions_for_every_try():
    rig, job, out = _window(
        "normal",
        extra={
            "FAKE_GATE": "bad,dead,good",
            "FAKE_VM": "gpu-a100-s-kkb-usc1b2-1111,gpu-a100-s-kkb-usc1b2-2222,gpu-a100-s-kkb-usw1a1-3333",
            "PIN_GATE_TIMEOUT_S": "3",
            "PIN_MOCK_S": "2",
        },
    )
    rows = _gate_rows(out)
    assert [(r["try"], r.get("vm"), r.get("zone"), r.get("pool"), r["gate"]["ok"]) for r in rows] == [
        (1, "gpu-a100-s-kkb-usc1b2-1111", "usc1b2", "kkb", False),
        (2, "gpu-a100-s-kkb-usc1b2-2222", "usc1b2", "kkb", False),
        (3, "gpu-a100-s-kkb-usw1a1-3333", "usw1a1", "kkb", True),
    ], rows


def test_toolcount_role_uploads_toolcount_pkg_tar_and_forwards_its_sha256():
    import json

    rig, job, out = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_MOCK_S": "1",
            "PIN_GATE": "0",
        },
    )
    st = out.parent / "st"
    uploads = (st / "uploads.log").read_text(encoding="utf-8").splitlines()
    assert "vm1 /content/toolcount-pkg.tar" in uploads, uploads
    assert "uploaded toolcount-pkg.tar" in rig, rig
    assert (st / "vm" / "content" / "toolcount-pkg.tar").is_file()
    job_sh = (st / "job-1.sh").read_text(encoding="utf-8")
    assert "PIN_TOOLCOUNT_PKG_SHA256=" in job_sh, job_sh
    res = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is True, res


def test_journal_copyback_grows_lands_whole_scrubs_tokens_and_rejects_corrupt_or_truncated_pulls():
    import json

    jwt_hdr = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjEifQ"
    jwt_pay = "eyJzdWIiOiJjb3B5YmFjayIsImlhdCI6MTcwMDAwMDAwMH0"
    jwt_sig = "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    fake_jwt = f"{jwt_hdr}.{jwt_pay}.{jwt_sig}"
    fake_access = "ya29.a0AfH6SMB_COPYBACK_SECRET_TOKEN_112233"
    fake_bearer = "ya29.BEARER_SECRET_99887766"
    fake_short_hf = "hf_" + "short98765"
    leak_line = (
        "https://8080-m-s-kkb-usc1b2-1a2b.prod.colab.dev/api/contents/content/pin/toolcount/granite-4.2-8b/calls.jsonl"
        f"?colab-runtime-proxy-token={fake_jwt}&access_token={fake_access} "
        f"Authorization: Bearer {fake_bearer} token={fake_short_hf}"
    )
    hb_leak_line = (
        f"[00:00:01] HEARTBEAT toolcount model=granite-4.2-8b call=1/7 elapsed=1s "
        f"auth=Bearer {fake_bearer} hf={fake_short_hf}"
    )

    # Refuses PIN_STUB_DROP_AFTER_CALLS on any GPU window or when PIN_STUB != 1, and PIN_STUB_CALL_S when PIN_STUB != 1
    rig_refused, job_refused, out_refused = _window(
        "normal",
        extra={"PIN_GPU": "A100", "PIN_STUB_DROP_AFTER_CALLS": "3"},
    )
    assert "REFUSED: PIN_STUB_DROP_AFTER_CALLS is a stub rung's setting" in rig_refused, rig_refused
    assert not (out_refused.parent / "st" / "open").exists() and not job_refused

    rig_call_refused, job_call_refused, out_call_refused = _window(
        "normal",
        extra={"PIN_GPU": "CPU", "PIN_STUB": "0", "PIN_STUB_CALL_S": "1"},
    )
    assert "REFUSED: PIN_STUB_CALL_S is a stub rung's setting" in rig_call_refused, rig_call_refused
    assert not (out_call_refused.parent / "st" / "open").exists() and not job_call_refused

    # Run in an output directory whose name contains "fail" and "error" to verify 0.7.4's
    # `[colab] Downloaded '<remote>' to '<local>'` output does not trigger false-positive error rejection.
    rig, job, out = _window(
        "normal",
        out_name="test-fail-error-out",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "1",
            "PIN_WATCH_S": "1",
            "PIN_STUB_CALL_S": "1.2",
            "PIN_GATE": "0",
            "FAKE_PARTIAL_LAST_LINE_ON": "2",
            "FAKE_UNASSIGN_MID_DOWNLOAD_ON": "3",
            "FAKE_CORRUPT_COPYBACK_ON": "4",
            "FAKE_TRUNCATE_COPYBACK_ON": "5",
            "FAKE_COPYBACK_LEAK_URL": leak_line,
            "FAKE_HB_LINE": hb_leak_line,
        },
    )
    assert job.count("PIN_RESULT") == 1, (rig, job)

    cb_dir = out / "copyback" / "granite-4.2-8b"
    calls_cb = cb_dir / "calls.jsonl"
    gens_cb = cb_dir / "generations.jsonl"
    hb_cb = cb_dir / "heartbeat.txt"
    hb_top = out / "copyback" / "heartbeat.txt"
    state_cb = out / "copyback" / ".copyback.state.json"
    assert calls_cb.is_file() and gens_cb.is_file() and hb_cb.is_file() and hb_top.is_file() and state_cb.is_file()
    assert "HEARTBEAT toolcount model=granite-4.2-8b" in hb_cb.read_text(encoding="utf-8")
    assert hb_cb.read_text(encoding="utf-8") == hb_top.read_text(encoding="utf-8")
    # No leftover .part files
    assert not list((out / "copyback").rglob("*.part*"))

    # Every line in copied calls.jsonl and generations.jsonl parses as JSON with call_id (partial tail was stripped)
    copied_calls = [json.loads(ln) for ln in calls_cb.read_text(encoding="utf-8").splitlines() if ln.strip()]
    copied_gens = [json.loads(ln) for ln in gens_cb.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(copied_calls) == 7 and len(copied_gens) == 7, (len(copied_calls), len(copied_gens))
    assert all(r["call_id"] != "partial_mid_append" for r in copied_calls)

    # Verify copyback.jsonl logs every pull with time, file, bytes, lines, rc
    cb_log_1 = out / "copyback" / "copyback.jsonl"
    cb_log_2 = out / "copyback.jsonl"
    assert cb_log_1.is_file() and cb_log_2.is_file()
    assert cb_log_1.read_text(encoding="utf-8") == cb_log_2.read_text(encoding="utf-8")
    log_rows = [json.loads(ln) for ln in cb_log_1.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for r in log_rows:
        assert set(r.keys()) == {"time", "file", "bytes", "lines", "rc"}, r

    call_pulls = [r for r in log_rows if r["file"] == "granite-4.2-8b/calls.jsonl"]
    assert len(call_pulls) >= 6, call_pulls
    # Pull #1 is valid (rc == 0, lines >= 1)
    assert call_pulls[0]["rc"] == 0 and call_pulls[0]["lines"] >= 1, call_pulls
    # Pull #2 had a partial trailing line after new complete lines -> salvaged (rc == 0, lines > Pull #1)
    assert call_pulls[1]["rc"] == 0 and call_pulls[1]["lines"] > call_pulls[0]["lines"], call_pulls
    # Pull #3 (mid-download unassign, rc == 1), Pull #4 (corrupt JSON line, rc == 2), Pull #5 (truncated 1-line pull, rc == 2)
    # are all rejected and preserve Pull #2's line count and bytes
    assert call_pulls[2]["rc"] == 1 and call_pulls[2]["lines"] == call_pulls[1]["lines"], call_pulls
    assert call_pulls[3]["rc"] == 2 and call_pulls[3]["lines"] == call_pulls[1]["lines"], call_pulls
    assert call_pulls[4]["rc"] == 2 and call_pulls[4]["lines"] == call_pulls[1]["lines"], call_pulls
    # Final pull reaches all 7 lines
    assert call_pulls[-1]["rc"] == 0 and call_pulls[-1]["lines"] == 7, call_pulls

    # Proxy tokens, Bearer tokens, and short hf_ tokens are scrubbed from all kept CLI output and .copyback.state.json
    fetch_out = (out / "copyback" / ".copyback.fetch.out").read_text(encoding="utf-8")
    assert "colab-runtime-proxy-token=[REDACTED]" in fetch_out, fetch_out
    assert "access_token=[REDACTED]" in fetch_out, fetch_out
    assert "Bearer REDACTED" in fetch_out, fetch_out
    for secret in (fake_jwt, jwt_pay, jwt_sig, fake_access, fake_bearer, fake_short_hf):
        assert secret not in rig, (secret, rig)
        for p in out.rglob("*"):
            if p.is_file() and p.name != "job.log":
                assert secret.encode("utf-8") not in p.read_bytes(), (secret, p)


def test_copyback_synchronization_under_fast_writer_and_mid_call_watcher_interleaving():
    """Deterministic regression test for the copy-back synchronization races in Run 40850724572 / e4d2057 (c2dbcf4c):
    Force the bad interleavings deterministically (no real sleeps in the test):
    1. In `scripts/toolcount/runner.py`, verify `generations.jsonl` is flushed BEFORE `calls.jsonl` on every call
       (fresh and reused). At c2dbcf4c, `calls_f.flush()` ran before `gen_f.write()` / `gen_f.flush()`, so a
       `copyback_journals` pull (or `PIN_STUB_DROP_AFTER_CALLS` / `FAKE_DROP_SESSION_AT_CALLS` trigger) immediately
       after `calls_f.flush()` saw `N` lines in `calls.jsonl` and `N - 1` lines in `generations.jsonl`.
       Here we intercept `calls.jsonl` flushes in `run_model_window` with a controlled wrapper and assert that at the
       exact moment each line of `calls.jsonl` is flushed, `generations.jsonl` on disk already has the matching line count.
    2. In `scripts/colab/smoke_window.sh` and `tests/fake_colab/colab`:
       a. When `copyback_journals` runs after `=== granite-4.2-8b ===` appears in `job.log` but before `calls.jsonl`
          exists on the VM (`[colab] Download failed: /content/pin/toolcount/granite-4.2-8b/calls.jsonl`), `copyback_journals`
          must skip the not-yet-created file rather than logging `bytes: 0, lines: 0, rc: 1` as Pull #1.
       b. When all 7 VM call lines finish before the watcher's first `copyback_journals` pull (`PIN_STUB_CALL_S=0`),
          Pull #2 (`FAKE_PARTIAL_LAST_LINE_ON=2`) must still have new complete lines after Pull #1 (`rc=0, lines > call_pulls[0]['lines']`)
          rather than failing with `rc=2` because Pull #1 already consumed all 7 lines.
    """
    import json
    import tempfile

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(ROOT / "scripts" / "toolcount") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts" / "toolcount"))
    if str(ROOT / "scripts" / "colab") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts" / "colab"))
    import gen_registry
    import compass_lists
    import runner as tc_runner

    with tempfile.TemporaryDirectory(prefix="tc-order-") as td:
        td_p = Path(td)
        r0_dir = td_p / "r0"
        out_tc = td_p / "out"
        gen_registry.generate_all(out_dir=r0_dir, seed=20261028)
        compass_lists.write_compass_artifacts(r0_dir=r0_dir, out_dir=r0_dir, seed=20261028, embedder_mode="fake")
        items_data = json.loads((r0_dir / "items.json").read_text(encoding="utf-8"))
        stub_sched = tc_runner.build_r1_stub_schedule(items_data)[:3]

        calls_path = out_tc / "calls.jsonl"
        gens_path = out_tc / "generations.jsonl"
        observed_counts: list[tuple[int, int]] = []
        orig_open = Path.open

        def wrapped_open(self_path: Path, mode: str = "r", *args, **kwargs):
            fh = orig_open(self_path, mode, *args, **kwargs)
            if self_path == calls_path and "w" in mode:
                orig_flush = fh.flush

                class _FlushObserver:
                    def __init__(self, inner):
                        self._inner = inner

                    def write(self, s):
                        return self._inner.write(s)

                    def flush(self):
                        res = orig_flush()
                        c_lines = (
                            len([ln for ln in calls_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
                            if calls_path.is_file()
                            else 0
                        )
                        g_lines = (
                            len([ln for ln in gens_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
                            if gens_path.is_file()
                            else 0
                        )
                        if c_lines > 0:
                            observed_counts.append((c_lines, g_lines))
                        return res

                    def __enter__(self):
                        self._inner.__enter__()
                        return self

                    def __exit__(self, *exc):
                        return self._inner.__exit__(*exc)

                    def __getattr__(self, name):
                        return getattr(self._inner, name)

                return _FlushObserver(fh)
            return fh

        Path.open = wrapped_open  # type: ignore[method-assign]
        try:
            with tc_runner.StubToolCountServer() as srv:
                tc_runner.run_model_window(
                    endpoint_url=srv.base_url,
                    model_name="granite-4.2-8b",
                    r0_dir=r0_dir,
                    out_dir=out_tc,
                    half="H1",
                    seed=20261028,
                    schedule_override=stub_sched,
                    machine_line="stub",
                )
        finally:
            Path.open = orig_open  # type: ignore[method-assign]

        assert observed_counts == [(1, 1), (2, 2), (3, 3)], (
            f"generations.jsonl lagged behind calls.jsonl at calls_f.flush(): {observed_counts}"
        )

        # Also verify the reuse path (`call_id in reused_map`) flushes generations.jsonl before calls.jsonl
        out_reused = td_p / "out-reused"
        calls_reused_path = out_reused / "calls.jsonl"
        gens_reused_path = out_reused / "generations.jsonl"
        observed_reused_counts: list[tuple[int, int]] = []

        def wrapped_open_reused(self_path: Path, mode: str = "r", *args, **kwargs):
            fh = orig_open(self_path, mode, *args, **kwargs)
            if self_path == calls_reused_path and "w" in mode:
                orig_flush = fh.flush

                class _ReusedFlushObserver:
                    def __init__(self, inner):
                        self._inner = inner

                    def write(self, s):
                        return self._inner.write(s)

                    def flush(self):
                        res = orig_flush()
                        c_lines = (
                            len([ln for ln in calls_reused_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
                            if calls_reused_path.is_file()
                            else 0
                        )
                        g_lines = (
                            len([ln for ln in gens_reused_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
                            if gens_reused_path.is_file()
                            else 0
                        )
                        if c_lines > 0:
                            observed_reused_counts.append((c_lines, g_lines))
                        return res

                    def __enter__(self):
                        self._inner.__enter__()
                        return self

                    def __exit__(self, *exc):
                        return self._inner.__exit__(*exc)

                    def __getattr__(self, name):
                        return getattr(self._inner, name)

                return _ReusedFlushObserver(fh)
            return fh

        Path.open = wrapped_open_reused  # type: ignore[method-assign]
        try:
            with tc_runner.StubToolCountServer() as srv:
                tc_runner.run_model_window(
                    endpoint_url=srv.base_url,
                    model_name="granite-4.2-8b",
                    r0_dir=r0_dir,
                    out_dir=out_reused,
                    half="H1",
                    seed=20261028,
                    resume_calls_dir=out_tc,
                    schedule_override=stub_sched,
                    machine_line="stub",
                )
        finally:
            Path.open = orig_open  # type: ignore[method-assign]

        assert observed_reused_counts == [(1, 1), (2, 2), (3, 3)], (
            f"reused generations.jsonl lagged behind calls.jsonl at calls_f.flush(): {observed_reused_counts}"
        )

    rig, job, out = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "1",
            "PIN_WATCH_S": "1",
            "PIN_GATE": "0",
            "FAKE_EXEC_HANG_AFTER_DONE_S": "10",
            "FAKE_PARTIAL_LAST_LINE_ON": "2",
            "FAKE_UNASSIGN_MID_DOWNLOAD_ON": "3",
            "FAKE_CORRUPT_COPYBACK_ON": "4",
            "FAKE_TRUNCATE_COPYBACK_ON": "5",
        },
    )
    assert job.count("PIN_RESULT") == 1, (rig, job)
    cb_log = out / "copyback" / "copyback.jsonl"
    assert cb_log.is_file(), rig
    log_rows = [json.loads(ln) for ln in cb_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
    call_pulls = [r for r in log_rows if r["file"] == "granite-4.2-8b/calls.jsonl"]
    gen_pulls = [r for r in log_rows if r["file"] == "granite-4.2-8b/generations.jsonl"]
    assert len(call_pulls) >= 6, call_pulls
    assert call_pulls[0]["rc"] == 0 and call_pulls[0]["lines"] >= 1, call_pulls
    assert call_pulls[1]["rc"] == 0 and call_pulls[1]["lines"] > call_pulls[0]["lines"], call_pulls
    assert call_pulls[2]["rc"] == 1 and call_pulls[2]["lines"] == call_pulls[1]["lines"], call_pulls
    assert call_pulls[3]["rc"] == 2 and call_pulls[3]["lines"] == call_pulls[1]["lines"], call_pulls
    assert call_pulls[4]["rc"] == 2 and call_pulls[4]["lines"] == call_pulls[1]["lines"], call_pulls
    assert call_pulls[-1]["rc"] == 0 and call_pulls[-1]["lines"] == 7, call_pulls
    assert gen_pulls[-1]["rc"] == 0 and gen_pulls[-1]["lines"] == 7, gen_pulls



def test_journal_copyback_survives_midrun_session_drop_and_resumes_byte_identically():
    import json

    seed = "20261028"
    # 1. Uninterrupted reference run (all 7 R1 stub calls for seed 20261028)
    rig_ref, job_ref, out_ref = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_TOOLCOUNT_SEED": seed,
            "PIN_COPYBACK_S": "0",
            "PIN_WATCH_S": "1",
            "PIN_GATE": "0",
            "PIN_DOWNLOADS": "pin-result.json calls.jsonl generations.jsonl",
        },
    )
    assert job_ref.count("PIN_RESULT") == 1, (rig_ref, job_ref)
    ref_pin = json.loads((out_ref / "pin-result.json").read_text(encoding="utf-8"))
    assert ref_pin["machine"]["cores"] == 2 and ref_pin["machine"]["disk_free_gb"] == 100.0, ref_pin["machine"]
    ref_calls_bytes = (out_ref / "calls.jsonl").read_bytes()
    ref_gens_bytes = (out_ref / "generations.jsonl").read_bytes()
    ref_calls_lines = ref_calls_bytes.decode("utf-8").splitlines()
    assert len(ref_calls_lines) == 7

    # 2. Dropped session mid-run after N=3 calls are copied back
    rig_drop, job_drop, out_drop = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_TOOLCOUNT_SEED": seed,
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "1",
            "PIN_WATCH_S": "1",
            "PIN_STUB_CALL_S": "0.35",
            "PIN_STUB_DROP_AFTER_CALLS": "3",
            "PIN_GATE": "0",
        },
    )
    assert "SESSION GONE" in rig_drop, rig_drop
    assert (out_drop / ".end").read_text(encoding="utf-8").strip() == "gone"
    draft_drop = json.loads((out_drop / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft_drop["end"] == "gone", draft_drop

    cb_calls = out_drop / "copyback" / "granite-4.2-8b" / "calls.jsonl"
    cb_gens = out_drop / "copyback" / "granite-4.2-8b" / "generations.jsonl"
    assert cb_calls.is_file() and cb_gens.is_file()
    copied_lines = cb_calls.read_text(encoding="utf-8").splitlines()
    assert len(copied_lines) == 3, copied_lines
    assert copied_lines == ref_calls_lines[:3]

    # Also verify a mid-run drop triggered directly by fake_colab (FAKE_DROP_AFTER_COPYBACKS=1) closes as "gone" with copy intact
    rig_fakedrop, _, out_fakedrop = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_TOOLCOUNT_SEED": seed,
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "1",
            "PIN_WATCH_S": "1",
            "PIN_STUB_CALL_S": "1.2",
            "FAKE_DROP_AFTER_COPYBACKS": "1",
            "PIN_GATE": "0",
        },
    )
    assert "SESSION GONE" in rig_fakedrop, rig_fakedrop
    assert json.loads((out_fakedrop / "outcome.draft.json").read_text(encoding="utf-8"))["end"] == "gone"
    assert (out_fakedrop / "copyback" / "granite-4.2-8b" / "calls.jsonl").is_file()

    # 3. Resume from PIN_RESUME_CALLS_DIR=<out_drop>/copyback
    rig_res, job_res, out_res = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_TOOLCOUNT_SEED": seed,
            "PIN_RESUME_CALLS_DIR": str(out_drop / "copyback"),
            "PIN_COPYBACK_S": "1",
            "PIN_WATCH_S": "1",
            "PIN_GATE": "0",
            "PIN_DOWNLOADS": "pin-result.json calls.jsonl generations.jsonl",
        },
    )
    assert "uploaded resume.tar" in rig_res, rig_res
    assert job_res.count("PIN_RESULT") == 1, (rig_res, job_res)
    pin_res = json.loads((out_res / "pin-result.json").read_text(encoding="utf-8"))
    assert pin_res["ok"] is True, pin_res
    summary = pin_res["models"][0]["window_summary"]
    assert summary["total_calls"] == 7, summary
    assert summary["reused_calls"] == 3, summary
    assert summary["fresh_calls"] == 4, summary  # total (7) - N (3) == 4 fresh calls

    resumed_calls_bytes = (out_res / "calls.jsonl").read_bytes()
    resumed_gens_bytes = (out_res / "generations.jsonl").read_bytes()
    resumed_rows = [json.loads(ln) for ln in resumed_calls_bytes.decode("utf-8").splitlines()]
    call_ids = [r["call_id"] for r in resumed_rows]
    assert len(call_ids) == 7 and len(set(call_ids)) == 7, call_ids  # no call_id repeated
    # Final journal equals what an uninterrupted run produces for the same seed (byte-for-byte)
    assert resumed_calls_bytes == ref_calls_bytes
    assert resumed_gens_bytes == ref_gens_bytes


def test_session_drop_after_new_journal_lines_runs_final_copyback_freezes_spend_and_closes_gone():
    import json
    import time

    # Periodic PIN_COPYBACK_S=60 means after the initial watcher pull at t=1s (when only 1-2 calls exist),
    # no further periodic copyback fires before call #3 lands. Once 3 calls land on the VM,
    # FAKE_DROP_SESSION_AT_CALLS=3 removes [$S] from `colab sessions` while the exec hangs ignoring SIGTERM.
    # The SESSION GONE handler must:
    #   1. Run one final copy-back (`fetch_log gone` + `copyback_journals`) so all 3 calls land in <out>/copyback/,
    #   2. Record .end.ts and freeze spend at that timestamp (not charging the 5s SIGKILL escalation + 5s stop sleep),
    #   3. Escalate from SIGTERM to SIGKILL to terminate the wedged exec and close as "gone".
    t_start = time.time()
    rig, _job, out = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "60",
            "PIN_WATCH_S": "1",
            "PIN_WALL_S": "15",
            "PIN_STUB_CALL_S": "0.6",
            "PIN_STUB_DROP_AFTER_CALLS": "3",
            "FAKE_DROP_SESSION_AT_CALLS": "3",
            "FAKE_EXEC_IGNORE_TERM": "1",
            "PIN_KILL_ESCALATE_S": "5",
            "PIN_GATE": "0",
        },
    )
    t_done = time.time()
    assert "SESSION GONE (fake)" in rig, rig
    assert (out / ".end").read_text(encoding="utf-8").strip() == "gone"
    assert (out / ".end.ts").is_file(), "missing .end.ts timestamp for spend freeze"
    end_ts = int((out / ".end.ts").read_text(encoding="utf-8").strip())
    # The script ran ~10s past .end.ts (5s SIGKILL escalation + 5s stop sleep), so end_ts must precede t_done by >= 7s
    assert t_done - end_ts >= 7, (end_ts, t_done)

    draft = json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["end"] == "gone", draft
    # Fake account rate is 5.30/hr when open; frozen runtime (end_ts - T_OPEN) is ~3-5s (<= 0.01 units),
    # whereas un-frozen runtime up to outcome.draft.json (~9-11s) would be >= 0.01-0.02 units.
    frozen_elapsed = max(0, end_ts - int(t_start))
    expected_max_spent = round(5.30 * (frozen_elapsed + 2) / 3600, 2)
    assert draft["spent"] <= expected_max_spent, (draft, frozen_elapsed, t_done - t_start)

    # Final copy-back on SESSION GONE pulled all 3 calls that landed before the drop
    cb_calls = out / "copyback" / "granite-4.2-8b" / "calls.jsonl"
    cb_gens = out / "copyback" / "granite-4.2-8b" / "generations.jsonl"
    assert cb_calls.is_file() and cb_gens.is_file()
    copied_calls = [json.loads(ln) for ln in cb_calls.read_text(encoding="utf-8").splitlines() if ln.strip()]
    copied_gens = [json.loads(ln) for ln in cb_gens.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(copied_calls) == 3 and len(copied_gens) == 3, (len(copied_calls), len(copied_gens))

    # Also verify Fix 3: when CAP CEILING fires in the watcher, it calls `end_window cap` (not `stop_all`),
    # so the main shell runs the final `copyback_journals` BEFORE `colab stop` tears down the VM.
    rig_cap, _job_cap, out_cap = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "60",
            "PIN_WATCH_S": "1",
            "PIN_WALL_S": "25",
            "PIN_STUB_CALL_S": "0.4",
            "PIN_STUB_DROP_AFTER_CALLS": "4",
            "FAKE_EXEC_HANG_AFTER_DONE_S": "15",
            "FAKE_EXEC_IGNORE_TERM": "1",
            "PIN_KILL_ESCALATE_S": "2",
            "PIN_CAP_UNITS": "0.008",
            "PIN_CAP_MARGIN": "0.0005",
            "PIN_DEADLINE_GRACE_S": "10",
            "PIN_FETCH_ALLOW_S": "1",
            "PIN_GATE": "0",
        },
    )
    assert "CAP CEILING" in rig_cap, rig_cap
    assert json.loads((out_cap / "outcome.draft.json").read_text(encoding="utf-8"))["end"] == "cap"
    cap_calls = [
        json.loads(ln)
        for ln in (out_cap / "copyback" / "granite-4.2-8b" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(cap_calls) == 4, len(cap_calls)


def test_hung_exec_after_pin_result_takes_exit_sequence_within_one_watch_interval_and_kills_exec():
    import json
    import re

    # Fix 4 (C2:82 finish detection): `colab exec` hangs after `PIN_RESULT` is written to job.log and ignores SIGTERM
    # (simulating a >1 h exec with a dead websocket). Within one watcher interval (PIN_WATCH_S=3s), the watcher must
    # see `^PIN_RESULT` with this window's job_token BEFORE running periodic copyback, call `end_window done`,
    # escalate to SIGKILL after PIN_KILL_ESCALATE_S=5s, run the single exit-sequence fetch/copyback, and close as "done".
    rig, job, out = _window(
        "normal",
        extra={
            "PIN_ROLE": "toolcount",
            "PIN_STUB": "1",
            "PIN_GPU": "CPU",
            "PIN_JOB_SCRIPT": "pin_window.py",
            "PIN_MODELS": "granite-4.2-8b",
            "PIN_HEARTBEAT_S": "0",
            "PIN_COPYBACK_S": "3",
            "PIN_WATCH_S": "3",
            "PIN_WALL_S": "25",
            "PIN_KILL_ESCALATE_S": "5",
            "FAKE_EXEC_IGNORE_TERM": "1",
            "FAKE_EXEC_HANG_AFTER_DONE_S": "25",
            "PIN_GATE": "0",
            "PIN_DOWNLOADS": "pin-result.json calls.jsonl generations.jsonl",
        },
    )
    assert "JOB DONE (PIN_RESULT in job.log)" in rig, rig
    assert job.count("PIN_RESULT") == 1, (rig, job)
    m_wall = re.search(r"exec exit=\d+ wall=(\d+)s end=(\w+)", rig)
    assert m_wall is not None, rig
    exec_wall_s = int(m_wall.group(1))
    end_reason = m_wall.group(2)
    # Exec was hung for 60s ignoring SIGTERM, but watcher detected PIN_RESULT within 1s and SIGKILLed it after 5s (< 20s total)
    assert exec_wall_s < 20 and end_reason == "done", (exec_wall_s, end_reason, rig)
    draft = json.loads((out / "outcome.draft.json").read_text(encoding="utf-8"))
    assert draft["end"] == "done", draft
    res = json.loads((out / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is True, res
    # Because PIN_RESULT was checked before periodic copyback in the watcher, copyback_journals only ran once (in the exit sequence)
    cb_rows = [
        json.loads(ln)
        for ln in (out / "copyback" / "copyback.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip() and "calls.jsonl" in ln
    ]
    assert len(cb_rows) == 1 and cb_rows[0]["lines"] == 7, cb_rows


if __name__ == "__main__":  # WSL on the rig: no pytest needed
    only = set(sys.argv[1:])
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and (not only or name in only):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {str(e)[:1500]}")
    sys.exit(1 if failed else 0)
