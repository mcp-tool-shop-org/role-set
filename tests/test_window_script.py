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
