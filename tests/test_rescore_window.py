"""PIN_ROLE=rescore (R&D, 2026-10-18): the second critic set up and served as in the screen's critique waves, then one
`role-set rescore` on read-only, sha-checked inputs, under the window's deadline. Without Colab, a GPU or the Hub."""

import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))
import pin_window as pw  # noqa: E402

LIGHTNING = "nemotron-3.5-lightning-30b-a3b"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="tar, chmod and a POSIX shell, as on the VM")


def _tar(path: Path, files: dict) -> dict:
    with tarfile.open(path, "w") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


@pytest.fixture
def vm(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "CONTENT", tmp_path)
    monkeypatch.setattr(pw, "WORK", tmp_path / "pin")
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.delenv("PIN_DEADLINE", raising=False)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pw, "TOKEN_PATH", tmp_path / "shm-hf.token")  # absent: no token reaches a rescore window
    monkeypatch.setattr(pw, "_TOKEN", None)
    (tmp_path / "pin").mkdir()
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "pack.json").write_text("{}", encoding="utf-8")
    (tmp_path / "card-values.json").write_text("{}", encoding="utf-8")
    shas = _tar(tmp_path / "rescore-inputs.tar", {"pilot-out-12f.tar": b"tar bytes", "w0/granite-4.2-30b.jsonl": b"w0",
                                                  "w1/granite-4.2-30b.jsonl": b"w1"})
    monkeypatch.setenv("PIN_RESCORE_INPUTS", str(tmp_path / "rescore-inputs.tar"))
    monkeypatch.setenv("PIN_RESCORE_SHAS", ",".join(f"{k}={v}" for k, v in shas.items()))
    monkeypatch.setenv("PIN_RESCORE_RUNS", "pilot-out-12f.tar")
    monkeypatch.setenv("PIN_RESCORE_DRAFTS", "granite-4.2-30b=w0/granite-4.2-30b.jsonl,w1/granite-4.2-30b.jsonl")
    monkeypatch.setenv("PIN_RESCORE_CRITIC", LIGHTNING)
    monkeypatch.setenv("PIN_RESCORE_SEED", "20261018")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "c" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "d" * 40)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: {"role_set": "role-set", "src": str(tmp_path)})
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "sh", lambda cmd, **k: subprocess.run(cmd, shell=True, check=True, capture_output=True,
                                                                     text=True).stdout)
    return tmp_path


def _fake_pool(monkeypatch, events, *, card=True):
    class Srv:
        log = None

        def __enter__(self):
            events.append("serve")
            return self

        def __exit__(self, *exc):
            events.append("stop")

    class Pool:
        def __init__(self, *a, **k):
            pass

        def get(self, name):
            events.append(("get", name))
            return {"served_as": name, "gguf_sha256": "5bea2069" + "0" * 56}

        def serve(self, name):
            return Srv()

        def proxy(self, name):
            return None

    monkeypatch.setattr(pw, "Pool", Pool)


def _fake_rescore(monkeypatch, events, *, card={"temperature": 0.6}, exit=0):
    def run_proc(cmd, **kw):
        events.append(("rescore", cmd))
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "sample-manifest.json").write_text("{}", encoding="utf-8")
        (out / "rescore-report.json").write_text(json.dumps({"critic_card": card, "flagged": []}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, exit, "| arm | n |", "")

    monkeypatch.setattr(pw, "run_proc", run_proc)


def test_the_critic_is_set_up_then_served_then_the_rescore_runs_on_the_checked_inputs(vm, monkeypatch):
    events, result = [], {}
    _fake_pool(monkeypatch, events)
    _fake_rescore(monkeypatch, events)
    pw.rescore_window({}, result)
    assert [e if isinstance(e, str) else e[0] for e in events] == ["get", "serve", "rescore", "stop"]
    cmd = events[2][1]
    root = vm / "pin" / "rescore-in"
    assert cmd[cmd.index("--runs") + 1] == str(root / "pilot-out-12f.tar")
    assert cmd[cmd.index("--drafts") + 1] == f"granite-4.2-30b={root / 'w0/granite-4.2-30b.jsonl'},{root / 'w1/granite-4.2-30b.jsonl'}"
    assert cmd[cmd.index("--seed") + 1] == "20261018" and cmd[cmd.index("--critic-model") + 1] == LIGHTNING
    assert all(result["token_clean"].values()), result["token_clean"]
    assert result["critic_card"] == {"temperature": 0.6} and set(result["rescore_inputs"]["files"]) == {
        "pilot-out-12f.tar", "w0/granite-4.2-30b.jsonl", "w1/granite-4.2-30b.jsonl"}
    names = tarfile.open(vm / "pin" / "rescore-out.tar").getnames()
    assert "rescore/sample-manifest.json" in names and "rescore/rescore-report.json" in names and "rescore/rescore.log" in names


def test_a_critic_gguf_that_isnt_its_pin_stops_before_any_serve(vm, monkeypatch):
    monkeypatch.delenv("PIN_STUB", raising=False)
    served, ran = [], []
    monkeypatch.setattr(pw, "fetch_model", lambda spec, built, keep_snapshot=False: {
        "served_as": spec["served_as"], "gguf_sha256": "f" * 64, "path": "/dev/null"})
    monkeypatch.setattr(pw, "Served", lambda *a, **k: served.append(a))
    monkeypatch.setattr(pw, "run_proc", lambda *a, **k: ran.append(a))
    with pytest.raises(RuntimeError, match="doesn't reproduce the pin"):
        pw.rescore_window({}, {})
    assert not served and not ran


def test_an_input_that_isnt_its_sha_refuses_before_anything_is_set_up(vm, monkeypatch):
    events = []
    _fake_pool(monkeypatch, events)
    _fake_rescore(monkeypatch, events)
    monkeypatch.setenv("PIN_RESCORE_SHAS", "pilot-out-12f.tar=" + "0" * 64 + ",w0/granite-4.2-30b.jsonl=" + "0" * 64
                       + ",w1/granite-4.2-30b.jsonl=" + "0" * 64)
    with pytest.raises(RuntimeError, match="REFUSED: rescore input sha256 mismatch"):
        pw.rescore_window({}, {})
    assert not events


def test_an_input_named_without_a_sha_refuses(vm, monkeypatch):
    monkeypatch.setenv("PIN_RESCORE_SHAS", "pilot-out-12f.tar=" + "0" * 64)
    with pytest.raises(RuntimeError, match="no sha256 in PIN_RESCORE_SHAS for w0/granite"):
        pw.rescore_inputs({})


def test_the_inputs_are_read_only_after_the_check(vm):
    root, _ = pw.rescore_inputs({})
    import os

    assert not os.access(root / "pilot-out-12f.tar", os.W_OK) or os.geteuid() == 0  # root ignores the mode


def test_the_deadline_cuts_the_rescore_and_the_tar_still_returns(vm, monkeypatch):
    events = []
    _fake_pool(monkeypatch, events)
    monkeypatch.setenv("PIN_DEADLINE", str(int(time.time() + 30)))

    def run_proc(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        (out / "sample-manifest.json").write_text("{}", encoding="utf-8")  # written before scoring
        assert kw["term_first"] is True and kw["timeout"] <= 30
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(pw, "run_proc", run_proc)
    with pytest.raises(pw.DeadlineReached, match="rescore was cut"):
        pw.rescore_window({}, {})
    assert pw.CUT_AT["wave"] == "rescore"
    assert "rescore/sample-manifest.json" in tarfile.open(vm / "pin" / "rescore-out.tar").getnames()


def test_a_real_report_without_critic_card_refuses_its_numbers(vm, monkeypatch):
    monkeypatch.delenv("PIN_STUB", raising=False)
    events = []
    _fake_pool(monkeypatch, events)
    _fake_rescore(monkeypatch, events, card=None)
    with pytest.raises(RuntimeError, match="no critic_card"):
        pw.rescore_window({}, {})


def test_a_failed_rescore_fails_the_window(vm, monkeypatch):
    events = []
    _fake_pool(monkeypatch, events)
    _fake_rescore(monkeypatch, events, exit=2)
    with pytest.raises(RuntimeError, match="role-set rescore exited 2"):
        pw.rescore_window({}, {})


def test_a_token_residue_stops_the_rescore_before_anything_is_set_up(vm, monkeypatch):
    events = []
    _fake_pool(monkeypatch, events)
    _fake_rescore(monkeypatch, events)
    monkeypatch.setenv("HF_TOKEN", "x")
    with pytest.raises(RuntimeError, match="token residue is left before any wave"):
        pw.rescore_window({}, {})
    assert not events
