"""The Colab window script's own rules that run without a VM."""

import os
import shutil
import subprocess
import sys
import threading
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


# ---- lazy setup and a critic per arm (R&D, 2026-10-17, from Argon's budget-150 reply)

import json  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _the_vm_python_is_cp313(monkeypatch):
    """pin_window.py runs on the VM's system Python (Colab: 3.13), so its zstandard wheel pin is the cp313 one whatever
    interpreter runs pytest. zstandard 0.25.0 has no cp315 wheel (PyPI, read 2026-10-25), so under 3.15 the lookup by the
    running interpreter would refuse. An explicit py_tag still goes through the real table. This assumes the VM's
    system Python is 3.13 (measured 3.13.16 on 2026-10-25); revisit it when Colab moves."""

    real = pw.zstandard_wheel_sha256
    monkeypatch.setattr(pw, "zstandard_wheel_sha256", lambda py_tag=None: real(py_tag or "cp313"))


class _Reclaimed(BaseException):
    """Colab taking the VM back: nothing in the window catches it."""


def _fake_window(monkeypatch, tmp_path, *, die_after_write_of=None, all_dropped=()):
    events = []

    def fetch_model(spec, built, keep_snapshot=False):
        events.append(("fetch", spec["served_as"]))
        return {"served_as": spec["served_as"], "gguf_sha256": spec.get("expect_sha256") or "x" * 64, "path": "/dev/null"}

    class Served:
        def __init__(self, built, m, pinned):
            self.m, self.starts = m, []

        def __enter__(self):
            events.append(("serve", self.m["served_as"]))
            return self

        def __exit__(self, *exc):
            events.append(("stop", self.m["served_as"]))

    def write_wave(rs, common, wave, out, *, writer=None, critic=None, **kw):
        who = (writer or critic)["served_as"]
        events.append((wave, out.parent.name, who))
        if wave == "write" and out.parent.name == die_after_write_of:
            raise _Reclaimed()
        # A write or rewrite leaves its drafts awaiting the critic, unless the case drops them all (R&D on 13b).
        left = 0 if wave == "critique" or out.parent.name in all_dropped else 1
        return {"exit": 0, "result": {"dropped": 0, "states": {"awaiting_critique": left, "needs_rewrite": 0}}}

    monkeypatch.setattr(pw, "fetch_model", fetch_model)
    monkeypatch.setattr(pw, "Served", Served)
    monkeypatch.setattr(pw, "choose_quant", lambda built, spec, m, policy, result: (m, {"policy": policy}))
    monkeypatch.setattr(pw, "j1_card", lambda m, cards, out: (tmp_path / "c.json", m["served_as"], {}))
    monkeypatch.setattr(pw, "write_wave", write_wave)
    monkeypatch.setattr(pw, "run_dir", lambda out: out)
    monkeypatch.setattr(pw, "checkpoint", lambda *a: None)
    monkeypatch.setattr(pw, "PILOT_WAVES", 2)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)
    monkeypatch.delenv("PIN_RESUME_PRIOR_DIR", raising=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"slots": [{"id": f"s{i}"} for i in range(8)]}), encoding="utf-8")
    pool = pw.Pool({}, {}, {}, tmp_path / "cards.json", tmp_path / "out")
    return events, plan, pool


def test_each_writer_is_set_up_just_before_its_own_write_and_the_critic_last(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    A, B, M = "qwen3.6-35b-a3b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    fetches = [e for e in events if e[0] == "fetch"]
    assert fetches == [("fetch", A), ("fetch", B), ("fetch", M)]
    order = [e for e in events if e[0] in ("fetch", "write", "critique")]
    assert order == [("fetch", A), ("write", A, A), ("fetch", B), ("write", B, B),
                     ("fetch", M), ("critique", A, M), ("critique", B, M)]
    assert events.count(("serve", M)) == 1  # one critic start reads both arms


def test_a_reclaim_after_the_first_arms_write_costs_only_that_arms_setup(monkeypatch, tmp_path):
    A, B, M = "qwen3.6-35b-a3b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    events, plan, pool = _fake_window(monkeypatch, tmp_path, die_after_write_of=A)
    with pytest.raises(_Reclaimed):
        pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    assert [e for e in events if e[0] == "fetch"] == [("fetch", A)]  # B and the critic were never set up


def test_two_arms_critique_each_other_with_each_model_fetched_once(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    L, M = "nemotron-3.5-lightning-30b-a3b", "muse-glimmer-30b"
    pw.pilot_batch({}, [], plan, tmp_path / "pilot", [L, M], {L: M, M: L}, pool)
    assert [e for e in events if e[0] == "fetch"] == [("fetch", L), ("fetch", M)]
    assert [e for e in events if e[0] == "critique"] == [("critique", L, M), ("critique", M, L)]


def test_critics_for_takes_the_default_and_the_per_arm_settings(monkeypatch):
    monkeypatch.setenv("PIN_PILOT_CRITIC", "muse-glimmer-30b")
    monkeypatch.setenv("PIN_PILOT_CRITIC_FOR", "muse-glimmer-30b=nemotron-3.5-lightning-30b-a3b")
    got = pw.critics_for(["nemotron-3.5-lightning-30b-a3b", "muse-glimmer-30b"])
    assert got == {"nemotron-3.5-lightning-30b-a3b": "muse-glimmer-30b", "muse-glimmer-30b": "nemotron-3.5-lightning-30b-a3b"}
    monkeypatch.setenv("PIN_PILOT_CRITIC_FOR", "muse-glimmer-30b=muse-glimmer-30b")
    with pytest.raises(RuntimeError, match="critique itself"):
        pw.critics_for(["muse-glimmer-30b"])
    monkeypatch.delenv("PIN_PILOT_CRITIC")
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR")
    with pytest.raises(RuntimeError, match="no critic"):
        pw.critics_for(["granite-4.2-30b"])


def test_critics_for_refuses_a_same_family_pair_before_any_model_loads(monkeypatch):
    monkeypatch.setenv("PIN_PILOT_CRITIC", "nemotron-3.5-lightning-30b-a3b")
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR", raising=False)
    families = {"nemotron-3.5-bolt-74b-a9b": "nemotron", "nemotron-3.5-lightning-30b-a3b": "nemotron", "granite-4.2-30b": "granite"}
    with pytest.raises(RuntimeError, match="same family: nemotron-3.5-bolt-74b-a9b<-nemotron-3.5-lightning-30b-a3b"):
        pw.critics_for(["granite-4.2-30b", "nemotron-3.5-bolt-74b-a9b"], families)
    assert pw.critics_for(["granite-4.2-30b"], families) == {"granite-4.2-30b": "nemotron-3.5-lightning-30b-a3b"}


def test_the_twins_switch(monkeypatch):
    monkeypatch.delenv("PIN_PILOT_TWINS", raising=False)
    assert pw.twins_enabled()
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    assert not pw.twins_enabled()


def test_every_window_model_has_a_pin_that_matches_writers_json():
    import json

    listed = json.loads((Path(__file__).resolve().parents[1] / "writers.json").read_text(encoding="utf-8"))
    pw.check_pins([m["served_as"] for m in pw.MODELS], listed, {})


def test_check_pins_refuses_a_missing_or_different_pin_before_any_download():
    listed = {"writers": [{"served_as": "qwen3.8-27b", "gguf_sha256": "0" * 64}], "critics": []}
    with pytest.raises(RuntimeError, match="nothing downloaded: qwen3.8-27b"):
        pw.check_pins(["qwen3.8-27b"], listed, {})
    listed = {"writers": [{"served_as": "deepseek-r1-distill-qwen-32b",
                           "gguf_sha256": next(m["expect_sha256"] for m in pw.MODELS if m["served_as"] == "deepseek-r1-distill-qwen-32b")}],
              "critics": []}
    with pytest.raises(RuntimeError, match="deepseek-r1-distill-qwen-32b-q6k: MODELS None vs writers.json None"):
        pw.check_pins(["deepseek-r1-distill-qwen-32b"], listed, {"deepseek-r1-distill-qwen-32b": "q6k"})


def test_a_resume_tar_unpacks_and_every_drafts_file_needs_its_seal(tmp_path, monkeypatch):
    import tarfile

    src = tmp_path / "src" / "pilot-w0"
    src.mkdir(parents=True)
    (src / "granite-4.2-30b.jsonl").write_text("{}\n", encoding="utf-8")
    (src / "granite-4.2-30b.jsonl.sha256").write_text("x  granite-4.2-30b.jsonl\n", encoding="utf-8")
    (src / "calls").mkdir()
    (src / "calls" / "granite-4.2-30b.jsonl").write_text("{}\n", encoding="utf-8")  # same name, its own directory
    tar = tmp_path / "resume.tar"
    with tarfile.open(tar, "w") as tf:
        tf.add(src, arcname="pilot-w0")
    dest = pw.unpack_resume(tar)
    monkeypatch.setenv("PIN_RESUME_WAVE", "0")
    monkeypatch.setenv("PIN_RESUME_DIR", str(dest / "pilot-w0"))
    start, found = pw.resume_from()
    assert start == 1 and list(found) == ["granite-4.2-30b"]
    (dest / "pilot-w0" / "granite-4.2-30b.jsonl.sha256").unlink()
    with pytest.raises(RuntimeError, match="no seal for granite-4.2-30b"):
        pw.resume_from()


def test_to_q6k_unlinks_q8_0_before_bf16_conversion_and_logs_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "WORK", tmp_path)
    monkeypatch.setattr(pw, "RESOURCES", [])
    q8 = tmp_path / "model-Q8_0.gguf"
    q8.write_bytes(b"q8")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    bf16 = tmp_path / "deepseek-r1-distill-qwen-32b-BF16.gguf"
    q6 = tmp_path / "deepseek-r1-distill-qwen-32b-Q6_K.gguf"
    q8_during_bf16 = []

    def fake_sh(cmd: str, **kw) -> str:
        if "--outtype bf16" in cmd:
            q8_during_bf16.append(q8.exists())
            bf16.write_bytes(b"bf16")
        elif "Q6_K" in cmd:
            assert not snap.exists()  # snapshot dropped before Q6_K quantize
            q6.write_bytes(b"q6k")
        return "ok"

    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setattr(pw, "build_quantize", lambda src: {"path": "/bin/quantize", "sha256": "0" * 64})
    spec = {"served_as": "deepseek-r1-distill-qwen-32b", "expect_q6k_sha256": pw.sha256(q6) if q6.exists() else None}
    m = {"served_as": "deepseek-r1-distill-qwen-32b", "path": str(q8), "snapshot": str(snap),
         "converter_py": sys.executable, "gguf_sha256": "a" * 64}
    out = pw.to_q6k({"src": str(tmp_path)}, spec, m, {})
    assert q8_during_bf16 == [False]  # Q8_0 was already unlinked before BF16 conversion started
    assert not q8.exists() and not bf16.exists() and not snap.exists() and q6.exists()
    assert out["served_as"] == "deepseek-r1-distill-qwen-32b-q6k"
    steps = [r["at"] for r in pw.RESOURCES]
    assert steps == [
        "deepseek-r1-distill-qwen-32b: after unlinking Q8_0",
        "deepseek-r1-distill-qwen-32b: after BF16 conversion",
        "deepseek-r1-distill-qwen-32b: after dropping snapshot",
        "deepseek-r1-distill-qwen-32b: after Q6_K",
    ]


def test_pool_deletes_writer_gguf_when_no_longer_needed_and_keeps_critic_until_last_critique(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "RESOURCES", [])
    A, B = "qwen3.6-35b-a3b", "nemotron-3.5-bolt-74b-a9b"
    L, M = "nemotron-3.5-lightning-30b-a3b", "muse-glimmer-30b"
    a_existed_when_b_fetched = []

    def fetch_model(spec, built, keep_snapshot=False):
        name = spec["served_as"]
        if name == B:
            a_existed_when_b_fetched.append((tmp_path / f"{A}.gguf").exists())
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(b"gguf")
        return {"served_as": name, "gguf_sha256": spec.get("expect_sha256") or "x" * 64, "path": str(p)}

    class Served:
        def __init__(self, built, m, pinned):
            self.m, self.starts = m, []

        def __enter__(self):
            assert Path(self.m["path"]).exists()
            return self

        def __exit__(self, *exc):
            pass

    def write_wave(rs, common, wave, out, *, writer=None, critic=None, **kw):
        out.mkdir(parents=True, exist_ok=True)
        (out / "wave-drafts.jsonl").write_text('{"type": "header"}\n', encoding="utf-8")
        return {"exit": 0, "result": {"dropped": 0, "states": {"awaiting_critique": 1 if wave == "write" else 0, "needs_rewrite": 0}}}

    monkeypatch.setattr(pw, "fetch_model", fetch_model)
    monkeypatch.setattr(pw, "Served", Served)
    monkeypatch.setattr(pw, "choose_quant", lambda built, spec, m, policy, result: (m, {"policy": policy}))
    monkeypatch.setattr(pw, "j1_card", lambda m, cards, out: (tmp_path / "c.json", m["served_as"], {}))
    monkeypatch.setattr(pw, "write_wave", write_wave)
    monkeypatch.setattr(pw, "run_dir", lambda out: out)
    monkeypatch.setattr(pw, "checkpoint", lambda *a: None)
    monkeypatch.setattr(pw, "PILOT_WAVES", 2)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)
    monkeypatch.delenv("PIN_STUB", raising=False)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"slots": [{"id": f"s{i}"} for i in range(8)]}), encoding="utf-8")

    # 1. Pure writers A and B are deleted right after w0; critic M is deleted after its last critique.
    pool = pw.Pool({}, {}, {}, tmp_path / "cards.json", tmp_path / "out")
    pw.pilot_batch({}, [], plan, tmp_path / "pilot1", [A, B], {A: M, B: M}, pool)
    assert a_existed_when_b_fetched == [False]  # A's GGUF was already deleted before B was fetched
    assert not (tmp_path / f"{A}.gguf").exists()
    assert not (tmp_path / f"{B}.gguf").exists()
    assert not (tmp_path / f"{M}.gguf").exists()
    assert [r["at"] for r in pw.RESOURCES] == [
        f"{A}: after deleting GGUF",
        f"{B}: after deleting GGUF",
        f"{M}: after deleting GGUF",
    ]

    # 2. L and M critiquing each other both stay through w0 and are deleted only after their own critique in w1.
    pw.RESOURCES.clear()
    seen_in_critique = []

    def write_wave_mutual(rs, common, wave, out, *, writer=None, critic=None, **kw):
        if wave == "critique":
            seen_in_critique.append((critic["served_as"], (tmp_path / f"{L}.gguf").exists(), (tmp_path / f"{M}.gguf").exists()))
        return write_wave(rs, common, wave, out, writer=writer, critic=critic, **kw)

    monkeypatch.setattr(pw, "write_wave", write_wave_mutual)
    pool2 = pw.Pool({}, {}, {}, tmp_path / "cards.json", tmp_path / "out2")
    pw.pilot_batch({}, [], plan, tmp_path / "pilot2", [L, M], {L: M, M: L}, pool2)
    # When M critiques L first in w1, both L.gguf and M.gguf still exist; when L critiques M second, M.gguf is already gone.
    assert seen_in_critique == [(M, True, True), (L, True, False)]
    assert not (tmp_path / f"{L}.gguf").exists() and not (tmp_path / f"{M}.gguf").exists()
    assert [r["at"] for r in pw.RESOURCES] == [f"{M}: after deleting GGUF", f"{L}: after deleting GGUF"]


def test_run_proc_writes_heartbeat_while_subprocess_lives_and_stops_when_it_exits(monkeypatch, capsys):
    import time

    monkeypatch.setattr(pw, "HEARTBEAT_S", 0.15)
    res = pw.run_proc([sys.executable, "-c", "import time; time.sleep(0.45)"], timeout=30)
    assert res.returncode == 0
    out1 = capsys.readouterr().out
    beats1 = [ln for ln in out1.splitlines() if "HEARTBEAT pid=" in ln]
    assert len(beats1) >= 1, out1
    assert "disk_free_gb=" in beats1[0] and "ram_free_mb=" in beats1[0]
    time.sleep(0.35)
    out2 = capsys.readouterr().out
    assert "HEARTBEAT pid=" not in out2  # tied to the subprocess, never a free-running thread


def test_checkpoints_and_resume_from_refuse_stub_drafts_without_pin_stub(tmp_path, monkeypatch):
    import base64
    import gzip

    import checkpoints as cp

    raw = b'{"type": "header", "stub": true, "writer_host": {"machine": {"stub": true}}}\n{"slot_id": "T02-A-01"}\n'
    sha = pw.sha256_bytes(raw) if hasattr(pw, "sha256_bytes") else __import__("hashlib").sha256(raw).hexdigest()
    blob = base64.b64encode(gzip.compress(raw)).decode()
    log = tmp_path / "job.log"
    log.write_text(f"WAVE_CHECKPOINT granite-4.2-30b pilot-w0 {sha} {blob}\n", encoding="utf-8")

    # 1. checkpoints.py extract refuses stub drafts unless PIN_STUB=1
    monkeypatch.delenv("PIN_STUB", raising=False)
    with pytest.raises(RuntimeError, match="STUB checkpoint"):
        cp.extract(log, tmp_path / "ext-refused")
    monkeypatch.setenv("PIN_STUB", "1")
    got = cp.extract(log, tmp_path / "ext-ok")
    assert len(got) == 2

    # 2. resume_from refuses stub drafts unless PIN_STUB=1
    monkeypatch.setenv("PIN_RESUME_WAVE", "0")
    monkeypatch.setenv("PIN_RESUME_DIR", str(tmp_path / "ext-ok" / "pilot-w0"))
    monkeypatch.delenv("PIN_STUB", raising=False)
    with pytest.raises(RuntimeError, match="STUB drafts"):
        pw.resume_from()
    monkeypatch.setenv("PIN_STUB", "1")
    start, found = pw.resume_from()
    assert start == 1 and list(found) == ["granite-4.2-30b"]


@pytest.mark.skipif(os.name == "nt", reason="runs pin_window.py in stub mode with Linux venv/flock")
def test_pin_window_stub_mode_end_to_end(tmp_path, monkeypatch):
    import tarfile

    import checkpoints as cp
    from test_cards import SAMPLING

    from role_set.pack import PackManifest, pack_hash

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()

    # Build a valid role pack tar in content/pack.tar covering T02 (pilot) and T04 (twins).
    pack_src = tmp_path / "pack_src"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
        "traits/T04.md": "## 4. As a verifier, you check your work\n\nCheck.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for name, text in trait_files.items():
        p = pack_src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(trait_files)
    (pack_src / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    with tarfile.open(content / "pack.tar", "w") as tf:
        for item in pack_src.iterdir():
            tf.add(item, arcname=item.name)

    writers = ["granite-4.2-30b", "nemotron-3.5-lightning-30b-a3b"]
    critic = "muse-glimmer-30b"
    cards = {"candidates": [{"candidate": c, "revision": "r1", "sampling": dict(SAMPLING),
                             "chat_template_kwargs": {"enable_thinking": True},
                             "transforms": ["reattach_thinking"]} for c in [*writers, critic]]}
    (content / "card-values.json").write_text(json.dumps(cards), encoding="utf-8")

    committed_pool = Path(__file__).resolve().parents[1] / "roles" / "verifier" / "pool.jsonl"
    shutil.copy(committed_pool, content / "pool.jsonl")
    pool_sha = pw.sha256(content / "pool.jsonl")

    slots_pilot = "T02-A-01,T02-C-01,T02-D-01,T02-E-01,T02-B-01,T02-A-02,T02-C-08,T02-D-05"
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_CONTENT", str(content))
    monkeypatch.setenv("PIN_WORK", str(work))
    monkeypatch.setenv("PIN_POOL_SHA256", pool_sha)
    # The commit under test, not a pinned old one: an old commit requires Python <3.13 and the window refuses it.
    # content/role-set is a local clone at HEAD, so setup needs no network clone of role-set.
    here = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "clone", "-q", str(here), str(content / "role-set")], check=True)
    subprocess.run(["git", "-C", str(content / "role-set"), "checkout", "-q", head], check=True)
    monkeypatch.setenv("PIN_ROLESET_COMMIT", head)
    monkeypatch.setenv("PIN_RND_COMMIT", "04dccfad9419cd79c4068703ac6cf1442850c9de")
    monkeypatch.setenv("PIN_PILOT_WRITERS", ",".join(writers))
    monkeypatch.setenv("PIN_PILOT_CRITIC", critic)
    monkeypatch.setenv("PIN_PILOT_WAVES", "2")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", slots_pilot)
    monkeypatch.delenv("PIN_PILOT_TWIN_IDS", raising=False)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_DIR", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)

    rc = pw.main()
    assert rc == 0, (work / "pin-result.json").read_text(encoding="utf-8")
    assert (work / "pilot-out.tar").exists()
    assert (work / "pin-result.json").exists()

    res = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is True and res["stub"] is True and res["pool_sha256"] == pool_sha
    job_log = (work / "job.log").read_text(encoding="utf-8")
    assert "STUB" in job_log.splitlines()[0]

    for arm in writers:
        for part in ("pilot", "twins"):
            assert "STUB" in res["arms"][arm][part]["readout_tail"]

    with tarfile.open(work / "pilot-out.tar") as tf:
        names = set(tf.getnames())
    for arm in writers:
        assert f"pilot/pilot/{arm}/w0/wave-drafts.jsonl" in names
        assert f"pilot/pilot/{arm}/w1/wave-drafts.jsonl" in names

    # Checkpoints extract with PIN_STUB=1 to sealed drafts that resume_from accepts (and refuse without PIN_STUB=1).
    ext_dir = tmp_path / "extracted"
    monkeypatch.delenv("PIN_STUB", raising=False)
    with pytest.raises(RuntimeError, match="STUB checkpoint"):
        cp.extract(work / "job.log", ext_dir)
    monkeypatch.setenv("PIN_STUB", "1")
    summary = cp.extract(work / "job.log", ext_dir)
    assert len(summary) >= 4
    monkeypatch.setenv("PIN_RESUME_WAVE", "1")
    monkeypatch.setenv("PIN_RESUME_DIR", str(ext_dir / "pilot-w1"))
    monkeypatch.delenv("PIN_STUB", raising=False)
    with pytest.raises(RuntimeError, match="STUB drafts"):
        pw.resume_from()
    monkeypatch.setenv("PIN_STUB", "1")
    start, found = pw.resume_from()
    assert start == 2 and set(found) == set(writers)


def test_a_dropped_model_refuses_a_later_request(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    m = pool.get("granite-4.2-30b")
    m["path"] = str(tmp_path / "granite.gguf")
    (tmp_path / "granite.gguf").write_bytes(b"x")
    monkeypatch.setattr(pw, "resources", lambda *a, **k: None)
    pool.drop("granite-4.2-30b")
    assert not (tmp_path / "granite.gguf").exists()
    with pytest.raises(RuntimeError, match="deleted after its last planned use"):
        pool.serve("granite-4.2-30b")


def test_the_repo_root_without___file__(monkeypatch):
    # On the VM, colab exec runs the script in the kernel, where __file__ isn't defined.
    assert pw._repo_root() == Path(pw.__file__).resolve().parents[2]
    monkeypatch.delitem(pw.__dict__, "__file__")
    assert pw._repo_root() == pw.CONTENT / "role-set"


# ---- Browser Argon's v2 re-review (2026-10-17)

LINUX = pytest.mark.skipif(os.name == "nt", reason="process groups and flock are Linux, where the window runs")


@LINUX
def test_run_proc_timeout_kills_the_shells_children_too():
    import time as _t

    t0 = _t.time()
    with pytest.raises(subprocess.TimeoutExpired):
        pw.run_proc("sleep 30 & sleep 30; wait", shell=True, timeout=1)
    assert _t.time() - t0 < 20  # the old kill left the children holding the pipes, and the wait never returned


@LINUX
def test_the_job_lock_stops_on_this_windows_result_only(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_JOB_TOKEN", "pinw-1")
    (tmp_path / "pin-result.json").write_text('{"job_token": "pinw-1"}', encoding="utf-8")
    assert pw.take_job_lock(tmp_path) is None  # this window's result is written: a later exec stops
    assert not (tmp_path / "job.lock").exists()  # before any lock file is opened
    (tmp_path / "pin-result.json").write_text('{"job_token": "another-window"}', encoding="utf-8")
    fh = pw.take_job_lock(tmp_path)
    assert fh is not None  # a stale result from another run never makes a real job skip itself
    fh.close()


def _second_exec(work):
    """take_job_lock in a daemon thread, as a second `colab exec -f` runs in the same kernel process."""
    import threading

    got = {}
    t = threading.Thread(target=lambda: got.__setitem__("lock", pw.take_job_lock(work)), daemon=True)
    t.start()
    return t, got


@LINUX
def test_a_second_exec_waits_for_the_first_then_stops_on_this_windows_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_JOB_TOKEN", "pinw-2")
    first = pw.take_job_lock(tmp_path)
    assert first is not None
    try:
        t, got = _second_exec(tmp_path)
        t.join(1.0)
        assert t.is_alive()  # the second waits while the first job runs
        assert "JOB ALREADY RUNNING: this exec waits" in (tmp_path / "job.log").read_text(encoding="utf-8")
        (tmp_path / "pin-result.json").write_text('{"job_token": "pinw-2"}', encoding="utf-8")
    finally:
        first.close()  # always released, so the waiter ends even when an assert fails
    t.join(10)
    assert not t.is_alive() and got["lock"] is None  # the first finished this window: the second stops
    assert "the first job finished this window" in (tmp_path / "job.log").read_text(encoding="utf-8")


@LINUX
def test_a_second_exec_runs_the_job_when_the_first_ended_with_no_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_JOB_TOKEN", "pinw-3")
    first = pw.take_job_lock(tmp_path)
    assert first is not None
    try:
        t, got = _second_exec(tmp_path)
        t.join(1.0)
        assert t.is_alive()
        assert "JOB ALREADY RUNNING: this exec waits" in (tmp_path / "job.log").read_text(encoding="utf-8")
    finally:
        first.close()  # the first died without a result for this window
    t.join(10)
    assert not t.is_alive() and got["lock"] is not None  # so the waiting exec runs the job
    got["lock"].close()
    assert "ended with no result" in (tmp_path / "job.log").read_text(encoding="utf-8")


def test_twins_skip_an_arm_with_a_failed_wave(tmp_path, monkeypatch):
    monkeypatch.delenv("PIN_PILOT_TWINS", raising=False)
    (tmp_path / "accepted.jsonl").write_text("{}\n", encoding="utf-8")
    ok = {"waves": [{"exit": 0}, {"exit": 0}], "last": str(tmp_path)}
    assert pw.twins_skip_reason(ok) is None
    assert pw.twins_skip_reason({**ok, "waves": [{"exit": 0}, {"exit": 2}]}) == "a pilot wave exited non-zero"
    assert "funnel" in pw.twins_skip_reason({**ok, "funnel_stop": "3 of 8"})


def test_a_resumed_batch_runs_only_the_arms_it_has_drafts_for(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    A, B, M = "granite-4.2-30b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    d = tmp_path / "resume"
    d.mkdir()
    (d / f"{A}.jsonl").write_text('{"type": "header"}\n{"type": "draft", "state": "awaiting_critique"}\n', encoding="utf-8")
    (d / f"{A}.jsonl.sha256").write_text(f"x  {A}.jsonl\n", encoding="utf-8")
    monkeypatch.setenv("PIN_RESUME_WAVE", "0")
    monkeypatch.setenv("PIN_RESUME_DIR", str(d))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    assert [e for e in events if e[0] == "critique"] == [("critique", A, M)]  # B has nothing to critique
    assert "no drafts in PIN_RESUME_DIR" in arms[B]["not_resumed"]


def test_check_pins_compares_the_q6k_pin(monkeypatch):
    listed = {"writers": [{"served_as": "deepseek-r1-distill-qwen-32b",
                           "gguf_sha256": next(m["expect_sha256"] for m in pw.MODELS if m["served_as"] == "deepseek-r1-distill-qwen-32b")},
                          {"served_as": "deepseek-r1-distill-qwen-32b-q6k", "gguf_sha256": "6" * 64}], "critics": []}
    with pytest.raises(RuntimeError, match="deepseek-r1-distill-qwen-32b-q6k: MODELS None"):
        pw.check_pins(["deepseek-r1-distill-qwen-32b"], listed, {"deepseek-r1-distill-qwen-32b": "q6k"})
    r1 = next(m for m in pw.MODELS if m["served_as"] == "deepseek-r1-distill-qwen-32b")
    monkeypatch.setitem(r1, "expect_q6k_sha256", "7" * 64)
    with pytest.raises(RuntimeError, match="MODELS 777777777777 vs writers.json 666666666666"):
        pw.check_pins(["deepseek-r1-distill-qwen-32b"], listed, {"deepseek-r1-distill-qwen-32b": "q6k"})
    monkeypatch.setitem(r1, "expect_q6k_sha256", "6" * 64)
    pw.check_pins(["deepseek-r1-distill-qwen-32b"], listed, {"deepseek-r1-distill-qwen-32b": "q6k"})


# ---- a resumed arm whose batch already finished (R&D, 2026-10-17): 12f's real w1 files for Qwen3.8 and Gemma

FIX12F = Path(__file__).resolve().parent / "fixtures" / "window12f-w1.tar.gz"
BATCH_FILES = ("accepted.jsonl", "receipts.jsonl", "accepted.models.jsonl")


def _resume_dir_12f(tmp_path) -> tuple[Path, Path]:
    import tarfile

    src = tmp_path / "w12f"
    with tarfile.open(FIX12F) as tf:
        tf.extractall(src, filter="data")
    d = tmp_path / "resume"
    d.mkdir()
    for arm in ("qwen3.8-27b", "gemma-4-31b-it"):  # as checkpoints.py extract lays them out: <arm>.jsonl + its seal
        (d / f"{arm}.jsonl").write_bytes((src / arm / "wave-drafts.jsonl").read_bytes())
        (d / f"{arm}.jsonl.sha256").write_bytes((src / arm / "wave-drafts.jsonl.sha256").read_bytes())
    return src, d


def test_a_resumed_arm_with_nothing_pending_runs_no_wave_and_gets_its_twins(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    monkeypatch.setattr(pw, "PILOT_WAVES", 4)
    monkeypatch.delenv("PIN_PILOT_TWINS", raising=False)
    src, d = _resume_dir_12f(tmp_path)
    Q, G, M = "qwen3.8-27b", "gemma-4-31b-it", "muse-glimmer-30b"
    monkeypatch.setenv("PIN_RESUME_WAVE", "1")
    monkeypatch.setenv("PIN_RESUME_DIR", str(d))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [Q, G], {Q: M, G: M}, pool, twins_after=True)
    # No refused rewrite, and neither writer (nor the critic) is set up for this batch.
    assert events == [], events
    for arm in (Q, G):
        a = arms[arm]
        assert a["done"] and a["resumed_finished"] and a["waves"] == []
        for name in BATCH_FILES:  # rebuilt byte for byte as the window that finished the batch wrote them
            assert (Path(a["last"]) / name).read_bytes() == (src / arm / name).read_bytes(), (arm, name)
        assert not (Path(a["last"]) / "card-pins-chain.json").exists()  # a new header keeps its card: no chain
        assert pw.twins_skip_reason(a) is None


def test_a_resumed_arm_with_a_draft_sent_back_runs_the_rewrite(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    monkeypatch.setattr(pw, "PILOT_WAVES", 4)
    src, d = _resume_dir_12f(tmp_path)
    Q, G, M = "qwen3.8-27b", "gemma-4-31b-it", "muse-glimmer-30b"
    lines = (d / f"{Q}.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["state"] = "needs_rewrite"
    lines[1] = json.dumps(row, ensure_ascii=False)
    (d / f"{Q}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("PIN_RESUME_WAVE", "1")
    monkeypatch.setenv("PIN_RESUME_DIR", str(d))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [Q, G], {Q: M, G: M}, pool)
    assert ("rewrite", Q, Q) in events and not [e for e in events if G in e], events
    assert not arms[Q].get("resumed_finished") and arms[G]["resumed_finished"]


# ---- the card-pin chain (stub rung 12s5a; R&D and Argon, 2026-10-18): 12g's and 12h's real w0 and w1

FIXCHAIN = Path(__file__).resolve().parent / "fixtures" / "window12g12h-chain.tar.gz"


def _chain_dirs(tmp_path) -> tuple[Path, Path]:
    import tarfile

    with tarfile.open(FIXCHAIN) as tf:
        tf.extractall(tmp_path / "chain", filter="data")
    return tmp_path / "chain" / "w1", tmp_path / "chain" / "w0"


def test_a_finished_old_header_arm_gets_its_card_pins_beside_byte_identical_batch_files(tmp_path):
    from role_set import waves

    w1, w0 = _chain_dirs(tmp_path)
    B = "nemotron-3.5-bolt-74b-a9b"
    plain = tmp_path / "plain"
    plain.mkdir()
    waves.write_batch(waves.read(w1 / f"{B}.jsonl")[1], plain)  # what the window's own finishing wave wrote
    out = pw.rebuild_batch({}, w1 / f"{B}.jsonl", tmp_path / "rebuilt", w0 / f"{B}.jsonl")
    for name in BATCH_FILES:
        assert (out / name).read_bytes() == (plain / name).read_bytes(), name
    side = json.loads((out / "card-pins-chain.json").read_text(encoding="utf-8"))
    assert side["via"] == "chain" and set(side["pins"]) == waves.CARD_KEYS
    assert side["prior_sha256"] == waves.file_sha256(w0 / f"{B}.jsonl")
    # Without its prior, or with another arm's, the sidecar records why, and the rebuild still stands.
    out = pw.rebuild_batch({}, w1 / f"{B}.jsonl", tmp_path / "noprior")
    assert json.loads((out / "card-pins-chain.json").read_text(encoding="utf-8"))["refused"] == "no prior drafts for this arm"
    out = pw.rebuild_batch({}, w1 / f"{B}.jsonl", tmp_path / "wrong", w0 / "granite-4.2-30b.jsonl")
    assert "is not the input" in json.loads((out / "card-pins-chain.json").read_text(encoding="utf-8"))["refused"]
    assert (out / "accepted.jsonl").read_bytes() == (plain / "accepted.jsonl").read_bytes()


def test_the_first_rewrite_after_a_resume_gets_its_prior_and_nothing_else_does(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    monkeypatch.setattr(pw, "PILOT_WAVES", 4)
    seen = []
    fake = pw.write_wave

    def write_wave(rs, common, wave, out, **kw):
        seen.append((wave, out.parent.name, out.name, list(kw.get("extra") or [])))
        return fake(rs, common, wave, out, **kw)

    monkeypatch.setattr(pw, "write_wave", write_wave)
    monkeypatch.setattr(pw, "rebuild_batch", lambda rs, f, out, prior=None: out)
    w1, w0 = _chain_dirs(tmp_path)
    G, B, M = "granite-4.2-30b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    monkeypatch.setenv("PIN_RESUME_WAVE", "1")
    monkeypatch.setenv("PIN_RESUME_DIR", str(w1))
    monkeypatch.setenv("PIN_RESUME_PRIOR_DIR", str(w0))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [G, B], {G: M, B: M}, pool)
    assert ("rewrite", G, "w2", ["--prior-drafts", str(w0 / f"{G}.jsonl")]) in seen, seen
    assert all(not extra for wave, arm, name, extra in seen if (wave, arm, name) != ("rewrite", G, "w2")), seen
    assert arms[B]["resumed_finished"] and not [e for e in seen if e[1] == B]


# ---- an arm whose write or rewrite leaves nothing to critique ends there (R&D on window 13b: Lightning's twin write
# dropped both slots, and the critique pin_window then ran was refused)

def test_a_write_that_drops_every_slot_runs_no_critique(monkeypatch, tmp_path):
    A, B, M = "granite-4.2-30b", "nemotron-3.5-bolt-74b-a9b", "muse-glimmer-30b"
    events, plan, pool = _fake_window(monkeypatch, tmp_path, all_dropped=(A,))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [A, B], {A: M, B: M}, pool)
    assert ("critique", A, M) not in events and ("critique", B, M) in events, events
    assert arms[A]["done"] and not any(w["exit"] for w in arms[A]["waves"])  # ended, not failed
    assert len(arms[A]["waves"]) == 1  # its write only


def test_a_rewrite_that_drops_every_slot_runs_no_critique(monkeypatch, tmp_path):
    Q, G, M = "qwen3.8-27b", "gemma-4-31b-it", "muse-glimmer-30b"
    events, plan, pool = _fake_window(monkeypatch, tmp_path, all_dropped=(Q,))
    monkeypatch.setattr(pw, "PILOT_WAVES", 4)
    src, d = _resume_dir_12f(tmp_path)
    lines = (d / f"{Q}.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["state"] = "needs_rewrite"
    lines[1] = json.dumps(row, ensure_ascii=False)
    (d / f"{Q}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("PIN_RESUME_WAVE", "1")
    monkeypatch.setenv("PIN_RESUME_DIR", str(d))
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot", [Q, G], {Q: M, G: M}, pool)
    assert ("rewrite", Q, Q) in events and ("critique", Q, M) not in events, events
    assert arms[Q]["done"] and len(arms[Q]["waves"]) == 1  # its rewrite only


# ---- the window's deadline (R&D, 2026-10-18, from Argon's golden-path reply): a cap stop returns results ----------------

def _deadline(monkeypatch, seconds_from_now: float):
    import time as _t

    monkeypatch.setenv("PIN_DEADLINE", str(int(_t.time() + seconds_from_now)))
    monkeypatch.setattr(pw, "CUT_AT", None)


def test_no_wave_starts_past_the_deadline(tmp_path, monkeypatch):
    _deadline(monkeypatch, -1)
    rs = {"role_set": str(tmp_path / "never-run"), "src": str(tmp_path)}
    with pytest.raises(pw.DeadlineReached, match="before the write wave"):
        pw.write_wave(rs, [], "write", tmp_path / "arm" / "w0")
    assert pw.CUT_AT is None  # nothing was under way to cut


def test_no_model_setup_or_hosted_download_starts_past_the_deadline(tmp_path, monkeypatch):
    _deadline(monkeypatch, -1)
    spec = next(m for m in pw.MODELS if m["served_as"] == "granite-4.2-30b")
    with pytest.raises(pw.DeadlineReached, match="granite-4.2-30b's setup"):
        pw.fetch_model(spec, {})


def test_a_deadline_is_never_swallowed_as_a_runtime_error():
    assert not issubclass(pw.DeadlineReached, RuntimeError)  # choose_quant falls back on RuntimeError


def test_without_a_deadline_nothing_changes(monkeypatch):
    monkeypatch.delenv("PIN_DEADLINE", raising=False)
    assert pw.deadline_left() is None
    pw.check_deadline("anything")  # no raise


@LINUX
def test_a_wave_under_way_at_the_deadline_is_cut_with_its_finished_calls_kept(tmp_path, monkeypatch):
    import time as _t

    monkeypatch.setattr(pw, "say", lambda *a: None)
    fake = tmp_path / "role-set"
    # A wave: one finished call journaled, then a long call in flight.
    fake.write_text('#!/usr/bin/env bash\nfor a in "$@"; do [ "$prev" = --call-journal ] && j=$a; prev=$a; done\n'
                    'echo \'{"call": 1}\' >> "$j"\nsleep 60\n', encoding="utf-8")
    fake.chmod(0o755)
    rs = {"role_set": str(fake), "src": str(tmp_path)}
    out = tmp_path / "arm" / "w2"
    out.parent.mkdir(parents=True)
    _deadline(monkeypatch, 2)
    t0 = _t.time()
    with pytest.raises(pw.DeadlineReached, match="cut at the window's deadline"):
        pw.write_wave(rs, [], "rewrite", out)
    assert _t.time() - t0 < 20  # SIGTERM ended it; no wait for the call in flight
    assert pw.CUT_AT["wave"] == "rewrite" and pw.CUT_AT["arm"] == "arm"
    journal = Path(pw.CUT_AT["journal"])
    assert journal.read_text(encoding="utf-8").strip() == '{"call": 1}'  # the finished call is kept for the resume


def test_the_result_says_the_window_was_cut_at_its_deadline(tmp_path, monkeypatch):
    import json as _json

    monkeypatch.setattr(pw, "WORK", tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "machine_line", lambda **k: {"card": "stub"})
    monkeypatch.setattr(pw, "read_token_file", lambda **k: None)
    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_STUB", "1")
    cut = {"wave": "critique", "arm": "granite-4.2-30b", "label": "pilot-w3", "seconds": 61}

    def pilot_window(built, result):
        monkeypatch.setattr(pw, "CUT_AT", cut)
        raise pw.DeadlineReached("the critique wave at w3 was cut at the window's deadline")

    monkeypatch.setattr(pw, "pilot_window", pilot_window)
    pw._main()
    r = _json.loads((tmp_path / "pin-result.json").read_text(encoding="utf-8"))
    assert r["ok"] is False and r["error"].startswith("DeadlineReached") and r["cut_at_deadline"] == cut


def test_a_deadline_in_a_twin_wave_leaves_the_batch_and_is_not_recorded_as_an_arm_error(monkeypatch, tmp_path):
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    A, M = "qwen3.8-27b", "muse-glimmer-30b"
    fake = pw.write_wave

    def write_wave(rs, common, wave, out, **kw):
        if "twins" in str(out):
            raise pw.DeadlineReached("the write wave was cut at the window's deadline")
        return fake(rs, common, wave, out, **kw)

    monkeypatch.setattr(pw, "write_wave", write_wave)
    with pytest.raises(pw.DeadlineReached):
        pw.pilot_batch({}, [], plan, tmp_path / "twins", [A], {A: M}, pool, extra=["--training", "x"], later_arms=["B"])
    assert not [e for e in events if e[0] == "critique"]  # nothing ran on after the cut


# Every broad handler in pin_window, by the function it sits in, and why a deadline can't be lost in it. A new one fails
# here until it re-raises DeadlineReached or is shown not to wrap a wave, a setup or a download (R&D, #91 review).
BROAD_HANDLERS = {
    "write": "Tee.write: a dead stream; no wave runs inside it",
    "run": "with_heartbeat's worker: every exception, DeadlineReached included, goes to the waiting thread, which raises it",
    "sample": "GPU sampling during a serve; no wave runs inside it",
    "smoke": "the server's startup poll; no wave runs inside it",
    "start_server": "the server's startup poll; no wave runs inside it",
    "critic_window": "re-raises DeadlineReached first",
    "q6k_window": "re-raises DeadlineReached first",
    "pilot_window": "the tar in finally, after the waves",
    "rescore_window": "the tar in finally, after the rescore",
    "critic_eval_window": "the tar in finally, after the critic-eval runs",
    "offrig_bench_window": "the tar in finally, after the calibrate run",
    "_main": "re-raises DeadlineReached first in the smoke role's two; the last records it as cut_at_deadline",
}


def test_no_broad_handler_can_swallow_the_deadline():
    import ast

    tree = ast.parse(Path(pw.__file__).read_text(encoding="utf-8"))
    funcs = [(n.lineno, n.end_lineno, n.name) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    found = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler) and (n.type is None or ast.unparse(n.type) in ("Exception", "BaseException")):
            name = [f for a, b, f in funcs if a <= n.lineno <= b][-1]
            found.setdefault(name, []).append(n.lineno)
    assert set(found) <= set(BROAD_HANDLERS), set(found) - set(BROAD_HANDLERS)
    src = Path(pw.__file__).read_text(encoding="utf-8").splitlines()
    for name in ("critic_window", "q6k_window"):
        for line in found.get(name, []):
            assert "except DeadlineReached:" in src[line - 3], (name, line)
    smoke_role = [ln for ln in found["_main"] if "a smoke window" in src[ln - 1] or "recorded, and the window" in src[ln - 1]]
    assert len(smoke_role) == 2 and all("except DeadlineReached:" in src[ln - 3] for ln in smoke_role), smoke_role


def test_a_stub_call_can_be_slowed_for_a_mid_wave_deadline(monkeypatch):
    import time as _t

    calls = []

    class Inner:
        answers = {"a": 1}

        def __call__(self, msgs, seed=0):
            calls.append(seed)
            return "ok"

    monkeypatch.delenv("PIN_STUB_CALL_S", raising=False)
    inner = Inner()
    assert pw._slow_stub(inner) is inner  # unset: the stub as it was
    monkeypatch.setenv("PIN_STUB_CALL_S", "0.2")
    slow = pw._slow_stub(inner)
    t0 = _t.time()
    assert slow([], seed=3) == "ok" and _t.time() - t0 >= 0.2 and calls == [3]
    assert slow.answers == {"a": 1}  # MockCritic reads the writer's answers through it


def test_the_stub_call_knob_refuses_a_real_run(tmp_path, monkeypatch):
    import json as _json

    monkeypatch.setattr(pw, "WORK", tmp_path)
    monkeypatch.setenv("PIN_STUB_CALL_S", "30")
    monkeypatch.delenv("PIN_STUB", raising=False)
    monkeypatch.setattr(pw, "machine_line", lambda **k: pytest.fail("nothing may run"))
    assert pw._main() == 1
    r = _json.loads((tmp_path / "pin-result.json").read_text(encoding="utf-8"))
    assert r["ok"] is False and "PIN_STUB_CALL_S is set without PIN_STUB=1" in r["error"]



# ---- the progress-gated heartbeat for in-process downloads (R&D, 2026-10-18, after window 13q) ---------------------------

def _hb(monkeypatch, tmp_path):
    lines = []
    monkeypatch.setattr(pw, "heartbeat", lambda msg: lines.append(msg))
    monkeypatch.setenv("PIN_HEARTBEAT_S", "0.2")
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    return lines


def test_a_download_that_grows_heartbeats_with_its_bytes_and_rate(monkeypatch, tmp_path):
    import time as _t

    lines = _hb(monkeypatch, tmp_path)
    d = tmp_path / "dl"
    d.mkdir()

    def download():
        with (d / ".cache-x.incomplete").open("wb") as f:
            for _ in range(6):
                f.write(b"x" * (2 << 20))
                f.flush()
                _t.sleep(0.2)
        return "done"

    assert pw.with_heartbeat(download, "maker:x", 60, watch=d) == "done"
    assert lines and all(x.startswith("HEARTBEAT maker:x elapsed=") and " bytes=" in x and " mbps=" in x for x in lines)


def test_a_download_whose_bytes_stop_growing_writes_no_heartbeat_after_the_stop(monkeypatch, tmp_path):
    import time as _t

    lines = _hb(monkeypatch, tmp_path)
    from types import SimpleNamespace

    monkeypatch.setattr(pw.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(100e9), total=int(200e9), used=0))
    d = tmp_path / "dl"
    d.mkdir()
    stopped = {}

    def download():
        (d / "part.incomplete").write_bytes(b"x" * (4 << 20))
        _t.sleep(1.0)  # several 0.2 s rounds: the growth is seen, even on a loaded runner
        stopped["at"] = len(lines)
        _t.sleep(1.5)  # hung: nothing more arrives
        return "late"

    pw.with_heartbeat(download, "hosted:x", 60, watch=d)
    assert len(lines) == stopped["at"] and len(lines) <= 2, lines  # silence after the stop: the rig's stall can fire


def test_a_download_past_its_hard_timeout_stops_the_window(monkeypatch, tmp_path):
    import time as _t

    _hb(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="not done after 1 s"):
        pw.with_heartbeat(lambda: _t.sleep(30), "snapshot:x", 1, watch=tmp_path)


def test_a_deadline_inside_a_download_reaches_the_job(monkeypatch, tmp_path):
    _hb(monkeypatch, tmp_path)

    def download():
        raise pw.DeadlineReached("cut")

    with pytest.raises(pw.DeadlineReached):
        pw.with_heartbeat(download, "hosted:x", 60, watch=tmp_path)


def test_the_hard_timeout_is_size_over_twenty_mb_per_s_plus_five_minutes():
    assert pw.download_timeout(None) == 3600.0
    assert pw.download_timeout(int(32e9)) == pytest.approx(32e9 / 20e6 + 300)


def test_every_in_process_hub_download_goes_through_the_heartbeat():
    import ast

    tree = ast.parse(Path(pw.__file__).read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) in ("hf_hub_download", "snapshot_download"):
            parents = [p for p in ast.walk(tree) if isinstance(p, ast.Call) and getattr(p.func, "id", None) == "with_heartbeat"
                       and any(c is n for c in ast.walk(p))]
            assert parents, f"line {n.lineno}: {ast.unparse(n.func)} outside with_heartbeat"


def test_a_download_running_at_the_deadline_is_cut_within_one_round(monkeypatch, tmp_path):
    import time as _t

    _hb(monkeypatch, tmp_path)
    monkeypatch.setenv("PIN_HEARTBEAT_S", "5")  # a long round: the deadline, not the heartbeat, ends the wait
    monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 1))
    t0 = _t.time()
    with pytest.raises(pw.DeadlineReached, match="hosted:x was under way at the window's deadline"):
        pw.with_heartbeat(lambda: _t.sleep(30), "hosted:x", 600, watch=tmp_path)
    assert _t.time() - t0 < 4  # cut at the deadline, well inside one 5 s round


# ---- PIN_PILOT_LESSONS and write-only pilot (PIN_PILOT_CRITIC=none) (rnd DESIGN §9, 2026-10-19) -------------------------

def _role_set_cmd() -> dict:
    root = Path(__file__).resolve().parents[1]
    exe = Path(sys.executable).with_name("role-set.exe" if os.name == "nt" else "role-set")
    return {"role_set": str(exe) if exe.exists() else f'"{sys.executable}" -m role_set.cli', "src": str(root)}


def test_plan_pilot_lessons_merges_in_env_order_screens_c_ids_and_role_set_write_accepts(
    tmp_path, monkeypatch, domain_table
):
    from test_waves import pack_dir as _make_pack_dir  # noqa: F401

    from role_set import waves
    from role_set.pack import PackManifest, pack_hash
    from role_set.write import WriteConfig, run

    rs = _role_set_cmd()
    # Two lessons in env order (T08 then T02, not sorted order)
    lessons = ["T08", "T02"]
    slot_ids = "T08-C-01,T08-C-04,T02-C-01,T02-C-06"
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", slot_ids)
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")

    plan_out = tmp_path / "pilot-plan.json"
    meta = pw.plan_pilot_lessons(rs, lessons, str(domain_table), plan_out, tmp_path / "work")

    assert meta["lessons"] == ["T08", "T02"]
    assert list(meta["per_lesson_sha256"]) == ["T08", "T02"]
    assert all(len(v) == 64 for v in meta["per_lesson_sha256"].values())
    assert len(meta["exclusion_state_sha256"] or "") == 64

    unscreened = json.loads(plan_out.read_text(encoding="utf-8"))
    unscreened_lessons = [s["lesson"] for s in unscreened["slots"]]
    # All T08 slots come before all T02 slots (env order preserved)
    first_t02 = unscreened_lessons.index("T02")
    assert all(x == "T08" for x in unscreened_lessons[:first_t02])
    assert all(x == "T02" for x in unscreened_lessons[first_t02:])
    assert len(unscreened["figures"]) == 2
    assert [f["lesson"] for f in unscreened["figures"]] == ["T08", "T02"]

    # Screen keeps C ids across lessons in the merged plan's order (T08 then T02)
    kept = pw.screen_plan(plan_out, "PIN_PILOT_SLOT_IDS")
    assert kept == ["T08-C-01", "T08-C-04", "T02-C-01", "T02-C-06"]
    screened = json.loads(plan_out.read_text(encoding="utf-8"))
    assert [s["id"] for s in screened["slots"]] == ["T08-C-01", "T08-C-04", "T02-C-01", "T02-C-06"]
    assert all(s["kind"] == "C" for s in screened["slots"])

    # Verify role-set write accepts the merged C-only plan as is (without loosening any role-set rule)
    pack = tmp_path / "pack"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
        "traits/T08.md": "## 8. As a verifier, you trace every step\n\nEvery step.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for name, text in trait_files.items():
        p = pack / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(trait_files)
    (pack / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    w0_out = tmp_path / "w0"
    res = run(WriteConfig(plan=plan_out, pack=pack, out=w0_out, mock=True, partition="trial:j1-wave1", wave="write"))
    assert res["halted"] is None
    _, drafts = waves.read(w0_out / "mock" / waves.DRAFTS_FILE)
    assert len(drafts) == 4
    assert {d["slot"]["id"] for d in drafts} == {"T08-C-01", "T08-C-04", "T02-C-01", "T02-C-06"}
    assert all(d["state"] == "awaiting_critique" for d in drafts)


def test_plan_pilot_lessons_refusals(tmp_path, monkeypatch, domain_table):
    rs = _role_set_cmd()
    plan_out = tmp_path / "pilot-plan.json"

    # 1. PIN_PILOT_LESSONS without PIN_PILOT_SLOT_IDS refuses
    monkeypatch.delenv("PIN_PILOT_SLOT_IDS", raising=False)
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    with pytest.raises(RuntimeError, match="PIN_PILOT_LESSONS requires PIN_PILOT_SLOT_IDS"):
        pw.plan_pilot_lessons(rs, ["T02", "T03"], str(domain_table), plan_out, tmp_path / "w1")

    # 2. PIN_PILOT_TWINS not set to 0 refuses
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T02-C-01")
    monkeypatch.setenv("PIN_PILOT_TWINS", "1")
    with pytest.raises(RuntimeError, match="PIN_PILOT_LESSONS requires PIN_PILOT_TWINS=0"):
        pw.plan_pilot_lessons(rs, ["T02", "T03"], str(domain_table), plan_out, tmp_path / "w2")

    # 3. A lesson that role-set plan refuses fails
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    with pytest.raises(RuntimeError, match="exit "):
        pw.plan_pilot_lessons(rs, ["T02", "T99"], str(domain_table), plan_out, tmp_path / "w3")

    # 4. An id missing from every lesson fails in screen_plan
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T02-C-01,T06-C-01")
    pw.plan_pilot_lessons(rs, ["T02", "T03"], str(domain_table), plan_out, tmp_path / "w4")
    with pytest.raises(RuntimeError, match="PIN_PILOT_SLOT_IDS: T06-C-01 not in pilot-plan.json"):
        pw.screen_plan(plan_out, "PIN_PILOT_SLOT_IDS")

    # 5. Mismatched exclusion_state_sha256 across lesson plans refuses
    def fake_sh_mismatch(cmd: str, **kw) -> str:
        out_idx = cmd.split().index("--out") + 1
        out_p = Path(cmd.split()[out_idx])
        sha = "a" * 64 if "T02" in cmd else "b" * 64
        out_p.write_text(
            json.dumps({"slots": [{"id": "T02-C-01"}], "figures": [], "exclusion_state_sha256": sha}),
            encoding="utf-8",
        )
        return "ok"

    monkeypatch.setattr(pw, "sh", fake_sh_mismatch)
    with pytest.raises(RuntimeError, match="mismatched exclusion_state_sha256"):
        pw.plan_pilot_lessons(rs, ["T02", "T03"], str(domain_table), plan_out, tmp_path / "w5")


def test_write_only_critics_for_and_pilot_batch_no_critic(monkeypatch, tmp_path):
    # 1. PIN_PILOT_CRITIC=none with waves=1 returns {}
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR", raising=False)
    assert pw.critics_for(["gemma-4-31b-it"], waves=1) == {}

    # 2. PIN_PILOT_CRITIC=none with waves > 1 refuses
    with pytest.raises(RuntimeError, match="PIN_PILOT_CRITIC=none is allowed only when PIN_PILOT_WAVES=1"):
        pw.critics_for(["gemma-4-31b-it"], waves=2)

    # 3. PIN_PILOT_CRITIC=none combined with PIN_PILOT_CRITIC_FOR refuses
    monkeypatch.setenv("PIN_PILOT_CRITIC_FOR", "gemma-4-31b-it=muse-glimmer-30b")
    with pytest.raises(RuntimeError, match="PIN_PILOT_CRITIC=none takes no PIN_PILOT_CRITIC_FOR"):
        pw.critics_for(["gemma-4-31b-it"], waves=1)
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR", raising=False)

    # 4. Write-only pilot_batch (PILOT_WAVES=1, critic_for={}) fetches and serves ONLY the writer
    events, plan, pool = _fake_window(monkeypatch, tmp_path)
    monkeypatch.setattr(pw, "PILOT_WAVES", 1)
    G = "gemma-4-31b-it"
    arms = pw.pilot_batch({}, [], plan, tmp_path / "pilot-wo", [G], {}, pool, twins_after=False)
    assert events == [("fetch", G), ("serve", G), ("write", G, G), ("stop", G)]
    assert len(arms[G]["waves"]) == 1 and arms[G]["waves"][0]["result"]["states"]["awaiting_critique"] == 1

    # 5. pilot_batch reaching a critique wave with no critic fails its assertion
    events.clear()
    monkeypatch.setattr(pw, "PILOT_WAVES", 2)
    pool2 = pw.Pool({}, {}, {}, tmp_path / "cards.json", tmp_path / "out2")
    with pytest.raises(AssertionError, match="pilot_batch reached a critique wave with no critic"):
        pw.pilot_batch({}, [], plan, tmp_path / "pilot-bad", [G], {}, pool2, twins_after=False)


def test_pilot_window_write_only_pins_and_prefetches_no_critic(tmp_path, monkeypatch, domain_table):
    """PIN_PILOT_CRITIC=none with PIN_PILOT_WAVES=1 pins, prefetches, and serves only the writer (no critic),
    and records result['critic_for'] = 'none (write only)' and plans.lessons / per_lesson_sha256 / pilot_sha256."""
    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    pack = content / "pack"
    pack.mkdir()
    (pack / "pack.json").write_text("{}", encoding="utf-8")
    (content / "card-values.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)

    rs = _role_set_cmd()
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})

    pinned_names: list[list[str]] = []
    prefetched_names: list[list[str]] = []
    batch_calls: list[tuple] = []

    orig_check_pins = pw.check_pins

    def tracking_check_pins(names, listed, policies):
        pinned_names.append(list(names))
        return orig_check_pins(names, listed, policies)

    def tracking_prefetch(names, by_name, result):
        prefetched_names.append(list(names))

    def fake_pilot_batch(rs_arg, common, plan, out_dir, writers, critic_for, pool, **kw):
        batch_calls.append((list(writers), dict(critic_for), kw.get("twins_after")))
        return {w: {"waves": [{"wave": "write", "exit": 0}], "last": str(out_dir / w / "w0")} for w in writers}

    orig_sh = pw.sh

    def selective_sh(cmd: str, **kw) -> str:
        if "pack verify" in cmd or cmd.startswith("tar "):
            return "verified"
        if "plan --lessons" in cmd:
            out_idx = cmd.split().index("--out") + 1
            out_p = Path(cmd.split()[out_idx])
            lesson = cmd.split()[cmd.split().index("--lessons") + 1]
            out_p.write_text(
                json.dumps({
                    "slots": [
                        {"id": f"{lesson}-C-01", "lesson": lesson, "kind": "C"},
                        {"id": f"{lesson}-C-02", "lesson": lesson, "kind": "C"},
                    ],
                    "figures": [{"id": f"{lesson}-fig"}],
                    "exclusion_state_sha256": "e" * 64,
                }),
                encoding="utf-8",
            )
            return "planned"
        return orig_sh(cmd, **kw)

    monkeypatch.setattr(pw, "check_pins", tracking_check_pins)
    monkeypatch.setattr(pw, "prefetch_hosted", tracking_prefetch)
    monkeypatch.setattr(pw, "pilot_batch", fake_pilot_batch)
    monkeypatch.setattr(pw, "sh", selective_sh)

    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T03,T02")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T03-C-01,T02-C-02")

    result: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, result)

    assert pinned_names == [["gemma-4-31b-it"]]
    assert prefetched_names == [["gemma-4-31b-it"]]
    assert batch_calls == [(["gemma-4-31b-it"], {}, False)]
    assert result["critic_for"] == "none (write only)"
    assert result["plans"]["lessons"] == ["T03", "T02"]
    assert list(result["plans"]["per_lesson_sha256"]) == ["T03", "T02"]
    assert len(result["plans"]["pilot_sha256"]) == 64
    assert result["plans"]["twins_sha256"] is None
    assert not (work / "pilot" / "twins-plan.json").exists()


def test_pin_window_stub_write_only_lessons_ends_with_awaiting_critique_drafts(tmp_path, monkeypatch):
    from test_cards import SAMPLING

    from role_set import waves
    from role_set.pack import PackManifest, pack_hash

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()

    pack_dir = content / "pack"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
        "traits/T03.md": "## 3. As a verifier, you check numbers\n\nNumbers.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for name, text in trait_files.items():
        p = pack_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(trait_files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    writer = "gemma-4-31b-it"
    cards = {
        "candidates": [
            {
                "candidate": writer,
                "revision": "r1",
                "sampling": dict(SAMPLING),
                "chat_template_kwargs": {"enable_thinking": True},
                "transforms": ["reattach_thinking"],
            }
        ]
    }
    (content / "card-values.json").write_text(json.dumps(cards), encoding="utf-8")

    rs = _role_set_cmd()
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw.Pool, "proxy", lambda self, name: None)
    # A free port of its own: under -n auto the window-script tests serve their stub windows on PILOT_PORT at the same
    # time (CI on #97: "Address already in use").
    import socket as _socket

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(pw, "PILOT_PORT", s.getsockname()[1])

    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "e7abbcdff51defefccd99b5a68c0e8dec6fa6f09")
    monkeypatch.setenv("PIN_RND_COMMIT", "04dccfad9419cd79c4068703ac6cf1442850c9de")
    monkeypatch.setenv("PIN_PILOT_WRITERS", writer)
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T03,T02")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T03-C-01,T02-C-06")
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR", raising=False)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_DIR", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)

    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_GGUF_REPO", "PIN_GGUF_REVISION"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pw, "TOKEN_PATH", tmp_path / "shm-hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)

    res: dict = {"machine": pw.machine_line()}
    pw.pilot_window({}, res)

    assert res["critic_for"] == "none (write only)"
    assert res["plans"]["lessons"] == ["T03", "T02"]
    assert list(res["plans"]["per_lesson_sha256"]) == ["T03", "T02"]
    assert len(res["plans"]["pilot_sha256"]) == 64
    assert list(res["serves"]) == [writer]
    assert res["model_errors"] == 0, res["arms"]

    drafts_path = work / "pilot" / "pilot" / writer / "w0" / waves.DRAFTS_FILE
    _, drafts = waves.read(drafts_path)
    assert {d["slot"]["id"] for d in drafts} == {"T03-C-01", "T02-C-06"}
    assert all(d["state"] == "awaiting_critique" for d in drafts)
    assert not (work / "pilot" / "pilot" / writer / "w1").exists()


def test_pilot_window_verifies_pool_sha_before_serving_and_passes_pool_to_common(tmp_path, monkeypatch):
    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    pack = content / "pack"
    pack.mkdir()
    (pack / "pack.json").write_text("{}", encoding="utf-8")
    (content / "card-values.json").write_text("{}", encoding="utf-8")
    pool_file = content / "pool.jsonl"
    pool_file.write_text('{"text": "a\\nb\\nc\\nd\\n", "content_kind": "changelog", "source": {}}\n', encoding="utf-8")
    pool_sha = pw.sha256(pool_file)

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    rs = _role_set_cmd()
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})

    events: list[str] = []
    seen_common: list[list[str]] = []

    def tracking_check_pins(names, listed, policies):
        events.append("check_pins")

    def tracking_prefetch(names, by_name, result):
        events.append("prefetch")

    def fake_pilot_batch(rs_arg, common, plan, out_dir, writers, critic_for, pool, **kw):
        events.append("pilot_batch")
        seen_common.append(list(common))
        return {w: {"waves": [{"wave": "write", "exit": 0}], "last": str(out_dir / w / "w0")} for w in writers}

    def selective_sh(cmd: str, **kw) -> str:
        if "pack verify" in cmd or cmd.startswith("tar "):
            return "verified"
        if "plan --lessons" in cmd:
            out_idx = cmd.split().index("--out") + 1
            out_p = Path(cmd.split()[out_idx])
            lesson = cmd.split()[cmd.split().index("--lessons") + 1]
            out_p.write_text(
                json.dumps({
                    "slots": [
                        {"id": f"{lesson}-C-01", "lesson": lesson, "kind": "C", "real_source": True, "content_kind": "changelog"},
                    ],
                    "figures": [{"id": f"{lesson}-fig"}],
                    "exclusion_state_sha256": "e" * 64,
                }),
                encoding="utf-8",
            )
            return "planned"
        raise AssertionError(f"unexpected sh: {cmd}")

    monkeypatch.setattr(pw, "check_pins", tracking_check_pins)
    monkeypatch.setattr(pw, "prefetch_hosted", tracking_prefetch)
    monkeypatch.setattr(pw, "pilot_batch", fake_pilot_batch)
    monkeypatch.setattr(pw, "sh", selective_sh)

    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T02")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T02-C-01")

    # 1. Missing PIN_POOL_SHA256 when plan has real_source slots refuses before check_pins / prefetch / serve.
    monkeypatch.delenv("PIN_POOL_SHA256", raising=False)
    monkeypatch.delenv("PIN_POOL", raising=False)
    with pytest.raises(RuntimeError, match="PIN_POOL_SHA256 is required when the plan has real_source slots"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})
    assert events == []

    # 2. Mismatched PIN_POOL_SHA256 refuses before check_pins / prefetch / serve.
    monkeypatch.setenv("PIN_POOL_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="!= PIN_POOL_SHA256"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})
    assert events == []

    # 3. When PIN_POOL is unset, resolve_pilot_pool prefers Path(rs["src"]) / "roles" / "verifier" / "pool.jsonl"
    # over CONTENT / "pool.jsonl", and falls back to CONTENT / "pool.jsonl" when the committed pool does not exist.
    committed_pool = Path(rs["src"]) / "roles" / "verifier" / "pool.jsonl"
    committed_sha = pw.sha256(committed_pool)
    monkeypatch.delenv("PIN_POOL", raising=False)
    monkeypatch.setenv("PIN_POOL_SHA256", committed_sha)
    result_default: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, result_default)
    assert result_default["pool_sha256"] == committed_sha
    assert events == ["check_pins", "prefetch", "pilot_batch"]
    assert seen_common and ["--pool", str(committed_pool)] == seen_common[0][-2:]

    plan_for_fallback = work / "pilot" / "pilot-plan.json"
    monkeypatch.setenv("PIN_POOL_SHA256", pool_sha)
    got_fb_path, got_fb_sha = pw.resolve_pilot_pool({"src": str(tmp_path / "no-such-repo")}, plan_for_fallback)
    assert got_fb_path == pool_file and got_fb_sha == pool_sha

    # 4. Explicit PIN_POOL overrides the default and passes --pool in common.
    events.clear()
    seen_common.clear()
    custom_pool = tmp_path / "custom-pool.jsonl"
    custom_pool.write_bytes(pool_file.read_bytes())
    monkeypatch.setenv("PIN_POOL", str(custom_pool))
    monkeypatch.setenv("PIN_POOL_SHA256", pool_sha)
    result: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, result)
    assert result["pool_sha256"] == pool_sha
    assert events == ["check_pins", "prefetch", "pilot_batch"]
    assert seen_common and ["--pool", str(custom_pool)] == seen_common[0][-2:]


# ---- T10 window plumbing: PIN_PILOT_BUFFERS=1, PIN_PILOT_SLOT_IDS=ALL, PIN_CRITIQUE_SCOPE (DESIGN §8/§11, 2026-10-21) ---


def test_t10_buffers_plan_and_screen_all_keeps_60_slots_and_refusals(tmp_path, monkeypatch, domain_table):
    rs = _role_set_cmd()
    plan_out = tmp_path / "t10-plan.json"
    seen_cmds: list[str] = []
    orig_sh = pw.sh

    def tracking_sh(cmd: str, **kw) -> str:
        seen_cmds.append(cmd)
        return orig_sh(cmd, **kw)

    monkeypatch.setattr(pw, "sh", tracking_sh)
    monkeypatch.setenv("PIN_PILOT_BUFFERS", "1")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T10")
    monkeypatch.setenv("PIN_PILOT_SEED", "0")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "ALL")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")

    meta = pw.plan_pilot_lessons(rs, ["T10"], str(domain_table), plan_out, tmp_path / "work")
    assert any("--lessons T10 --buffers --seed 0" in c for c in seen_cmds), seen_cmds
    assert meta["buffers"] is True
    assert meta["lessons"] == ["T10"]
    assert len(meta["per_lesson_sha256"]["T10"]) == 64

    kept = pw.screen_plan(plan_out, "PIN_PILOT_SLOT_IDS")
    assert kept is not None and len(kept) == 60
    screened = json.loads(plan_out.read_text(encoding="utf-8"))
    slot_ids = [s["id"] for s in screened["slots"]]
    assert len(slot_ids) == 60 and slot_ids == kept
    for buf_id in ("T10-C-09", "T10-C-10", "T10-C-11", "T10-C-12", "T10-C-AP-06", "T10-D-AP-06"):
        assert buf_id in slot_ids, buf_id
    assert sum(1 for s in screened["slots"] if s.get("real_source")) == 6

    # Refusals:
    # 1. Non-T10 lesson with PIN_PILOT_BUFFERS=1
    with pytest.raises(RuntimeError, match="PIN_PILOT_BUFFERS=1 is allowed only for T10"):
        pw.plan_pilot_lessons(rs, ["T02"], str(domain_table), tmp_path / "bad1.json", tmp_path / "w1")
    with pytest.raises(RuntimeError, match="PIN_PILOT_BUFFERS=1 is allowed only for T10"):
        pw.plan_pilot_lessons(rs, ["T10", "T02"], str(domain_table), tmp_path / "bad2.json", tmp_path / "w2")

    # 2. Twins on with PIN_PILOT_BUFFERS=1
    monkeypatch.setenv("PIN_PILOT_TWINS", "1")
    with pytest.raises(RuntimeError, match="PIN_PILOT_BUFFERS=1 requires PIN_PILOT_TWINS=0"):
        pw.plan_pilot_lessons(rs, ["T10"], str(domain_table), tmp_path / "bad3.json", tmp_path / "w3")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")

    # 3. PIN_PILOT_SLOT_IDS=ALL without PIN_PILOT_BUFFERS=1
    monkeypatch.setenv("PIN_PILOT_BUFFERS", "0")
    with pytest.raises(RuntimeError, match="PIN_PILOT_SLOT_IDS=ALL requires PIN_PILOT_BUFFERS=1"):
        pw.screen_plan(plan_out, "PIN_PILOT_SLOT_IDS")

    # 4. Invalid PIN_PILOT_BUFFERS value
    monkeypatch.setenv("PIN_PILOT_BUFFERS", "2")
    with pytest.raises(RuntimeError, match="PIN_PILOT_BUFFERS must be '0' or '1'"):
        pw.buffers_enabled()


def test_critique_scope_passes_to_critique_waves_only_and_refuses_bad_value_before_serving(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    seen_cmds: list[list[str]] = []

    def fake_run_proc(cmd, **kw):
        seen_cmds.append(list(cmd))
        return CompletedProcess(cmd, 0, '{"dropped": 0, "states": {"awaiting_critique": 0, "needs_rewrite": 0}}\n', "")

    monkeypatch.setattr(pw, "run_proc", fake_run_proc)
    monkeypatch.delenv("PIN_DEADLINE", raising=False)
    rs = {"role_set": "role-set", "src": str(tmp_path)}
    writer = {"served_as": "gemma-4-31b-it", "gguf_sha256": "a" * 64}
    critic = {"served_as": "muse-glimmer-30b", "gguf_sha256": "b" * 64}

    monkeypatch.setenv("PIN_CRITIQUE_SCOPE", "verdict")
    pw.write_wave(rs, ["--plan", "p.json"], "write", tmp_path / "arm" / "w0", writer=writer)
    pw.write_wave(rs, ["--plan", "p.json"], "critique", tmp_path / "arm" / "w1", critic=critic, drafts=tmp_path / "d.jsonl")
    pw.write_wave(rs, ["--plan", "p.json"], "rewrite", tmp_path / "arm" / "w2", writer=writer, drafts=tmp_path / "d.jsonl")

    assert "--critique-scope" not in seen_cmds[0]
    assert "--critique-scope" in seen_cmds[1]
    idx = seen_cmds[1].index("--critique-scope")
    assert seen_cmds[1][idx + 1] == "verdict"
    assert "--critique-scope" not in seen_cmds[2]

    # Bad PIN_CRITIQUE_SCOPE refuses before check_pins / prefetch / serve in pilot_window.
    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    (content / "pack").mkdir()
    (content / "pack" / "pack.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "sh", lambda cmd, **kw: "ok")
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: pytest.fail("check_pins must not run"))
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_CRITIQUE_SCOPE", "everything")
    with pytest.raises(RuntimeError, match="REFUSED: PIN_CRITIQUE_SCOPE must be 'all' or 'verdict'"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})


def _make_training_tar(tar_path: Path, files: dict[str, bytes]) -> dict[str, str]:
    import hashlib
    import io
    import tarfile

    shas: dict[str, str] = {}
    with tarfile.open(tar_path, "w") as tf:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
            shas[name] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    return shas


def test_training_inputs_refuses_missing_vars_sidecar_or_sha_mismatch_before_serving_and_orders_by_frozen_md(tmp_path, monkeypatch):
    real_frozen = pw.FROZEN_TRAINING_FILES
    rs = _role_set_cmd()
    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    (content / "pack").mkdir()
    (content / "pack" / "pack.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "resolve_pilot_pool", lambda *a, **kw: (None, None))
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: pytest.fail("check_pins must not run before training_inputs"))
    monkeypatch.setattr(pw, "Pool", lambda *a, **kw: pytest.fail("Pool must not be created before training_inputs"))

    # Unbuffered plan that still has an apply slot (gating on split == 'apply', not buffers_enabled()).
    def fake_sh(cmd, **kw):
        if "plan --lessons T10" in cmd:
            out = Path(cmd.split("--out ", 1)[1].split()[0])
            out.write_text(
                json.dumps({
                    "slots": [
                        {"id": "T10-C-01", "lesson": "T10", "kind": "C", "real_source": False},
                        {"id": "T10-C-AP-01", "lesson": "T10", "kind": "C", "split": "apply", "real_source": False},
                    ],
                    "exclusion_state_sha256": "0" * 64,
                }),
                encoding="utf-8",
            )
        return "ok"

    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_BUFFERS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T10")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T10-C-01,T10-C-AP-01")
    monkeypatch.delenv("PIN_TRAINING_INPUTS", raising=False)
    monkeypatch.delenv("PIN_TRAINING_SHAS", raising=False)

    # 1. Missing PIN_TRAINING_SHAS
    with pytest.raises(RuntimeError, match="REFUSED: plan has apply slots; PIN_TRAINING_SHAS is required"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 2. Missing tar file
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={'a' * 64},pilot-t04.models.jsonl={'b' * 64}")
    monkeypatch.setenv("PIN_TRAINING_INPUTS", str(tmp_path / "absent.tar"))
    with pytest.raises(RuntimeError, match="REFUSED: plan has apply slots; PIN_TRAINING_INPUTS tar not found"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 3. File in tar without a sha in PIN_TRAINING_SHAS
    tar_path = tmp_path / "training.tar"
    shas = _make_training_tar(tar_path, {"pilot-t04.jsonl": b"{}\n", "pilot-t04.models.jsonl": b"", "extra.jsonl": b"{}\n"})
    monkeypatch.setenv("PIN_TRAINING_INPUTS", str(tar_path))
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={shas['pilot-t04.jsonl']},pilot-t04.models.jsonl={shas['pilot-t04.models.jsonl']}")
    with pytest.raises(RuntimeError, match="REFUSED: no sha256 in PIN_TRAINING_SHAS for extra.jsonl"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 4. Missing frozen file from FROZEN_TRAINING_FILES refuses before serving
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        (
            ("docs/runs/stage1-training-data/pilot-t04.jsonl", shas["pilot-t04.jsonl"]),
            ("docs/runs/stage1-training-data/batch2-t01.jsonl", "c" * 64),
        ),
    )
    shas_one = _make_training_tar(tar_path, {"pilot-t04.jsonl": b"{}\n", "pilot-t04.models.jsonl": b""})
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={shas_one['pilot-t04.jsonl']},pilot-t04.models.jsonl={shas_one['pilot-t04.models.jsonl']}")
    with pytest.raises(RuntimeError, match=r"REFUSED: missing frozen training file: docs/runs/stage1-training-data/batch2-t01\.jsonl"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 5. Extra training .jsonl data file not in FROZEN_TRAINING_FILES refuses before serving
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        (("docs/runs/stage1-training-data/pilot-t04.jsonl", shas["pilot-t04.jsonl"]),),
    )
    shas_extra = _make_training_tar(
        tar_path,
        {
            "pilot-t04.jsonl": b"{}\n",
            "pilot-t04.models.jsonl": b"",
            "extra.jsonl": b"{}\n",
            "extra.models.jsonl": b"",
        },
    )
    monkeypatch.setenv("PIN_TRAINING_SHAS", ",".join(f"{k}={v}" for k, v in shas_extra.items()))
    with pytest.raises(RuntimeError, match=r"REFUSED: extra training \.jsonl file not in FROZEN_TRAINING_FILES: extra\.jsonl"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 6. Missing .models.jsonl sidecar refuses before serving
    shas_no_sidecar = _make_training_tar(tar_path, {"pilot-t04.jsonl": b"{}\n"})
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={shas_no_sidecar['pilot-t04.jsonl']}")
    with pytest.raises(RuntimeError, match=r"REFUSED: missing training sidecar: pilot-t04\.models\.jsonl"):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 7. SHA mismatch against PIN_TRAINING_SHAS refuses before serving
    shas_ok = _make_training_tar(tar_path, {"pilot-t04.jsonl": b'{"id":"T04-A-01"}\r\n', "pilot-t04.models.jsonl": b""})
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        (("docs/runs/stage1-training-data/pilot-t04.jsonl", shas_ok["pilot-t04.jsonl"]),),
    )
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={'0' * 64},pilot-t04.models.jsonl={shas_ok['pilot-t04.models.jsonl']}")
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: training input sha256 mismatch: pilot-t04\.jsonl is {shas_ok['pilot-t04.jsonl'][:12]}, not 000000000000",
    ):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 8. Data file whose sha matches PIN_TRAINING_SHAS (env) but not hard-coded FROZEN_TRAINING_FILES refuses before serving
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        (("docs/runs/stage1-training-data/pilot-t04.jsonl", "f" * 64),),
    )
    monkeypatch.setenv("PIN_TRAINING_SHAS", f"pilot-t04.jsonl={shas_ok['pilot-t04.jsonl']},pilot-t04.models.jsonl={shas_ok['pilot-t04.models.jsonl']}")
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: frozen training sha256 mismatch: pilot-t04\.jsonl is {shas_ok['pilot-t04.jsonl'][:12]}, not FROZEN_TRAINING_FILES ffffffffffff",
    ):
        pw.pilot_window({}, {"machine": {"gpu": "A100"}})

    # 9. LF-normalized sha matches CRLF bytes, unpacks read-only, and orders in FROZEN.md training order (not sorted path order).
    tar_files: dict[str, bytes] = {}
    for rel_path, _ in real_frozen:
        name = Path(rel_path).name
        stem = name[:-len(".jsonl")]
        tar_files[name] = f'{{"id": "{stem}"}}\r\n'.encode("utf-8")
        tar_files[f"{stem}.models.jsonl"] = b""
    all_shas = _make_training_tar(tar_path, tar_files)
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        tuple((rel_path, all_shas[Path(rel_path).name]) for rel_path, _ in real_frozen),
    )
    # Pass PIN_TRAINING_SHAS in sorted (alphabetical) order, which puts batch2-t01 before pilot-t04.
    monkeypatch.setenv("PIN_TRAINING_SHAS", ",".join(f"{k}={all_shas[k]}" for k in sorted(all_shas)))
    res: dict = {}
    paths = pw.training_inputs(res)
    expected_order = [Path(rel).name for rel, _ in real_frozen]
    assert [p.name for p in paths] == expected_order
    assert res["training_inputs"]["order"] == expected_order
    assert not os.access(paths[0], os.W_OK)


def test_training_inputs_with_real_frozen_training_files_refuses_tar_missing_a_file(tmp_path, monkeypatch):
    assert len(pw.FROZEN_TRAINING_FILES) == 21
    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)

    tar_files: dict[str, bytes] = {}
    for rel_path, _ in pw.FROZEN_TRAINING_FILES[:-1]:
        name = Path(rel_path).name
        stem = name[:-len(".jsonl")]
        tar_files[name] = f'{{"id": "{stem}"}}\n'.encode("utf-8")
        tar_files[f"{stem}.models.jsonl"] = b""
    tar_path = tmp_path / "missing-one.tar"
    shas = _make_training_tar(tar_path, tar_files)
    monkeypatch.setenv("PIN_TRAINING_INPUTS", str(tar_path))
    monkeypatch.setenv("PIN_TRAINING_SHAS", ",".join(f"{k}={v}" for k, v in shas.items()))
    with pytest.raises(
        RuntimeError,
        match=r"REFUSED: missing frozen training file: docs/runs/stage1-training-data/batch8-l3\.jsonl",
    ):
        pw.training_inputs({})


def test_t10_stub_end_to_end_write_only_then_critique_resume_with_scope_verdict(tmp_path, monkeypatch):
    from test_cards import SAMPLING

    from role_set import waves
    from role_set.pack import PackManifest, pack_hash
    from role_set.runner import models_line

    content = tmp_path / "content"
    work1 = content / "pin-w0"
    work2 = content / "pin-w1"
    content.mkdir()
    work1.mkdir()
    work2.mkdir()

    pack_dir = content / "pack"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. As a verifier, you test whether the reason holds\n\nTest the link.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for name, text in trait_files.items():
        p = pack_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(trait_files)
    (pack_dir / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")

    writer = "gemma-4-31b-it"
    critic = "muse-glimmer-30b"
    cards = {
        "candidates": [
            {
                "candidate": c,
                "revision": "r1",
                "sampling": dict(SAMPLING),
                "chat_template_kwargs": {"enable_thinking": True},
                "transforms": ["reattach_thinking"],
            }
            for c in (writer, critic)
        ]
    }
    (content / "card-values.json").write_text(json.dumps(cards), encoding="utf-8")

    # Build a non-empty --training fixture tar with two files in FROZEN.md and their .models.jsonl sidecars.
    # Include a kind-B row with its stamped sidecar and a kind-D row carrying a known 8-word span.
    pinned_span = "harbor master logbook on the west pier states"
    t04_d_row = {
        "id": "T04-D-01",
        "lesson": "T04",
        "trait": "calibrates confidence",
        "kind": "D",
        "side": "calibrate",
        "content_kind": "notice",
        "tier": 3,
        "sources": [{"invented": "fixture"}],
        "invented": True,
        "key_basis": "construction",
        "tags": [],
        "material": "Pier hours\nThe west pier opens at 06:00.\n",
        "verdict": "unsupported",
        "deciding": ["The west pier opens at 06:00."],
        "escalate": "no",
        "turns": [{
            "user": f"Check the claim.\nMaterial:\nPier hours\nThe west pier opens at 06:00.\nStatement: The {pinned_span} the pier is shut.",
            "thinking": "The line says 06:00.",
            "reply": "It opens at 06:00.\nVERDICT: unsupported\nDECIDING: The west pier opens at 06:00.\nESCALATE: no",
        }],
    }
    t01_b_model = {
        "entities": ["Ash", "Briar", "Cole"],
        "attributes": {"turn": {"values": ["turn 1", "turn 2", "turn 3"], "ordered": True}},
        "facts": [{"line": 2, "rule": ["is_not", "Ash", "turn", "turn 1"]}, {"line": 3, "rule": ["after", "Briar", "Ash", "turn"]}],
        "flavour": [1],
        "asks": {"attribute": "turn"},
    }
    t01_b_row = {
        "id": "T01-B-01",
        "lesson": "T01",
        "trait": "skeptical",
        "kind": "B",
        "side": "skeptical",
        "content_kind": "puzzle",
        "tier": 3,
        "sources": [{"invented": "fixture"}],
        "invented": True,
        "key_basis": "construction",
        "tags": [],
        "answer": "ash=turn 2; briar=turn 3; cole=turn 1",
        "structure": "constraint_grid",
        "turns": [{"user": "1. Three canoes.\nQuestion: Which turn?", "thinking": "1. Step.", "reply": "1. Step.\nAnswer: ash=turn 2; briar=turn 3; cole=turn 1"}],
    }
    t01_b_line = json.dumps(t01_b_row, ensure_ascii=False)
    t01_b_sidecar = json.dumps(models_line("T01-B-01", "constraint_grid", t01_b_line, t01_b_model), ensure_ascii=False) + "\n"
    training_tar = content / "training-inputs.tar"
    training_shas = _make_training_tar(
        training_tar,
        {
            "batch2-t01.jsonl": (t01_b_line + "\n").encode("utf-8"),
            "batch2-t01.models.jsonl": t01_b_sidecar.encode("utf-8"),
            "pilot-t04.jsonl": (json.dumps(t04_d_row, ensure_ascii=False) + "\n").encode("utf-8"),
            "pilot-t04.models.jsonl": b"",
        },
    )
    monkeypatch.setattr(
        pw,
        "FROZEN_TRAINING_FILES",
        (
            ("docs/runs/stage1-training-data/pilot-t04.jsonl", training_shas["pilot-t04.jsonl"]),
            ("docs/runs/stage1-training-data/batch2-t01.jsonl", training_shas["batch2-t01.jsonl"]),
        ),
    )

    rs = _role_set_cmd()
    committed_pool = Path(rs["src"]) / "roles" / "verifier" / "pool.jsonl"
    committed_pool_sha = pw.sha256(committed_pool)

    # Plant the training fixture's 8-word span into the first apply slot's first draft to prove SPENT refuses it.
    from role_set.mock import MockWriter

    base_mock = MockWriter()
    first_apply_planted = {"done": False}

    class _PlantingWriter:
        def __init__(self, inner: MockWriter):
            self.inner = inner
            self.answers = inner.answers

        def __call__(self, messages, seed=0, slot_id=None):
            c = self.inner(messages, seed=seed, slot_id=slot_id)
            user_text = messages[-1]["content"] if messages else ""
            if not first_apply_planted["done"] and "Add r10_twin" in user_text and not user_text.startswith("Refused:"):
                first_apply_planted["done"] = True
                data = json.loads(c.text)
                data["statement"] = f"The {pinned_span} the desk is closed on Sundays."
                return type(c)(
                    text=json.dumps(data),
                    prompt_tokens=c.prompt_tokens,
                    completion_tokens=c.completion_tokens,
                    model=c.model,
                    seconds=c.seconds,
                    finish_reason=c.finish_reason,
                )
            return c

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw.Pool, "proxy", lambda self, name: None)
    monkeypatch.setattr(pw, "_STUB_WRITER", _PlantingWriter(base_mock))

    import socket as _socket

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(pw, "PILOT_PORT", s.getsockname()[1])

    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_GGUF_REPO", "PIN_GGUF_REVISION", "PIN_POOL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pw, "TOKEN_PATH", tmp_path / "shm-hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)

    # Window 1: T10 write-only with PIN_PILOT_BUFFERS=1 and PIN_PILOT_SLOT_IDS=ALL
    monkeypatch.setattr(pw, "WORK", work1)
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "e7abbcdff51defefccd99b5a68c0e8dec6fa6f09")
    monkeypatch.setenv("PIN_RND_COMMIT", "04dccfad9419cd79c4068703ac6cf1442850c9de")
    monkeypatch.setenv("PIN_PILOT_WRITERS", writer)
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_BUFFERS", "1")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T10")
    monkeypatch.setenv("PIN_PILOT_SEED", "0")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "ALL")
    monkeypatch.setenv("PIN_POOL_SHA256", committed_pool_sha)
    monkeypatch.setenv("PIN_TRAINING_INPUTS", str(training_tar))
    # Pass batch2-t01 before pilot-t04 in PIN_TRAINING_SHAS to verify FROZEN.md ordering puts pilot-t04 first.
    monkeypatch.setenv("PIN_TRAINING_SHAS", ",".join(f"{k}={v}" for k, v in training_shas.items()))
    monkeypatch.delenv("PIN_CRITIQUE_SCOPE", raising=False)
    monkeypatch.delenv("PIN_PILOT_CRITIC_FOR", raising=False)
    monkeypatch.delenv("PIN_RESUME_WAVE", raising=False)
    monkeypatch.delenv("PIN_RESUME_DIR", raising=False)
    monkeypatch.delenv("PIN_RESUME_CALLS_DIR", raising=False)

    res1: dict = {"machine": pw.machine_line()}
    pw.pilot_window({}, res1)
    assert res1["model_errors"] == 0, res1["arms"]
    assert res1["plans"]["buffers"] is True
    assert res1["plans"]["lessons"] == ["T10"]
    assert len(res1["plans"]["per_lesson_sha256"]["T10"]) == 64
    assert res1["screen"]["kept"] == 60 and len(res1["screen"]["pilot"]) == 60
    assert res1["pool_sha256"] == committed_pool_sha
    assert res1["critique_scope"] == "all"
    assert res1["training_inputs"]["order"] == ["pilot-t04.jsonl", "batch2-t01.jsonl"]

    w0_drafts_path = work1 / "pilot" / "pilot" / writer / "w0" / waves.DRAFTS_FILE
    w0_header, w0_drafts = waves.read(w0_drafts_path)
    assert len(w0_drafts) == 60
    assert [x["file"] for x in w0_header["batch_pins"]["training"]] == ["pilot-t04.jsonl", "batch2-t01.jsonl"]
    # All 48 training slots come before all 12 apply slots, and the first apply slot was refused SPENT against the fixture on attempt 1.
    assert [d["slot"].get("split") == "apply" for d in w0_drafts] == [False] * 48 + [True] * 12
    assert w0_drafts[48]["attempt"] == 2
    assert w0_drafts[48]["refusals"][0]["code"] == "SPENT"
    assert w0_drafts[48]["refusals"][0]["detail"] == pinned_span

    # Window 2: Critique resume with PIN_CRITIQUE_SCOPE=verdict
    resume_dir = tmp_path / "resume-w0"
    resume_dir.mkdir()
    shutil.copyfile(w0_drafts_path, resume_dir / f"{writer}.jsonl")
    shutil.copyfile(w0_drafts_path.with_name(w0_drafts_path.name + ".sha256"), resume_dir / f"{writer}.jsonl.sha256")

    monkeypatch.setattr(pw, "WORK", work2)
    monkeypatch.setenv("PIN_PILOT_CRITIC", critic)
    monkeypatch.setenv("PIN_PILOT_WAVES", "2")
    monkeypatch.setenv("PIN_CRITIQUE_SCOPE", "verdict")
    monkeypatch.setenv("PIN_RESUME_WAVE", "0")
    monkeypatch.setenv("PIN_RESUME_DIR", str(resume_dir))

    res2: dict = {"machine": pw.machine_line()}
    pw.pilot_window({}, res2)
    assert res2["model_errors"] == 0, res2["arms"]
    assert res2["critique_scope"] == "verdict"
    assert res2["training_inputs"]["order"] == ["pilot-t04.jsonl", "batch2-t01.jsonl"]
    assert list(res2["serves"]) == [critic]

    w1_dir = work2 / "pilot" / "pilot" / writer / "w1"
    w1_header, w1_drafts = waves.read(w1_dir / waves.DRAFTS_FILE)
    assert len(w1_drafts) == 60
    assert [x["file"] for x in w1_header["batch_pins"]["training"]] == ["pilot-t04.jsonl", "batch2-t01.jsonl"]

    # Kinds A, B, and written E get no critic call in the critique wave's call journal or review records.
    critic_journal = work2 / "pilot" / "pilot" / writer / f"calls-w1-{critic}.jsonl"
    journal_entries = [json.loads(ln) for ln in critic_journal.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert journal_entries
    checks = [e.get("check") for e in journal_entries]
    assert "answer" not in checks
    assert checks.count("twin_verdict") == 12
    assert checks.count("trait") == 36 * 3
    review_by_id = {
        r["id"]: r["critic"]
        for r in (json.loads(ln) for ln in (w1_dir / "review.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip())
    }
    for d in w1_drafts:
        slot = d["slot"]
        assert d["state"] == "accepted", slot["id"]
        if slot["kind"] in ("A", "B") or (slot["kind"] == "E" and slot.get("e_type") == "written"):
            assert review_by_id[slot["id"]] == [], slot["id"]
        else:
            assert len(review_by_id[slot["id"]]) > 0, slot["id"]


def test_pilot_window_records_readout_stderr_on_nonzero_exit_and_omits_it_on_zero(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    pack = content / "pack"
    pack.mkdir()
    (pack / "pack.json").write_text("{}", encoding="utf-8")
    (content / "card-values.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    rs = _role_set_cmd()
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: None)
    monkeypatch.setattr(pw, "prefetch_hosted", lambda *a, **kw: None)

    w0_dir = work / "pilot" / "pilot" / "gemma-4-31b-it" / "w0"
    w0_dir.mkdir(parents=True)
    (w0_dir / "batch-report.json").write_text("{}", encoding="utf-8")

    def fake_pilot_batch(rs_arg, common, plan, out_dir, writers, critic_for, pool, **kw):
        return {w: {"waves": [{"wave": "write", "exit": 0}], "last": str(w0_dir)} for w in writers}

    def fake_sh(cmd: str, **kw) -> str:
        if "pack verify" in cmd or cmd.startswith("tar "):
            return "ok"
        if "plan --lessons" in cmd:
            out_idx = cmd.split().index("--out") + 1
            out_p = Path(cmd.split()[out_idx])
            out_p.write_text(
                json.dumps({"slots": [{"id": "T02-C-01", "lesson": "T02", "kind": "C"}], "exclusion_state_sha256": "e" * 64}),
                encoding="utf-8",
            )
            return "planned"
        raise AssertionError(f"unexpected sh: {cmd}")

    monkeypatch.setattr(pw, "pilot_batch", fake_pilot_batch)
    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T02")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T02-C-01")
    monkeypatch.delenv("PIN_STUB", raising=False)

    # 1. Failing readout (returncode != 0) records readout_stderr[-2500:]
    long_err = "X" * 3000 + "KeyError: 'accepted'\n"
    monkeypatch.setattr(
        pw,
        "run_proc",
        lambda cmd, **kw: CompletedProcess(cmd, 1, "", long_err),
    )
    res_fail: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, res_fail)
    arm_pilot = res_fail["arms"]["gemma-4-31b-it"]["pilot"]
    assert arm_pilot["readout_exit"] == 1
    assert arm_pilot["readout_tail"] == ""
    assert arm_pilot["readout_stderr"] == long_err[-2500:]
    assert len(arm_pilot["readout_stderr"]) == 2500 and arm_pilot["readout_stderr"].endswith("KeyError: 'accepted'\n")

    # 2. Passing readout (returncode == 0) keeps readout_tail and sets no readout_stderr
    monkeypatch.setattr(
        pw,
        "run_proc",
        lambda cmd, **kw: CompletedProcess(cmd, 0, "# Readout: w0\n", "benign warning"),
    )
    res_ok: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, res_ok)
    arm_ok = res_ok["arms"]["gemma-4-31b-it"]["pilot"]
    assert arm_ok["readout_exit"] == 0
    assert arm_ok["readout_tail"] == "# Readout: w0\n"
    assert "readout_stderr" not in arm_ok


def test_pin_cuda_archs_default_override_and_refusals(tmp_path, monkeypatch):
    import sys as _sys
    import types as _types

    assert pw.CUDA_ARCHS == "80;90;100"
    assert pw.LLAMA_CPP_COMMIT == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert pw.SERVER_SHA256 == "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"

    fake_nvidia_root = tmp_path / "nvidia" / "cu13"
    nvcc_bin = fake_nvidia_root / "bin" / "nvcc"
    nvcc_bin.parent.mkdir(parents=True)
    nvcc_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    (fake_nvidia_root / "lib").mkdir()
    monkeypatch.setitem(_sys.modules, "nvidia", _types.SimpleNamespace(__path__=[str(tmp_path / "nvidia")]))
    monkeypatch.setattr(pw, "WORK", tmp_path / "work")
    monkeypatch.setattr(pw, "say", lambda *a: None)

    seen_cmds: list[str] = []

    def fake_sh(cmd: str, **kw) -> str:
        seen_cmds.append(cmd)
        if "--version" in cmd:
            return "Cuda compilation tools, release 13.4, V13.4.92\nBuild cuda_13.4.r13.4/compiler.000_0"
        if "ldconfig -p" in cmd:
            return "/usr/lib64-nvidia/libcuda.so"
        if cmd.startswith("cmake --build"):
            bin_dir = tmp_path / "work" / "llama.cpp" / "build" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            (bin_dir / "llama-server").write_bytes(b"server-bytes")
            (bin_dir / "llama-quantize").write_bytes(b"quantize-bytes")
            return "built"
        return "ok"

    monkeypatch.setattr(pw, "sh", fake_sh)

    # 1. Default archs unchanged when PIN_CUDA_ARCHS is unset
    monkeypatch.delenv("PIN_CUDA_ARCHS", raising=False)
    assert pw.cuda_archs() == "80;90;100"
    b_default = pw.build()
    assert b_default["cuda_archs"] == "80;90;100"
    assert b_default["commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert any("-DCMAKE_CUDA_ARCHITECTURES='80;90;100'" in c for c in seen_cmds), seen_cmds

    # 2. Override used when PIN_CUDA_ARCHS matches ^[0-9]+(;[0-9]+)*$
    seen_cmds.clear()
    monkeypatch.setenv("PIN_CUDA_ARCHS", "80;90;100;120")
    assert pw.cuda_archs() == "80;90;100;120"
    b_override = pw.build()
    assert b_override["cuda_archs"] == "80;90;100;120"
    assert b_override["commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert any("-DCMAKE_CUDA_ARCHITECTURES='80;90;100;120'" in c for c in seen_cmds), seen_cmds

    # 3. Malformed override refused before running any shell command
    seen_cmds.clear()
    for bad in ("", "sm120", "80,90,100", "80;90;", ";80;90", "80;;90", "80;90\n", "80' && echo bad"):
        monkeypatch.setenv("PIN_CUDA_ARCHS", bad)
        with pytest.raises(RuntimeError, match=r"REFUSED: PIN_CUDA_ARCHS=.* must match \^\[0-9\]\+\(;\[0-9\]\+\)\*\$"):
            pw.build()
    assert seen_cmds == []


def test_pin_server_sha256_default_override_all_sites_and_refusals(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    default_sha = "b4659b3fc74037deb96b2148c455753121fa21dde174d7124547e2bf9ffd37c3"
    g4_sha = "39742d8a0354d4577da8dcc9cb3a5b9f03f02593d106eff41297e952d6ebfc95"
    other_sha = "1111111111111111111111111111111111111111111111111111111111111111"
    assert pw.SERVER_SHA256 == default_sha

    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    pack = content / "pack"
    pack.mkdir()
    (pack / "pack.json").write_text("{}", encoding="utf-8")
    (content / "card-values.json").write_text("{}", encoding="utf-8")
    machine_file = work / "machine.json"
    machine_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    rs = _role_set_cmd()
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: None)
    monkeypatch.setattr(pw, "prefetch_hosted", lambda *a, **kw: None)

    seen_critic_cmds: list[list[str]] = []
    seen_pilot_commons: list[list[str]] = []
    seen_sh_cmds: list[str] = []

    def fake_run_proc(cmd, **kw):
        seen_critic_cmds.append(list(cmd))
        return CompletedProcess(cmd, 0, "ok", "")

    def fake_pilot_batch(rs_arg, common, plan, out_dir, writers, critic_for, pool, **kw):
        seen_pilot_commons.append(list(common))
        return {w: {"waves": [{"wave": "write", "exit": 0}], "last": None} for w in writers}

    def fake_sh(cmd: str, **kw) -> str:
        seen_sh_cmds.append(cmd)
        if "pip freeze" in cmd:
            return f"nvidia-cublas=={pw.CUBLAS_VERSION}\nnvidia-cuda-runtime=={pw.NVCC_VERSION}"
        if "pack verify" in cmd or cmd.startswith("tar "):
            return "ok"
        if "plan --lessons" in cmd:
            out_idx = cmd.split().index("--out") + 1
            out_p = Path(cmd.split()[out_idx])
            out_p.write_text(
                json.dumps({"slots": [{"id": "T02-C-01", "lesson": "T02", "kind": "C"}], "exclusion_state_sha256": "e" * 64}),
                encoding="utf-8",
            )
            return "planned"
        return "ok"

    monkeypatch.setattr(pw, "run_proc", fake_run_proc)
    monkeypatch.setattr(pw, "pilot_batch", fake_pilot_batch)
    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setattr(pw, "machine_line", lambda **kw: {"gpu": "A100"})
    monkeypatch.setattr(pw, "read_token_file", lambda **kw: None)
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "gemma-4-31b-it")
    monkeypatch.setenv("PIN_PILOT_CRITIC", "none")
    monkeypatch.setenv("PIN_PILOT_WAVES", "1")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    monkeypatch.setenv("PIN_PILOT_LESSONS", "T02")
    monkeypatch.setenv("PIN_PILOT_SLOT_IDS", "T02-C-01")

    server_path = work / "llama-server"
    server_path.write_bytes(b"fake-server")
    current_binary_sha = default_sha
    monkeypatch.setattr(pw, "sha256", lambda p: current_binary_sha if Path(p) == server_path else "0" * 64)

    # 1. Unset gives default b4659b3f at all four sites + pin-result.json records server_sha256_override=False
    monkeypatch.delenv("PIN_SERVER_SHA256", raising=False)
    assert pw.server_sha256() == default_sha

    seen_critic_cmds.clear()
    pw.critic_control(rs, {"served_as": "muse-glimmer-30b", "gguf_sha256": "c" * 64}, machine_file, 0, None)
    cc_cmd = seen_critic_cmds[-1]
    assert cc_cmd[cc_cmd.index("--llama-server-sha256") + 1] == default_sha

    seen_pilot_commons.clear()
    pw.pilot_window({}, {"machine": {"gpu": "A100"}})
    pw_common = seen_pilot_commons[-1]
    assert pw_common[pw_common.index("--llama-server-sha256") + 1] == default_sha

    rt_default = pw.runtime()
    assert rt_default["sha256"] == default_sha and rt_default["sha256_override"] is False
    assert rt_default["commit"] == pw.LLAMA_CPP_COMMIT and rt_default["archs"] == "80;90;100 (SASS only, no PTX)"

    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_STUB", "1")
    rc = pw._main()
    assert rc == 0
    pin_res_default = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert pin_res_default["server_sha256"] == default_sha
    assert pin_res_default["server_sha256_override"] is False
    assert pin_res_default["server_pin"] == {
        "commit": pw.LLAMA_CPP_COMMIT,
        "sha256": default_sha,
        "archs": "80;90;100 (SASS only, no PTX)",
    }
    assert pin_res_default["llama_server"]["sha256"] == default_sha
    assert pin_res_default["llama_server"]["archs"] == "80;90;100 (SASS only, no PTX)"
    assert pin_res_default["llama_server"]["sha256_override"] is False

    # 2. Valid 64-char lowercase hex override reaches all four sites + pin-result.json records server_sha256_override=True and G4 server_pin
    monkeypatch.setenv("PIN_SERVER_SHA256", g4_sha)
    assert pw.server_sha256() == g4_sha

    seen_critic_cmds.clear()
    pw.critic_control(rs, {"served_as": "muse-glimmer-30b", "gguf_sha256": "c" * 64}, machine_file, 0, None)
    cc_cmd_ov = seen_critic_cmds[-1]
    assert cc_cmd_ov[cc_cmd_ov.index("--llama-server-sha256") + 1] == g4_sha

    seen_pilot_commons.clear()
    pw.pilot_window({}, {"machine": {"gpu": "G4"}})
    pw_common_ov = seen_pilot_commons[-1]
    assert pw_common_ov[pw_common_ov.index("--llama-server-sha256") + 1] == g4_sha

    # 3. runtime() refuses a binary whose sha differs from the override (including window 1's default binary) or an unlisted sha, and accepts matching G4 sha
    current_binary_sha = default_sha
    with pytest.raises(RuntimeError, match=f"llama-server {default_sha[:12]} isn't pinned {g4_sha[:12]}"):
        pw.runtime()
    current_binary_sha = other_sha
    with pytest.raises(RuntimeError, match=f"llama-server {other_sha[:12]} isn't pinned {g4_sha[:12]}"):
        pw.runtime()
    current_binary_sha = g4_sha
    rt_ov = pw.runtime()
    assert rt_ov["sha256"] == g4_sha and rt_ov["sha256_override"] is True
    assert rt_ov["commit"] == pw.LLAMA_CPP_COMMIT and rt_ov["archs"] == "80;90;100;120 (sm_120a SASS)"

    rc_ov = pw._main()
    assert rc_ov == 0
    pin_res_ov = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert pin_res_ov["server_sha256"] == g4_sha
    assert pin_res_ov["server_sha256_override"] is True
    assert pin_res_ov["server_pin"] == {
        "commit": pw.LLAMA_CPP_COMMIT,
        "sha256": g4_sha,
        "archs": "80;90;100;120 (sm_120a SASS)",
    }
    assert pin_res_ov["llama_server"]["sha256"] == g4_sha
    assert pin_res_ov["llama_server"]["archs"] == "80;90;100;120 (sm_120a SASS)"
    assert pin_res_ov["llama_server"]["sha256_override"] is True
    assert seen_pilot_commons[-1][seen_pilot_commons[-1].index("--llama-server-sha256") + 1] == g4_sha

    # Unlisted 64-hex sha is refused by server_pin_entry before work
    monkeypatch.setenv("PIN_SERVER_SHA256", other_sha)
    with pytest.raises(RuntimeError, match=f"REFUSED: llama-server {other_sha[:12]} is not a pinned build in writers.json"):
        pw.runtime()
    rc_unlisted = pw._main()
    assert rc_unlisted == 1
    pin_res_unlisted = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert pin_res_unlisted["ok"] is False
    assert f"REFUSED: llama-server {other_sha[:12]} is not a pinned build in writers.json" in pin_res_unlisted["error"]

    # 4. Malformed values (63 hex, 65 hex, uppercase, non-hex, empty) are refused with REFUSED: before any work
    for bad in (
        "",
        "a" * 63,
        "a" * 65,
        g4_sha.upper(),
        "A" + "a" * 63,
        "g" + "a" * 63,
        "b4659b3f-not-hex" + "0" * 48,
        g4_sha + "\n",
    ):
        monkeypatch.setenv("PIN_SERVER_SHA256", bad)
        seen_sh_cmds.clear()
        with pytest.raises(RuntimeError, match=r"REFUSED: PIN_SERVER_SHA256=.* must be 64 lowercase hex characters"):
            pw.server_sha256()
        with pytest.raises(RuntimeError, match=r"REFUSED: PIN_SERVER_SHA256=.* must be 64 lowercase hex characters"):
            pw.runtime()
        assert seen_sh_cmds == []
        rc_bad = pw._main()
        assert rc_bad == 1
        assert seen_sh_cmds == []
        pin_res_bad = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
        assert pin_res_bad["ok"] is False
        assert "REFUSED: PIN_SERVER_SHA256=" in pin_res_bad["error"]


# ---- PIN_ROLE=offrig-bench (R&D, 2026-10-23): Ollama 0.40.4 + offrig verify calibrate cross-rig baseline -----------------


from stub_llama import FAKE_OFFRIG_PY  # noqa: E402


def _write_offrig_fixtures(content: Path) -> tuple[Path, str, Path, str]:
    offrig_bin = content / "offrig"
    offrig_bin.write_text(FAKE_OFFRIG_PY, encoding="utf-8", newline="\n")
    offrig_bin.chmod(0o755)
    gold = content / "bench30.jsonl"
    gold_lines = [
        json.dumps({"id": "c-01", "claim": "West pier opens at 06:00.", "label": "unsupported", "split": "tune"}),
        json.dumps({"id": "c-02", "claim": "East pier closes at 22:00.", "label": "unsupported", "split": "tune"}),
    ]
    gold.write_text("\n".join(gold_lines) + "\n", encoding="utf-8", newline="\n")
    return offrig_bin, pw.sha256(offrig_bin), gold, pw.sha256(gold)


def test_offrig_bench_inputs_and_install_ollama_pin_checks_and_refusals(tmp_path, monkeypatch):
    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "say", lambda *a: None)

    assert pw.OLLAMA_VERSION == "0.40.4"
    assert pw.OLLAMA_SHA256 == "15f31e9576d15a481e099b4145dba6e9f8c1f295927d6b8895300ae30ed3740a"
    assert pw.OFFRIG_MODEL == "gemma4:31b"
    assert pw.OFFRIG_MODEL_DIGEST == "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"
    assert pw.OFFRIG_GOLD_SHA256 == "0c88ec0f915c911a99d3fcf2d2e91104c0c65b0f279656f27afa2244099ce547"
    assert pw.SMI_INTERVAL_S == 2.0

    _offrig_bin, bin_sha, gold, gold_sha = _write_offrig_fixtures(content)

    # 1. Missing or malformed PIN_OFFRIG_SHA256 / PIN_OFFRIG_GOLD_SHA256 refuses
    monkeypatch.delenv("PIN_OFFRIG_SHA256", raising=False)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OFFRIG_SHA256 must be a 64-char hex sha256"):
        pw.offrig_bench_inputs({})

    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", "not-a-sha")
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OFFRIG_GOLD_SHA256 must be a 64-char hex sha256"):
        pw.offrig_bench_inputs({})

    # 1b. F4: When PIN_OFFRIG_GOLD_SHA256 is unset and gold is bench30.jsonl, OFFRIG_GOLD_SHA256 is the default and is checked:
    # fixture bench30.jsonl has gold_sha != OFFRIG_GOLD_SHA256, so it refuses with sha mismatch.
    monkeypatch.delenv("PIN_OFFRIG_GOLD_SHA256", raising=False)
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: offrig gold bench30\.jsonl sha256 mismatch: {gold_sha[:12]} != PIN_OFFRIG_GOLD_SHA256 {pw.OFFRIG_GOLD_SHA256[:12]}",
    ):
        pw.offrig_bench_inputs({})

    # And when gold is not bench30.jsonl, an unset PIN_OFFRIG_GOLD_SHA256 refuses before reading the file.
    other_gold = content / "custom.jsonl"
    other_gold.write_text(gold.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    monkeypatch.setenv("PIN_OFFRIG_GOLD", str(other_gold))
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OFFRIG_GOLD_SHA256 must be a 64-char hex sha256"):
        pw.offrig_bench_inputs({})
    monkeypatch.delenv("PIN_OFFRIG_GOLD", raising=False)

    # When bench30.jsonl matches OFFRIG_GOLD_SHA256 and PIN_OFFRIG_GOLD_SHA256 is unset, it stages cleanly.
    monkeypatch.setattr(pw, "OFFRIG_GOLD_SHA256", gold_sha)
    res_default_gold: dict = {}
    pw.offrig_bench_inputs(res_default_gold)
    assert res_default_gold["offrig_bench_inputs"]["gold_sha256"] == gold_sha
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    monkeypatch.setattr(pw, "OFFRIG_GOLD_SHA256", "0c88ec0f915c911a99d3fcf2d2e91104c0c65b0f279656f27afa2244099ce547")

    # 2. Binary or gold sha256 mismatch refuses before anything runs
    monkeypatch.setenv("PIN_OFFRIG_SHA256", "0" * 64)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    with pytest.raises(RuntimeError, match=rf"REFUSED: offrig binary sha256 mismatch: {bin_sha[:12]} != PIN_OFFRIG_SHA256 000000000000"):
        pw.offrig_bench_inputs({})

    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", "f" * 64)
    with pytest.raises(RuntimeError, match=rf"REFUSED: offrig gold bench30\.jsonl sha256 mismatch: {gold_sha[:12]} != PIN_OFFRIG_GOLD_SHA256 ffffffffffff"):
        pw.offrig_bench_inputs({})

    # 3. Matching shas stage read-only inputs and record offrig_bench_inputs
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    res_in: dict = {}
    staged_bin, staged_gold, meta = pw.offrig_bench_inputs(res_in)
    assert staged_bin.is_file() and staged_gold.is_file()
    assert meta == {"offrig_sha256": bin_sha, "gold": "bench30.jsonl", "gold_sha256": gold_sha}
    assert res_in["offrig_bench_inputs"] == meta
    assert not os.access(staged_gold, os.W_OK)

    # 4. install_ollama refuses wrong version or bad sha256 before unpacking or running
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)

    monkeypatch.setenv("PIN_OLLAMA_VERSION", "0.39.0")
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OLLAMA_VERSION must be '0.40.4'"):
        pw.install_ollama({})
    monkeypatch.delenv("PIN_OLLAMA_VERSION", raising=False)

    fixture_zst = Path(__file__).resolve().parent / "fixtures" / "stub-ollama.tar.zst"
    fake_archive = content / "ollama-linux-amd64.tar.zst"
    shutil.copyfile(fixture_zst, fake_archive)
    arch_sha = pw.sha256(fake_archive)
    monkeypatch.setenv("PIN_OLLAMA_SHA256", "1" * 64)
    with pytest.raises(RuntimeError, match=rf"REFUSED: ollama archive sha256 mismatch: {arch_sha[:12]} != PIN_OLLAMA_SHA256 111111111111"):
        pw.install_ollama({})

    # 5. install_ollama unpacks archive in Python with pinned zstandard when sha matches and checks `ollama --version` == 0.40.4
    monkeypatch.setenv("PIN_OLLAMA_SHA256", arch_sha)
    ver_out = "ollama version is 0.40.4"
    pip_cmds: list[str] = []

    def fake_sh(cmd: str, **kw) -> str:
        if "pip install -q --require-hashes --only-binary=:all: --no-deps -r" in cmd:
            pip_cmds.append(cmd)
            return ""
        if cmd.endswith("--version"):
            return ver_out
        raise AssertionError(f"unexpected sh: {cmd}")

    monkeypatch.setattr(pw, "sh", fake_sh)
    res_ol: dict = {}
    installed_bin = pw.install_ollama(res_ol)
    assert len(pip_cmds) == 1
    assert installed_bin == work / "ollama-install" / "bin" / "ollama"
    assert (work / "ollama-install" / "lib" / "ollama" / "stub.txt").is_file()
    wheel_sha = pw.zstandard_wheel_sha256()
    assert (work / "zstandard-requirements.txt").read_text(encoding="utf-8") == (
        f"zstandard=={pw.ZSTANDARD_VERSION} --hash=sha256:{wheel_sha}\n"
    )
    assert res_ol["ollama"] == {
        "version": "0.40.4",
        "sha256": arch_sha,
        "bin": str(installed_bin),
        "version_line": "ollama version is 0.40.4",
        "zstandard": {"version": "0.25.0", "wheel_sha256": wheel_sha},
        "zstandard_version": "0.25.0",
        "zstandard_wheel_sha256": wheel_sha,
    }

    # Wrong version reported by binary refuses
    ver_out = "ollama version is 0.35.0"
    with pytest.raises(RuntimeError, match="REFUSED: ollama --version is 'ollama version is 0.35.0', expected 0.40.4"):
        pw.install_ollama({})


def test_offrig_bench_token_rules_unchanged_and_residue_refused(tmp_path, monkeypatch):
    import socket as _socket
    import tarfile

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    shm = tmp_path / "shm"
    shm.mkdir()
    tok_file = shm / "hf.token"
    sentinel = "hf_" + "SENTINELneverPERSIST9f3c2a71abcdEFGHijk"
    tok_file.write_text(sentinel + "\n", encoding="utf-8")

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", tok_file)
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_DEADLINE", "PIN_STUB_CALL_S", "PIN_GGUF_REPO"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    monkeypatch.setenv("PIN_STUB", "1")

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(pw, "OLLAMA_PORT", free_port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.05)

    _offrig_bin, bin_sha, _gold, gold_sha = _write_offrig_fixtures(content)
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    orig_run_proc = pw.run_proc

    def cross_platform_run_proc(cmd, **kw):
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("offrig") and os.name == "nt":
            cmd = [sys.executable, str(cmd[0]), *cmd[1:]]
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", cross_platform_run_proc)

    def tar_sh(cmd: str, **kw) -> str:
        if cmd.startswith(f"tar -C {work} -cf {work}/offrig-bench-out.tar offrig-bench"):
            with tarfile.open(work / "offrig-bench-out.tar", "w") as tf:
                tf.add(work / "offrig-bench", arcname="offrig-bench")
            return ""
        return pw.sh(cmd, **kw)

    monkeypatch.setattr(pw, "sh", tar_sh)

    # 1. Full _main() run with role == "offrig-bench": needs_token is False, tok_file is deleted, _TOKEN stays None,
    # and the sentinel never appears in pin-result.json.
    assert pw._main() == 0
    assert not tok_file.exists() and pw._TOKEN is None
    pin_res_text = (work / "pin-result.json").read_text(encoding="utf-8")
    assert sentinel not in pin_res_text
    pin_res = json.loads(pin_res_text)
    assert pin_res["ok"] is True
    assert pin_res["role"] == "offrig-bench"
    assert all(pin_res["token_clean"].values())
    assert pin_res["offrig_bench"]["verdicts_count"] == 2

    # 2. Full _main() run with token residue before offrig_bench_window: stops before inputs or Ollama setup and exits 1.
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    hf_home = tmp_path / "hf_home"
    hf_home.mkdir()
    (hf_home / "token").write_text("residue\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    inputs_called = False

    def fail_if_inputs_read(*a):
        nonlocal inputs_called
        inputs_called = True
        raise AssertionError("must not read inputs when token_clean fails")

    monkeypatch.setattr(pw, "offrig_bench_inputs", fail_if_inputs_read)
    assert pw._main() == 1
    assert not inputs_called
    res_fail = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert res_fail["ok"] is False
    assert "token residue is left before any wave" in res_fail["error"] and "no_cached_token" in res_fail["error"]


def test_offrig_bench_stub_mode_end_to_end_and_real_path_checks(tmp_path, monkeypatch):
    import socket as _socket
    import tarfile

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    shm = tmp_path / "shm"
    shm.mkdir()

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", shm / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_DEADLINE", "PIN_STUB_CALL_S"):
        monkeypatch.delenv(k, raising=False)

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(pw, "OLLAMA_PORT", free_port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.05)

    _offrig_bin, bin_sha, _gold, gold_sha = _write_offrig_fixtures(content)
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    orig_run_proc = pw.run_proc
    seen_calibrate_cmds: list[list[str]] = []

    def cross_platform_run_proc(cmd, **kw):
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("offrig"):
            seen_calibrate_cmds.append(list(cmd))
            if os.name == "nt":
                cmd = [sys.executable, str(cmd[0]), *cmd[1:]]
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", cross_platform_run_proc)

    def tar_sh(cmd: str, **kw) -> str:
        if cmd.startswith(f"tar -C {work} -cf {work}/offrig-bench-out.tar offrig-bench"):
            with tarfile.open(work / "offrig-bench-out.tar", "w") as tf:
                tf.add(work / "offrig-bench", arcname="offrig-bench")
            return ""
        return pw.sh(cmd, **kw)

    monkeypatch.setattr(pw, "sh", tar_sh)

    last_stub_holder: dict = {}
    orig_start_stub_ollama = pw._start_stub_ollama

    def capture_stub_ollama(model, digest, port, version="0.40.4"):
        s = orig_start_stub_ollama(model, digest, port, version)
        last_stub_holder["stub"] = s
        return s

    monkeypatch.setattr(pw, "_start_stub_ollama", capture_stub_ollama)

    # 1. Stub mode (PIN_STUB=1) runs the full path against StubLlama on CPU and warms up with num_ctx=8192
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    res_stub: dict = {"machine": pw.machine_line(), "role": "offrig-bench", "stub": True}
    pw.offrig_bench_window(res_stub)

    gen_calls = [b for path, b in last_stub_holder["stub"].ollama_requests if path == "/api/generate"]
    assert gen_calls[0] == {
        "model": "gemma4:31b",
        "prompt": "",
        "keep_alive": "60m",
        "stream": False,
        "options": {"num_ctx": 8192},
    }
    assert res_stub["model_errors"] == 0
    assert all(res_stub["token_clean"].values())
    assert res_stub["ollama"]["version"] == "0.40.4" and res_stub["ollama"]["stub"] is True
    assert res_stub["ollama"]["sha256"] == pw.STUB_OLLAMA_SHA256
    assert res_stub["ollama"]["zstandard_version"] == "0.25.0"
    assert res_stub["ollama"]["zstandard_wheel_sha256"] == pw.zstandard_wheel_sha256()
    assert res_stub["ollama"]["zstandard"] == {"version": "0.25.0", "wheel_sha256": pw.zstandard_wheel_sha256()}
    assert (work / "ollama-install" / "bin" / "ollama").is_file()
    assert (work / "ollama-install" / "lib" / "ollama" / "stub.txt").is_file()
    assert res_stub["offrig_model"]["model"] == "gemma4:31b"
    assert res_stub["offrig_model"]["digest"] == pw.OFFRIG_MODEL_DIGEST
    assert len(res_stub["ps"]["models"]) == 1
    assert res_stub["ps"]["models"][0]["context_length"] == 8192
    assert res_stub["ps"]["models"][0]["size_vram"] == res_stub["ps"]["models"][0]["size"]
    assert res_stub["ps_end"]["models"][0]["context_length"] == 8192
    assert res_stub["ps_end"]["models"][0]["size_vram"] == res_stub["ps"]["models"][0]["size_vram"]
    assert (work / "ps.json").is_file()
    assert (work / "offrig-bench-out.tar").is_file()
    assert res_stub["smi"]["samples"] >= 1
    assert res_stub["offrig_bench"]["verdicts_count"] == 2
    assert res_stub["offrig_bench"]["stub"] is True
    assert res_stub["offrig_bench"]["stdout_tail"].startswith("# STUB RUN (PIN_STUB=1)\n")

    assert len(seen_calibrate_cmds) == 1
    cal_cmd = seen_calibrate_cmds[0]
    assert cal_cmd[1:13] == [
        "verify",
        "calibrate",
        str(work / "offrig-bench-in" / "bench30.jsonl"),
        "--model",
        "gemma4:31b",
        "--think",
        "on",
        "--structured",
        "on",
        "--split",
        "tune",
        "--jobs",
    ]
    assert cal_cmd[13] == "1"

    with tarfile.open(work / "offrig-bench-out.tar") as tf:
        names = set(tf.getnames())
    for expected in (
        "offrig-bench/ps.json",
        "offrig-bench/ps-end.json",
        "offrig-bench/smi.csv",
        "offrig-bench/offrig.log",
        "offrig-bench/out/verdicts.jsonl",
        "offrig-bench/out/manifest.json",
        "offrig-bench/out/metrics.json",
    ):
        assert expected in names, (expected, names)
    manifest = json.loads((work / "offrig-bench" / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stub"] is True and manifest["jobs"] == 1 and manifest["think"] == "on" and manifest["structured"] == "on"

    # 2. Digest mismatch in /api/tags refuses before warm-up or calibrate
    seen_calibrate_cmds.clear()
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    monkeypatch.setattr(
        pw,
        "_start_stub_ollama",
        lambda model, digest, port, version="0.40.4": __import__("stub_llama").StubLlama(
            model, model, port=port, ollama_model=model, ollama_digest="a" * 64, ollama_version=version
        ),
    )
    with pytest.raises(RuntimeError, match="REFUSED: model gemma4:31b digest aaaaaaaaaaaa != PIN_OFFRIG_MODEL_DIGEST 6316f0629137"):
        pw.offrig_bench_window({})
    assert seen_calibrate_cmds == []

    # 2b. F4: Unconditional digest check in warmup_ollama_model (/api/ps): refuses when /api/ps digest is empty or wrong
    for bad_ps_digest in ("", "b" * 64):
        for p in sorted((work / "offrig-bench-in").rglob("*")):
            os.chmod(p, 0o700)
        os.chmod(work / "offrig-bench-in", 0o700)

        def _bad_ps_digest_stub(model, digest, port, version="0.40.4", _bad=bad_ps_digest):
            s = __import__("stub_llama").StubLlama(
                model, model, port=port, ollama_model=model, ollama_digest=digest, ollama_version=version
            )
            s.ollama_on_generate = lambda _body: setattr(s, "ollama_digest", _bad)
            return s

        monkeypatch.setattr(pw, "_start_stub_ollama", _bad_ps_digest_stub)
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: /api/ps model digest {bad_ps_digest[:12]} != PIN_OFFRIG_MODEL_DIGEST 6316f0629137",
        ):
            pw.offrig_bench_window({})
        assert seen_calibrate_cmds == []

    # 3. Offloaded model in /api/ps (size_vram < size) refuses before calibrate
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)

    def _offloaded_stub(model, digest, port, version="0.40.4"):
        s = __import__("stub_llama").StubLlama(
            model, model, port=port, ollama_model=model, ollama_digest=digest, ollama_version=version
        )
        s.ollama_size_vram = s.ollama_size - 1024
        return s

    monkeypatch.setattr(pw, "_start_stub_ollama", _offloaded_stub)
    with pytest.raises(RuntimeError, match="REFUSED: gemma4:31b offloaded in /api/ps: size_vram"):
        pw.offrig_bench_window({})
    assert seen_calibrate_cmds == []

    # 4. F2: Post-run /api/ps check refuses (records error and exits non-zero) if context_length or size_vram changed during offrig run
    for mutate_kind, expected_err in (
        ("ctx", "REFUSED: /api/ps context_length changed during offrig run: warm-up 8192 != post-run 4096"),
        ("vram", "REFUSED: /api/ps size_vram changed during offrig run: warm-up 20281829620 != post-run 20281828596"),
    ):
        for p in sorted((work / "offrig-bench-in").rglob("*")):
            os.chmod(p, 0o700)
        os.chmod(work / "offrig-bench-in", 0o700)

        def _mutating_stub(model, digest, port, version="0.40.4", _kind=mutate_kind):
            s = __import__("stub_llama").StubLlama(
                model, model, port=port, ollama_model=model, ollama_digest=digest, ollama_version=version
            )
            if _kind == "ctx":
                s.ollama_on_chat = lambda _body: setattr(s, "ollama_context_length", 4096)
            else:
                s.ollama_on_chat = lambda _body: setattr(s, "ollama_size_vram", s.ollama_size - 1024)
            return s

        monkeypatch.setattr(pw, "_start_stub_ollama", _mutating_stub)
        assert pw._main() == 1
        post_res = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
        assert post_res["ok"] is False
        assert expected_err in post_res["error"]


def test_ollama_served_non_stub_popen_env_and_refusals_and_inline_stub_fallback(tmp_path, monkeypatch):
    import socket as _socket
    import tarfile

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    shm = tmp_path / "shm"
    shm.mkdir()

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", shm / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_STUB", "PIN_DEADLINE", "PIN_STUB_CALL_S"):
        monkeypatch.delenv(k, raising=False)

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]

    fake_ollama_bin = work / "ollama-install" / "bin" / "ollama"
    (work / "ollama-install" / "lib" / "ollama").mkdir(parents=True)
    fake_ollama_bin.parent.mkdir(parents=True, exist_ok=True)
    fake_ollama_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    popen_calls: list[tuple[list[str], dict]] = []
    stopped_procs: list[object] = []

    class FakeProc:
        returncode = None

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        popen_calls.append((list(cmd), kw))
        return FakeProc()

    def fake_stop_server(proc, log_fh):
        stopped_procs.append(proc)
        log_fh.close()

    orig_popen = pw.subprocess.Popen
    monkeypatch.setattr(pw.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pw, "stop_server", fake_stop_server)

    # 1. F1 & F3: OllamaServed.__enter__ non-stub path passes exact env with NO OLLAMA_FLASH_ATTENTION key
    # (even if OLLAMA_FLASH_ATTENTION is present in os.environ)
    monkeypatch.setenv("OLLAMA_FLASH_ATTENTION", "auto")
    bg_stub = __import__("stub_llama").StubLlama(
        pw.OFFRIG_MODEL,
        pw.OFFRIG_MODEL,
        port=free_port,
        ollama_model=pw.OFFRIG_MODEL,
        ollama_digest=pw.OFFRIG_MODEL_DIGEST,
        ollama_version=pw.OLLAMA_VERSION,
    )
    try:
        with pw.OllamaServed(fake_ollama_bin, pw.OFFRIG_MODEL, pw.OFFRIG_MODEL_DIGEST, free_port) as srv:
            assert srv.log == work / "ollama-serve.log"
        assert len(popen_calls) == 1
        cmd, kw = popen_calls[0]
        assert cmd == [str(fake_ollama_bin), "serve"]
        env = kw["env"]
        assert "OLLAMA_FLASH_ATTENTION" not in env
        assert env["OLLAMA_HOST"] == f"127.0.0.1:{free_port}"
        assert env["OLLAMA_NUM_PARALLEL"] == "1"
        assert env["OLLAMA_KV_CACHE_TYPE"] == "f16"
        assert env["OLLAMA_MODELS"] == str(work / "ollama-models")
        assert str(work / "ollama-install" / "lib" / "ollama") in env["LD_LIBRARY_PATH"]
        assert len(stopped_procs) == 1

        # 2. F3: /api/version mismatch in OllamaServed.__enter__ stops server and refuses
        bg_stub.ollama_version = "0.39.0"
        stopped_procs.clear()
        with (
            pytest.raises(RuntimeError, match="REFUSED: ollama /api/version is '0.39.0', not '0.40.4'"),
            pw.OllamaServed(fake_ollama_bin, pw.OFFRIG_MODEL, pw.OFFRIG_MODEL_DIGEST, free_port),
        ):
            pass
        assert len(stopped_procs) == 1

        # 3. F3: non-empty /api/ps at start in OllamaServed.__enter__ stops server and refuses
        bg_stub.ollama_version = pw.OLLAMA_VERSION
        bg_stub.ollama_loaded = True
        stopped_procs.clear()
        with (
            pytest.raises(RuntimeError, match="REFUSED: ollama card not empty at start:"),
            pw.OllamaServed(fake_ollama_bin, pw.OFFRIG_MODEL, pw.OFFRIG_MODEL_DIGEST, free_port),
        ):
            pass
        assert len(stopped_procs) == 1
    finally:
        bg_stub.close()
    monkeypatch.delenv("OLLAMA_FLASH_ATTENTION", raising=False)
    monkeypatch.setattr(pw.subprocess, "Popen", orig_popen)

    # 4. F3: _InlineStubOllama fallback when stub_llama cannot be imported runs offrig_bench_window end-to-end
    monkeypatch.setitem(sys.modules, "stub_llama", None)
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    monkeypatch.setattr(pw, "OLLAMA_PORT", free_port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.05)

    _offrig_bin, bin_sha, _gold, gold_sha = _write_offrig_fixtures(content)
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    orig_run_proc = pw.run_proc

    def cross_platform_run_proc(cmd, **kw):
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("offrig") and os.name == "nt":
            cmd = [sys.executable, str(cmd[0]), *cmd[1:]]
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", cross_platform_run_proc)

    def tar_sh(cmd: str, **kw) -> str:
        if cmd.startswith(f"tar -C {work} -cf {work}/offrig-bench-out.tar offrig-bench"):
            with tarfile.open(work / "offrig-bench-out.tar", "w") as tf:
                tf.add(work / "offrig-bench", arcname="offrig-bench")
            return ""
        return pw.sh(cmd, **kw)

    monkeypatch.setattr(pw, "sh", tar_sh)

    inline_srv = pw._start_stub_ollama(pw.OFFRIG_MODEL, pw.OFFRIG_MODEL_DIGEST, 0)
    try:
        assert type(inline_srv).__name__ == "_InlineStubOllama"
    finally:
        inline_srv.close()

    res_inline: dict = {}
    pw.offrig_bench_window(res_inline)
    assert res_inline["model_errors"] == 0
    assert res_inline["ps"]["models"][0]["context_length"] == 8192
    assert res_inline["ps_end"]["models"][0]["context_length"] == 8192
    assert res_inline["offrig_bench"]["verdicts_count"] == 2


def test_offrig_bench_real_mode_env_smi_sampling_and_deadline_cut(tmp_path, monkeypatch):
    import socket as _socket
    import tarfile
    import time as _t

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    shm = tmp_path / "shm"
    shm.mkdir()

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", shm / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PIN_STUB", "PIN_DEADLINE", "PIN_STUB_CALL_S"):
        monkeypatch.delenv(k, raising=False)

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(pw, "OLLAMA_PORT", free_port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.005)

    _offrig_bin, bin_sha, _gold, gold_sha = _write_offrig_fixtures(content)
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    fake_ollama_bin = work / "ollama-install" / "bin" / "ollama"
    fake_ollama_bin.parent.mkdir(parents=True)
    fake_ollama_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(
        pw,
        "install_ollama",
        lambda res: res.__setitem__("ollama", {"version": "0.40.4", "sha256": pw.OLLAMA_SHA256, "bin": str(fake_ollama_bin)}) or fake_ollama_bin,
    )

    # Stand in OllamaServed.__enter__/__exit__ with StubLlama while verifying pull_ollama_model runs `ollama pull`
    stub_holder: dict = {}

    def fake_enter(self):
        stub_holder["srv"] = __import__("stub_llama").StubLlama(
            self.model,
            self.model,
            port=self.port,
            ollama_model=self.model,
            ollama_digest=self.digest,
            ollama_version=pw.OLLAMA_VERSION,
        )
        return self

    def fake_exit(self, *exc):
        if stub_holder.get("srv") is not None:
            stub_holder["srv"].close()
            stub_holder["srv"] = None

    monkeypatch.setattr(pw.OllamaServed, "__enter__", fake_enter)
    monkeypatch.setattr(pw.OllamaServed, "__exit__", fake_exit)

    smi_samples = iter([
        "2026/10/23 00:00:01.000, 21450 MiB, 82 %, 58, 265.10 W\n",
        "2026/10/23 00:00:03.000, 24120 MiB, 94 %, 61, 298.40 W\n",
    ])
    third_sample_requested = threading.Event()  # the third reading means the second was fully recorded

    def fake_smi() -> str:
        sample = next(smi_samples, None)
        if sample is None:
            third_sample_requested.set()
            return "2026/10/23 00:00:05.000, 23800 MiB, 90 %, 60, 280.00 W\n"
        return sample

    monkeypatch.setattr(pw, "_query_smi_csv", fake_smi)

    def tar_sh(cmd: str, **kw) -> str:
        if cmd.startswith(f"tar -C {work} -cf {work}/offrig-bench-out.tar offrig-bench"):
            with tarfile.open(work / "offrig-bench-out.tar", "w") as tf:
                tf.add(work / "offrig-bench", arcname="offrig-bench")
            return ""
        raise AssertionError(f"unexpected sh: {cmd}")

    monkeypatch.setattr(pw, "sh", tar_sh)

    orig_run_proc = pw.run_proc
    seen_proc_calls: list[tuple[list[str], dict]] = []

    def real_mode_run_proc(cmd, **kw):
        seen_proc_calls.append((list(cmd), kw))
        if cmd[1:3] == ["pull", "gemma4:31b"]:
            assert kw["env"]["OLLAMA_HOST"] == f"127.0.0.1:{free_port}"
            return subprocess.CompletedProcess(cmd, 0, "pulling manifest\nsuccess\n", "")
        if os.name == "nt" and str(cmd[0]).endswith("offrig"):
            cmd = [sys.executable, str(cmd[0]), *cmd[1:]]
        if cmd[1:2] == ["verify"]:
            # Deterministic, not timing-based: the fake offrig can return before the sampler has taken both readings.
            assert third_sample_requested.wait(30), "the sampler never took its third reading"
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", real_mode_run_proc)

    res_real: dict = {}
    pw.offrig_bench_window(res_real)
    assert [c[0][1] for c in seen_proc_calls] == ["pull", "verify"]
    assert res_real["smi"]["peak_vram_mib"] == 24120
    assert res_real["smi"]["samples"] >= 1
    assert "stub" not in res_real["offrig_bench"]
    assert not res_real["offrig_bench"]["stdout_tail"].startswith("# STUB")

    # Deadline cut during offrig verify calibrate records CUT_AT and still tars offrig-bench-out.tar in finally
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    monkeypatch.setenv("PIN_DEADLINE", str(int(_t.time() + 20)))

    def deadline_run_proc(cmd, **kw):
        if cmd[1:3] == ["pull", "gemma4:31b"]:
            return subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        assert kw["term_first"] is True and kw["timeout"] <= 20
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(pw, "run_proc", deadline_run_proc)
    with pytest.raises(pw.DeadlineReached, match="offrig-bench calibrate run was cut at the window's deadline"):
        pw.offrig_bench_window({})
    assert pw.CUT_AT["wave"] == "offrig-bench" and pw.CUT_AT["model"] == "gemma4:31b"
    assert (work / "offrig-bench-out.tar").is_file() and (work / "ps.json").is_file()


def _make_zst_tar(path: Path, members: list[tuple[str, bytes | None, int, str | None, bytes]]) -> str:
    """Create a `.tar.zst` archive with arbitrary raw member names/links for testing `_safe_tar_filter`."""

    import io
    import tarfile
    import zstandard

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data, mode, linkname, ttype in members:
            ti = tarfile.TarInfo(name)
            ti.mode = mode
            ti.mtime = 0
            ti.type = ttype
            if linkname is not None:
                ti.linkname = linkname
                tf.addfile(ti)
            else:
                payload = data or b""
                ti.size = len(payload)
                tf.addfile(ti, io.BytesIO(payload))
    path.write_bytes(zstandard.ZstdCompressor(level=3).compress(buf.getvalue()))
    return pw.sha256(path)


def test_install_ollama_zstd_absent_tampered_archive_unsafe_members_and_zstandard_pin(tmp_path, monkeypatch):
    import importlib.metadata as _im
    import tarfile

    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir()
    work.mkdir()
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in ("PIN_STUB", "PIN_OLLAMA_VERSION", "PIN_OLLAMA_SHA256", "PIN_OLLAMA_ARCHIVE", "PIN_OLLAMA_URL"):
        monkeypatch.delenv(k, raising=False)

    # 1. Pinned zstandard version and manylinux x86_64 wheel hashes
    assert pw.ZSTANDARD_VERSION == "0.25.0"
    assert pw.ZSTANDARD_WHEEL_SHA256 == "8e735494da3db08694d26480f1493ad2cf86e99bdd53e8e9771b2752a5c0246a"
    assert pw.zstandard_wheel_sha256("cp313") == "8e735494da3db08694d26480f1493ad2cf86e99bdd53e8e9771b2752a5c0246a"
    assert pw.zstandard_wheel_sha256("cp312") == "5a56ba0db2d244117ed744dfa8f6f5b366e14148e00de44723413b2f3938a902"
    with pytest.raises(RuntimeError, match="REFUSED: no pinned zstandard==0.25.0 manylinux x86_64 wheel sha256 for Python cp38"):
        pw.zstandard_wheel_sha256("cp38")

    fixture_zst = Path(__file__).resolve().parent / "fixtures" / "stub-ollama.tar.zst"
    assert fixture_zst.is_file()
    assert pw.sha256(fixture_zst) == pw.STUB_OLLAMA_SHA256

    # 2. Unpack works with `zstd` (and `tar`) absent from PATH (both PATH="" and shutil.which returning None)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(pw.shutil, "which", lambda *_a, **_k: None)
    assert pw.shutil.which("zstd") is None
    assert pw.shutil.which("tar") is None

    arch = content / "ollama-linux-amd64.tar.zst"
    shutil.copyfile(fixture_zst, arch)
    monkeypatch.setenv("PIN_OLLAMA_SHA256", pw.STUB_OLLAMA_SHA256)

    seen_sh: list[str] = []

    def no_zstd_sh(cmd: str, **kw) -> str:
        seen_sh.append(cmd)
        assert not cmd.startswith("tar ")
        if "pip install -q --require-hashes --only-binary=:all: --no-deps -r" in cmd:
            return ""
        if cmd.endswith("--version"):
            return "ollama version is 0.40.4"
        raise AssertionError(f"unexpected sh: {cmd}")

    monkeypatch.setattr(pw, "sh", no_zstd_sh)
    res_ok: dict = {}
    out_bin = pw.install_ollama(res_ok)
    assert out_bin == work / "ollama-install" / "bin" / "ollama"
    assert out_bin.is_file()
    assert "ollama version is 0.40.4" in out_bin.read_text(encoding="utf-8")
    assert (work / "ollama-install" / "lib" / "ollama" / "stub.txt").read_text(encoding="utf-8") == "stub\n"
    assert res_ok["ollama"]["zstandard_version"] == "0.25.0"
    assert res_ok["ollama"]["zstandard_wheel_sha256"] == pw.zstandard_wheel_sha256()
    assert res_ok["ollama"]["zstandard"] == {"version": "0.25.0", "wheel_sha256": pw.zstandard_wheel_sha256()}

    # 2b. Stub mode (including standalone CPU VM fallback when tests/fixtures is not cloned) runs the same unpack step
    arch.unlink()
    monkeypatch.delenv("PIN_OLLAMA_SHA256", raising=False)
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setattr(pw, "_repo_root", lambda: content / "no-repo")
    res_stub_unpack: dict = {}
    stub_bin = pw.install_ollama(res_stub_unpack)
    assert (work / "stub-ollama.tar.zst").read_bytes() == fixture_zst.read_bytes()
    assert stub_bin == work / "ollama-install" / "bin" / "ollama"
    assert stub_bin.is_file()
    assert res_stub_unpack["ollama"]["stub"] is True
    assert res_stub_unpack["ollama"]["sha256"] == pw.STUB_OLLAMA_SHA256
    assert res_stub_unpack["ollama"]["zstandard_version"] == "0.25.0"
    assert res_stub_unpack["ollama"]["zstandard_wheel_sha256"] == pw.zstandard_wheel_sha256()
    monkeypatch.delenv("PIN_STUB", raising=False)
    monkeypatch.setenv("PIN_OLLAMA_SHA256", pw.STUB_OLLAMA_SHA256)

    # 3. Tampered archive is refused by the sha256 check BEFORE any unpacking
    shutil.rmtree(work / "ollama-install", ignore_errors=True)
    tampered = content / "ollama-linux-amd64.tar.zst"
    raw_bytes = fixture_zst.read_bytes()
    tampered.write_bytes(raw_bytes[:-1] + bytes([raw_bytes[-1] ^ 0xFF]))
    tampered_sha = pw.sha256(tampered)
    unpack_called = False
    orig_unpack = pw.unpack_ollama_archive

    def spy_unpack(*a, **kw):
        nonlocal unpack_called
        unpack_called = True
        return orig_unpack(*a, **kw)

    monkeypatch.setattr(pw, "unpack_ollama_archive", spy_unpack)
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: ollama archive sha256 mismatch: {tampered_sha[:12]} != PIN_OLLAMA_SHA256 {pw.STUB_OLLAMA_SHA256[:12]}",
    ):
        pw.install_ollama({})
    assert unpack_called is False
    assert not (work / "ollama-install").exists()
    monkeypatch.setattr(pw, "unpack_ollama_archive", orig_unpack)

    # 4. Archive with a `..` member, an absolute member, or a link outside the target is refused
    escape_sentinel = content / "escaped.txt"
    for bad_members, expected_err in [
        ([("../escaped.txt", b"bad\n", 0o644, None, tarfile.REGTYPE)], r"REFUSED: '\.\.' segment in tar archive member: '\.\./escaped\.txt'"),
        ([("bin/../../escaped.txt", b"bad\n", 0o644, None, tarfile.REGTYPE)], r"REFUSED: '\.\.' segment in tar archive member: 'bin/\.\./\.\./escaped\.txt'"),
        ([("/etc/passwd", b"bad\n", 0o644, None, tarfile.REGTYPE)], r"REFUSED: absolute path in tar archive: '/etc/passwd'"),
        ([("/bin/ollama", b"bad\n", 0o755, None, tarfile.REGTYPE)], r"REFUSED: absolute path in tar archive: '/bin/ollama'"),
        ([("bin/link", None, 0o777, "../../escaped.txt", tarfile.SYMTYPE)], r"REFUSED: link outside destination in tar member 'bin/link' -> '\.\./\.\./escaped\.txt'"),
        ([("bin/abslink", None, 0o777, "/etc/passwd", tarfile.SYMTYPE)], r"REFUSED: absolute link target in tar member 'bin/abslink': '/etc/passwd'"),
        ([("bin/hardlink", None, 0o644, "../escaped.txt", tarfile.LNKTYPE)], r"REFUSED: link outside destination in tar member 'bin/hardlink' -> '\.\./escaped\.txt'"),
    ]:
        bad_sha = _make_zst_tar(arch, bad_members)
        monkeypatch.setenv("PIN_OLLAMA_SHA256", bad_sha)
        with pytest.raises(RuntimeError, match=expected_err):
            pw.install_ollama({})
        assert not escape_sentinel.exists()

    # 5. Wrong installed zstandard version refuses in ensure_zstandard
    orig_version = _im.version
    monkeypatch.setattr(_im, "version", lambda pkg: "0.23.0" if pkg == "zstandard" else orig_version(pkg))
    with pytest.raises(RuntimeError, match="REFUSED: zstandard version is '0.23.0', expected '0.25.0'"):
        pw.ensure_zstandard()


# ---- PIN_OFFRIG_MANIFEST (R&D, 2026-10-23, window 17l): install pinned Ollama model from manifest + blobs ----------------

FIXTURE_MANIFEST_DIR = Path(__file__).resolve().parent / "fixtures" / "ollama-manifest"
FIXTURE_MANIFEST_PATH = FIXTURE_MANIFEST_DIR / "manifest.json"
FIXTURE_MANIFEST_SHA256 = "ef71a9e44671e4680a59cfa480bb5465c68927a574fb6a5308d4b67b9976b1c5"
FIXTURE_MODEL_BLOB_SHA256 = "359ca17dcd875c3348e7804457abc3e0c0fd4dd24bd505f58c2e0dc281741f24"


def _start_test_blob_registry(
    blob_overrides: dict[str, bytes] | None = None,
    status_overrides: dict[str, int] | None = None,
    on_blob=None,
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    overrides_bytes = blob_overrides or {}
    overrides_status = status_overrides or {}
    blobs: dict[str, bytes] = {
        p.name.removeprefix("sha256-"): p.read_bytes()
        for p in sorted(FIXTURE_MANIFEST_DIR.glob("sha256-*"))
        if p.is_file()
    }
    blobs.update(overrides_bytes)

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class _TestRegistry:
        def __init__(self):
            self.requests: list[str] = []
            reg = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_GET(self):  # noqa: N802
                    reg.requests.append(self.path)
                    if self.path.startswith("/v2/") and "/blobs/sha256:" in self.path:
                        for known_hex, code in overrides_status.items():
                            if self.path.endswith(f"/blobs/sha256:{known_hex}"):
                                self.send_response(code)
                                self.end_headers()
                                return
                        for known_hex in blobs:
                            if self.path.endswith(f"/blobs/sha256:{known_hex}"):
                                self.send_response(307)
                                self.send_header("Location", f"{reg.base_url}/_blobs/sha256:{known_hex}")
                                self.end_headers()
                                return
                        self.send_response(404)
                        self.end_headers()
                        return
                    for known_hex, data in blobs.items():
                        if self.path == f"/_blobs/sha256:{known_hex}":
                            if on_blob is not None:
                                on_blob(known_hex)
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                    self.send_response(404)
                    self.end_headers()

            self.server = _Reuse(("127.0.0.1", 0), Handler)
            self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self) -> None:
            self.server.shutdown()
            self.server.server_close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    return _TestRegistry()


def _unlock_offrig_in(work: Path) -> None:
    root = work / "offrig-bench-in"
    if root.exists():
        os.chmod(root, 0o700)
        for p in sorted(root.rglob("*")):
            os.chmod(p, 0o700)


def _setup_offrig_manifest_env(tmp_path: Path, monkeypatch, *, stub: bool = True) -> tuple[Path, Path, int]:
    import socket as _socket
    import tarfile

    content = tmp_path / "content"
    work = content / "pin"
    shm = tmp_path / "shm"
    content.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    shm.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", shm / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    for k in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PIN_DEADLINE",
        "PIN_STUB_CALL_S",
        "PIN_OFFRIG_MANIFEST",
        "PIN_OLLAMA_REGISTRY",
        "PIN_OLLAMA_MIRROR",
    ):
        monkeypatch.delenv(k, raising=False)

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(pw, "OLLAMA_PORT", free_port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.05)

    _offrig_bin, bin_sha, _gold, gold_sha = _write_offrig_fixtures(content)
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    monkeypatch.setenv("PIN_OFFRIG_SHA256", bin_sha)
    monkeypatch.setenv("PIN_OFFRIG_GOLD_SHA256", gold_sha)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    if stub:
        monkeypatch.setenv("PIN_STUB", "1")
    else:
        monkeypatch.delenv("PIN_STUB", raising=False)
        fake_ollama_bin = work / "ollama-install" / "bin" / "ollama"
        fake_ollama_bin.parent.mkdir(parents=True, exist_ok=True)
        fake_ollama_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        monkeypatch.setattr(
            pw,
            "install_ollama",
            lambda res: res.__setitem__(
                "ollama", {"version": "0.40.4", "sha256": pw.OLLAMA_SHA256, "bin": str(fake_ollama_bin)}
            )
            or fake_ollama_bin,
        )
        stub_holder: dict = {}

        def fake_enter(self):
            stub_holder["srv"] = __import__("stub_llama").StubLlama(
                self.model,
                self.model,
                port=self.port,
                ollama_model=self.model,
                ollama_digest=self.digest,
                ollama_version=pw.OLLAMA_VERSION,
            )
            return self

        def fake_exit(self, *exc):
            if stub_holder.get("srv") is not None:
                stub_holder["srv"].close()
                stub_holder["srv"] = None

        monkeypatch.setattr(pw.OllamaServed, "__enter__", fake_enter)
        monkeypatch.setattr(pw.OllamaServed, "__exit__", fake_exit)
        monkeypatch.setattr(
            pw,
            "_query_smi_csv",
            lambda: "2026/10/23 00:00:01.000, 21450 MiB, 82 %, 58, 265.10 W\n",
        )

    orig_run_proc = pw.run_proc

    def cross_platform_run_proc(cmd, **kw):
        if isinstance(cmd, list) and cmd and str(cmd[0]).endswith("offrig") and os.name == "nt":
            cmd = [sys.executable, str(cmd[0]), *cmd[1:]]
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", cross_platform_run_proc)

    def tar_sh(cmd: str, **kw) -> str:
        if cmd.startswith(f"tar -C {work} -cf {work}/offrig-bench-out.tar offrig-bench"):
            with tarfile.open(work / "offrig-bench-out.tar", "w") as tf:
                tf.add(work / "offrig-bench", arcname="offrig-bench")
            return ""
        return pw.sh(cmd, **kw)

    monkeypatch.setattr(pw, "sh", tar_sh)
    return content, work, free_port


def test_offrig_bench_manifest_sha_mismatch_refuses_before_anything_else(tmp_path, monkeypatch):
    _content, _work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    inputs_called = False
    install_called = False

    def spy_inputs(*a, **kw):
        nonlocal inputs_called
        inputs_called = True
        raise AssertionError("offrig_bench_inputs must not run when PIN_OFFRIG_MANIFEST sha256 mismatches")

    def spy_install(*a, **kw):
        nonlocal install_called
        install_called = True
        raise AssertionError("install_ollama must not run when PIN_OFFRIG_MANIFEST sha256 mismatches")

    monkeypatch.setattr(pw, "offrig_bench_inputs", spy_inputs)
    monkeypatch.setattr(pw, "install_ollama", spy_install)

    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: ollama manifest manifest\.json sha256 mismatch: {FIXTURE_MANIFEST_SHA256[:12]} != PIN_OFFRIG_MODEL_DIGEST {pw.OFFRIG_MODEL_DIGEST[:12]}",
    ):
        pw.offrig_bench_window({})
    assert inputs_called is False
    assert install_called is False


def test_offrig_bench_manifest_blob_sha_mismatch_deletes_temp_and_refuses(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    bad_model_bytes = b"X" * 36
    with _start_test_blob_registry(blob_overrides={FIXTURE_MODEL_BLOB_SHA256: bad_model_bytes}) as reg:
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", reg.base_url)
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: ollama blob sha256:{FIXTURE_MODEL_BLOB_SHA256} sha256 mismatch:",
        ):
            pw.offrig_bench_window({})

    blobs_dir = work / "ollama-models" / "blobs"
    assert not (blobs_dir / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}").exists()
    assert not (blobs_dir / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}.part").exists()


def test_offrig_bench_manifest_blob_size_mismatch_deletes_temp_and_refuses(tmp_path, monkeypatch):
    import hashlib

    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    bad_size_manifest = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    bad_size_manifest["layers"][0]["size"] = 999
    bad_manifest_path = content / "bad-size-manifest.json"
    bad_manifest_bytes = json.dumps(bad_size_manifest, separators=(",", ":")).encode("utf-8")
    bad_manifest_path.write_bytes(bad_manifest_bytes)
    bad_manifest_sha = hashlib.sha256(bad_manifest_bytes).hexdigest()

    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(bad_manifest_path))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", bad_manifest_sha)

    with _start_test_blob_registry() as reg:
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", reg.base_url)
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: ollama blob sha256:{FIXTURE_MODEL_BLOB_SHA256} size mismatch: got 36 bytes, expected 999",
        ):
            pw.offrig_bench_window({})

    blobs_dir = work / "ollama-models" / "blobs"
    assert not (blobs_dir / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}").exists()
    assert not (blobs_dir / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}.part").exists()


def test_offrig_bench_manifest_blob_404_refuses_with_digest_and_status_without_pull(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_proc_cmds: list[list[str]] = []
    orig_run_proc = pw.run_proc

    def spy_run_proc(cmd, **kw):
        if isinstance(cmd, list):
            seen_proc_cmds.append(list(cmd))
            if cmd[1:3] == ["pull", "gemma4:31b"]:
                return subprocess.CompletedProcess(cmd, 0, "pulled\n", "")
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", spy_run_proc)

    with _start_test_blob_registry(status_overrides={FIXTURE_MODEL_BLOB_SHA256: 404}) as reg:
        monkeypatch.setattr(pw, "_resolve_ollama_registry_base", lambda _raw=None: (reg.base_url, None))
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: ollama blob sha256:{FIXTURE_MODEL_BLOB_SHA256} HTTP 404",
        ):
            pw.offrig_bench_window({})

    assert not any("pull" in c for c in seen_proc_cmds), seen_proc_cmds
    assert not (work / "ollama-models" / "blobs" / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}").exists()


def test_offrig_bench_manifest_mode_never_runs_ollama_pull_on_success_or_failure(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_proc_cmds: list[list[str]] = []
    orig_run_proc = pw.run_proc

    def spy_run_proc(cmd, **kw):
        if isinstance(cmd, list):
            seen_proc_cmds.append(list(cmd))
            if cmd[1:3] == ["pull", "gemma4:31b"]:
                return subprocess.CompletedProcess(cmd, 0, "pulled\n", "")
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", spy_run_proc)

    # 1. Success in manifest mode: no `ollama pull` command runs
    with _start_test_blob_registry() as reg:
        monkeypatch.setattr(pw, "_resolve_ollama_registry_base", lambda _raw=None: (reg.base_url, None))
        res_ok: dict = {}
        pw.offrig_bench_window(res_ok)
    pull_cmds_ok = [c for c in seen_proc_cmds if "pull" in c]
    assert pull_cmds_ok == []
    assert res_ok["model_errors"] == 0
    assert res_ok["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256

    # 2. Failure in manifest mode (503 on a blob): refuses and never falls back to `ollama pull`
    seen_proc_cmds.clear()
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    with _start_test_blob_registry(status_overrides={FIXTURE_MODEL_BLOB_SHA256: 503}) as reg:
        monkeypatch.setattr(pw, "_resolve_ollama_registry_base", lambda _raw=None: (reg.base_url, None))
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: ollama blob sha256:{FIXTURE_MODEL_BLOB_SHA256} HTTP 503",
        ):
            pw.offrig_bench_window({})
    pull_cmds_fail = [c for c in seen_proc_cmds if "pull" in c]
    assert pull_cmds_fail == []


def test_offrig_bench_manifest_writes_blobs_and_manifest_byte_for_byte(tmp_path, monkeypatch):
    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    expected_manifest_dst = (
        work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b"
    )
    manifest_existed_before_serve = False
    orig_enter = pw.OllamaServed.__enter__

    def check_before_serve(self):
        nonlocal manifest_existed_before_serve
        manifest_existed_before_serve = expected_manifest_dst.is_file()
        return orig_enter(self)

    monkeypatch.setattr(pw.OllamaServed, "__enter__", check_before_serve)

    res: dict = {}
    pw.offrig_bench_window(res)
    assert manifest_existed_before_serve is True
    assert expected_manifest_dst.read_bytes() == FIXTURE_MANIFEST_PATH.read_bytes()

    parsed = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    all_descs = [parsed["config"], *parsed["layers"]]
    for desc in all_descs:
        bhex = desc["digest"].removeprefix("sha256:")
        written_blob = work / "ollama-models" / "blobs" / f"sha256-{bhex}"
        fixture_blob = FIXTURE_MANIFEST_DIR / f"sha256-{bhex}"
        assert written_blob.is_file()
        assert written_blob.read_bytes() == fixture_blob.read_bytes()
        assert len(written_blob.read_bytes()) == desc["size"]

    recorded_blobs = res["offrig_model"]["blobs"]
    assert len(recorded_blobs) == 4
    assert [b["digest"] for b in recorded_blobs] == [d["digest"] for d in all_descs]
    assert [b["size"] for b in recorded_blobs] == [d["size"] for d in all_descs]
    assert all(isinstance(b["seconds"], (int, float)) and b["seconds"] >= 0 for b in recorded_blobs)

    # Also verify inline stub fallback when tests/fixtures/ollama-manifest is absent on a standalone CPU VM
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    monkeypatch.setattr(pw, "_repo_root", lambda: content / "no-repo")
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", "stub")
    monkeypatch.delenv("PIN_OFFRIG_MODEL_DIGEST", raising=False)
    res_inline_fixture: dict = {}
    pw.offrig_bench_window(res_inline_fixture)
    assert expected_manifest_dst.read_bytes() == FIXTURE_MANIFEST_PATH.read_bytes()
    assert res_inline_fixture["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256


def test_offrig_bench_manifest_api_tags_confirms_digest_from_written_manifest(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    # 1. Stub Ollama's /api/tags reports the digest read from the written manifest file on disk
    res_ok: dict = {}
    pw.offrig_bench_window(res_ok)
    assert res_ok["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256

    # 2. If the written manifest file on disk has a different digest when OllamaServed starts, /api/tags reports the
    #    tampered file's digest (not a constant passed to OllamaServed) and pull_ollama_model refuses
    for p in sorted((work / "offrig-bench-in").rglob("*")):
        os.chmod(p, 0o700)
    os.chmod(work / "offrig-bench-in", 0o700)
    expected_manifest_dst = (
        work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b"
    )
    orig_enter = pw.OllamaServed.__enter__

    def tamper_written_manifest(self):
        expected_manifest_dst.parent.mkdir(parents=True, exist_ok=True)
        expected_manifest_dst.write_bytes(b'{"tampered":true}')
        return orig_enter(self)

    monkeypatch.setattr(pw.OllamaServed, "__enter__", tamper_written_manifest)
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: model gemma4:31b digest .* != PIN_OFFRIG_MODEL_DIGEST {FIXTURE_MANIFEST_SHA256[:12]}",
    ):
        pw.offrig_bench_window({})
    tampered_sha = pw.sha256(expected_manifest_dst)
    assert tampered_sha != FIXTURE_MANIFEST_SHA256


def test_offrig_bench_manifest_deadline_cut_during_blob_download(tmp_path, monkeypatch):
    import time as _t

    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)
    monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 60))

    def expire_deadline_on_model_blob(blob_hex: str) -> None:
        if blob_hex == FIXTURE_MODEL_BLOB_SHA256:
            monkeypatch.setenv("PIN_DEADLINE", str(_t.time() - 1))

    with _start_test_blob_registry(on_blob=expire_deadline_on_model_blob) as reg:
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", reg.base_url)
        with pytest.raises(
            pw.DeadlineReached,
            match=rf"ollama blob sha256:{FIXTURE_MODEL_BLOB_SHA256} download was cut at the window's deadline",
        ):
            pw.offrig_bench_window({})

    assert pw.CUT_AT is not None
    assert pw.CUT_AT["wave"] == "offrig-bench-blob"
    assert pw.CUT_AT["blob"] == f"sha256:{FIXTURE_MODEL_BLOB_SHA256}"
    assert not (work / "ollama-models" / "blobs" / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}.part").exists()
    assert not (work / "ollama-models" / "blobs" / f"sha256-{FIXTURE_MODEL_BLOB_SHA256}").exists()
    assert (work / "offrig-bench-out.tar").is_file()


def test_offrig_bench_unset_manifest_behaves_as_before(tmp_path, monkeypatch):
    _content, work, free_port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.delenv("PIN_OFFRIG_MANIFEST", raising=False)
    monkeypatch.setenv("PIN_OFFRIG_MODEL", pw.OFFRIG_MODEL)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    seen_proc_cmds: list[list[str]] = []
    orig_run_proc = pw.run_proc

    def spy_run_proc(cmd, **kw):
        if isinstance(cmd, list):
            seen_proc_cmds.append(list(cmd))
            if cmd[1:3] == ["pull", "gemma4:31b"]:
                assert kw["env"]["OLLAMA_HOST"] == f"127.0.0.1:{free_port}"
                return subprocess.CompletedProcess(cmd, 0, "pulling manifest\nsuccess\n", "")
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", spy_run_proc)

    res: dict = {}
    pw.offrig_bench_window(res)
    assert [c[1] for c in seen_proc_cmds] == ["pull", "verify"]
    assert "offrig_manifest" not in res
    assert res["offrig_model"] == {
        "model": "gemma4:31b",
        "digest": pw.OFFRIG_MODEL_DIGEST,
        "size": 20281829620,
    }
    assert not (work / "ollama-models" / "manifests").exists()


def test_offrig_bench_manifest_registry_allowlist_on_paid_and_stub_windows(tmp_path, monkeypatch):
    _content, _work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_urls: list[str] = []
    orig_urlopen = pw.urllib.request.urlopen

    def fail_on_blob_network(req_or_url, *a, **kw):
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        parts = pw.urllib.parse.urlsplit(url)
        if "/v2/" in parts.path and "/blobs/" in parts.path:
            seen_urls.append(url)
            raise AssertionError(f"network call must not happen on disallowed registry: {url}")
        return orig_urlopen(req_or_url, *a, **kw)

    monkeypatch.setattr(pw.urllib.request, "urlopen", fail_on_blob_network)

    # 1. Paid window (PIN_STUB unset) refuses non-listed https:// base, prefix-match attack, http://, and file:// before any network call
    for bad_reg, expected_match in (
        (
            "http://registry.ollama.ai",
            r"REFUSED: PIN_OLLAMA_REGISTRY='http://registry\.ollama\.ai' must use https:// on a paid window",
        ),
        (
            "http://127.0.0.1:9999",
            r"REFUSED: PIN_OLLAMA_REGISTRY='http://127\.0\.0\.1:9999' must use https:// on a paid window",
        ),
        (
            "file:///tmp/blobs",
            r"REFUSED: PIN_OLLAMA_REGISTRY='file:///tmp/blobs' must use https:// on a paid window",
        ),
        (
            "https://evil.example.com",
            r"REFUSED: PIN_OLLAMA_REGISTRY='https://evil\.example\.com' .* is not in OLLAMA_REGISTRY_ALLOWLIST on a paid window",
        ),
        (
            "https://registry.ollama.ai.evil.example.com",
            r"REFUSED: PIN_OLLAMA_REGISTRY='https://registry\.ollama\.ai\.evil\.example\.com' .* is not in OLLAMA_REGISTRY_ALLOWLIST on a paid window",
        ),
        (
            "https://registry.ollama.ai/extra-path",
            r"REFUSED: PIN_OLLAMA_REGISTRY='https://registry\.ollama\.ai/extra-path' .* is not in OLLAMA_REGISTRY_ALLOWLIST on a paid window",
        ),
    ):
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", bad_reg)
        with pytest.raises(RuntimeError, match=expected_match):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        assert seen_urls == []

    assert pw.OLLAMA_REGISTRY == "https://registry.ollama.ai"
    assert pw.OLLAMA_REGISTRY_ALLOWLIST == ()

    # 2. Default registry (and scheme-case / trailing-slash normalised form) is accepted on a paid window and recorded in pin-result.json
    with _start_test_blob_registry() as reg:

        def intercept_https_to_local(req_or_url, *a, **kw):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
            parts = pw.urllib.parse.urlsplit(url)
            if (parts.scheme, parts.hostname, parts.port) == ("https", "registry.ollama.ai", None) and parts.path.startswith("/"):
                seen_urls.append(url)
                local_url = f"{reg.base_url}{parts.path}" + (f"?{parts.query}" if parts.query else "")
                return orig_urlopen(pw.urllib.request.Request(local_url), *a, **kw)
            if (parts.scheme, parts.hostname, parts.port) == ("https", "mirror.studio.example", None) and parts.path.startswith("/ollama/"):
                seen_urls.append(url)
                rel_path = parts.path.removeprefix("/ollama")
                local_url = f"{reg.base_url}{rel_path}" + (f"?{parts.query}" if parts.query else "")
                return orig_urlopen(pw.urllib.request.Request(local_url), *a, **kw)
            return orig_urlopen(req_or_url, *a, **kw)

        monkeypatch.setattr(pw.urllib.request, "urlopen", intercept_https_to_local)

        # Default (PIN_OLLAMA_REGISTRY unset)
        monkeypatch.delenv("PIN_OLLAMA_REGISTRY", raising=False)
        res_default: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_default)
        assert res_default["offrig_manifest"]["registry"] == "https://registry.ollama.ai"
        assert seen_urls and all(
            ((p := pw.urllib.parse.urlsplit(u)).scheme, p.hostname, p.port) == ("https", "registry.ollama.ai", None)
            and p.path.startswith("/v2/")
            for u in seen_urls
        )

        # Default with uppercase scheme and trailing slashes
        seen_urls.clear()
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", "  HTTPS://registry.ollama.ai///  ")
        res_norm: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_norm)
        assert res_norm["offrig_manifest"]["registry"] == "https://registry.ollama.ai"
        assert len(seen_urls) == 4

        # 3. Explicitly listed base in OLLAMA_REGISTRY_ALLOWLIST is accepted (and prefix-extension of it is still refused)
        seen_urls.clear()
        monkeypatch.setattr(pw, "OLLAMA_REGISTRY_ALLOWLIST", ("https://mirror.studio.example/ollama/",))
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", "HTTPS://mirror.studio.example/ollama//")
        res_listed: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_listed)
        assert res_listed["offrig_manifest"]["registry"] == "https://mirror.studio.example/ollama"
        assert len(seen_urls) == 4 and all(
            ((p := pw.urllib.parse.urlsplit(u)).scheme, p.hostname, p.port) == ("https", "mirror.studio.example", None)
            and p.path.startswith("/ollama/v2/")
            for u in seen_urls
        )

        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", "https://mirror.studio.example/ollama-evil")
        with pytest.raises(
            RuntimeError,
            match=r"REFUSED: PIN_OLLAMA_REGISTRY='https://mirror\.studio\.example/ollama-evil' .* is not in OLLAMA_REGISTRY_ALLOWLIST on a paid window",
        ):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})

        # 4. Stub mode accepts its local fixture registry (when PIN_OLLAMA_REGISTRY is unset) and an explicit http://127.0.0.1: test server
        monkeypatch.setattr(pw.urllib.request, "urlopen", orig_urlopen)
        monkeypatch.setenv("PIN_STUB", "1")
        monkeypatch.delenv("PIN_OLLAMA_REGISTRY", raising=False)
        res_stub_auto: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_stub_auto)
        stub_reg_parts = pw.urllib.parse.urlsplit(res_stub_auto["offrig_manifest"]["registry"])
        assert (stub_reg_parts.scheme, stub_reg_parts.hostname) == ("http", "127.0.0.1")
        assert res_stub_auto["offrig_manifest"]["stub"] is True

        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", reg.base_url + "/")
        res_stub_explicit: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_stub_explicit)
        assert res_stub_explicit["offrig_manifest"]["registry"] == reg.base_url


def test_offrig_bench_manifest_blob_read_timeout_stall_deadline_cut_and_slow_live_stream(tmp_path, monkeypatch):
    import hashlib
    import threading
    import time as _t
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)

    # Build a manifest whose model blob is 3 MiB (3 chunks of 1 MiB) so multi-chunk streaming is tested without a large file
    three_mib = b"A" * (1 << 20) + b"B" * (1 << 20) + b"C" * (1 << 20)
    three_mib_sha = hashlib.sha256(three_mib).hexdigest()
    custom_manifest_obj = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    custom_manifest_obj["layers"][0]["digest"] = f"sha256:{three_mib_sha}"
    custom_manifest_obj["layers"][0]["size"] = len(three_mib)
    custom_manifest_bytes = json.dumps(custom_manifest_obj, separators=(",", ":")).encode("utf-8")
    custom_manifest_path = content / "three-mib-manifest.json"
    custom_manifest_path.write_bytes(custom_manifest_bytes)
    custom_manifest_sha = hashlib.sha256(custom_manifest_bytes).hexdigest()

    base_blobs: dict[str, bytes] = {
        p.name.removeprefix("sha256-"): p.read_bytes()
        for p in sorted(FIXTURE_MANIFEST_DIR.glob("sha256-*"))
        if p.is_file()
    }
    base_blobs[three_mib_sha] = three_mib

    mode_holder = {"mode": "stall"}

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            for known_hex, data in base_blobs.items():
                if self.path.endswith(f"/blobs/sha256:{known_hex}"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if known_hex != three_mib_sha:
                        self.wfile.write(data)
                        self.wfile.flush()
                        return
                    mode = mode_holder["mode"]
                    if mode == "stall":
                        # Write 1 MiB, then stall longer than OLLAMA_BLOB_READ_TIMEOUT_S while PIN_DEADLINE still has >500s left
                        self.wfile.write(data[: 1 << 20])
                        self.wfile.flush()
                        _t.sleep(0.35)
                        return
                    if mode == "deadline_mid_blob":
                        # Write chunk 1 (1 MiB) and chunk 2 (1 MiB); advance PIN_DEADLINE while chunk 2 is in flight so
                        # the client cuts at the between-chunk check before reading chunk 3 (within one chunk of deadline)
                        self.wfile.write(data[: 1 << 20])
                        self.wfile.flush()
                        _t.sleep(0.05)
                        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() - 1))
                        try:
                            self.wfile.write(data[1 << 20 : 2 << 20])
                            self.wfile.flush()
                        except OSError:
                            pass
                        _t.sleep(0.2)
                        return
                    if mode == "slow_live":
                        # Write 3 chunks of 1 MiB with a pause between chunks (< idle timeout, but total > 1x idle timeout)
                        for idx in range(3):
                            if idx > 0:
                                _t.sleep(0.06)
                            self.wfile.write(data[idx * (1 << 20) : (idx + 1) * (1 << 20)])
                            self.wfile.flush()
                        return
            self.send_response(404)
            self.end_headers()

    srv = _Reuse(("127.0.0.1", 0), Handler)
    base_url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", base_url)

        # 1. Socket stall with plenty of window time left (e.g. 600s, < 1800s!) refuses as a stall naming blob & bytes, NEVER DeadlineReached
        # Even if urlopen timeout is forced to 0.1s (as on a94bf977 when monkeypatched), a94bf977 misclassifies the timeout
        # as DeadlineReached because initial `left < 1800` is True when PIN_DEADLINE is +600s!
        orig_urlopen = pw.urllib.request.urlopen
        monkeypatch.setattr(
            pw.urllib.request,
            "urlopen",
            lambda req_or_url, *a, **kw: orig_urlopen(req_or_url, *a, **{**kw, "timeout": min(float(kw.get("timeout") or 0.1), 0.1)}),
        )
        if hasattr(pw, "OLLAMA_BLOB_READ_TIMEOUT_S"):
            monkeypatch.setattr(pw, "OLLAMA_BLOB_READ_TIMEOUT_S", 0.1)
        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 600))
        mode_holder["mode"] = "stall"
        res_stall: dict = {}
        with pytest.raises(
            RuntimeError,
            match=rf"REFUSED: ollama blob sha256:{three_mib_sha} socket read stall after {1 << 20} of {3 << 20} bytes \(idle timeout 0\.1s\)",
        ):
            pw.install_ollama_manifest(custom_manifest_path, pw.OFFRIG_MODEL, custom_manifest_sha, res_stall)
        assert not (work / "ollama-models" / "blobs" / f"sha256-{three_mib_sha}.part").exists()
        assert pw.OLLAMA_BLOB_READ_TIMEOUT_S == 0.1

        # 2. Deadline passing mid-blob cuts within one chunk (at 1 MiB or 2 MiB of 3 MiB, never completing 3 MiB) and raises DeadlineReached
        monkeypatch.setattr(pw.urllib.request, "urlopen", orig_urlopen)
        monkeypatch.setattr(pw, "OLLAMA_BLOB_READ_TIMEOUT_S", 2.0)
        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 600))
        mode_holder["mode"] = "deadline_mid_blob"
        res_cut: dict = {}
        with pytest.raises(
            pw.DeadlineReached,
            match=rf"ollama blob sha256:{three_mib_sha} download was cut at the window's deadline after (1048576|2097152) of {3 << 20} bytes",
        ):
            pw.install_ollama_manifest(custom_manifest_path, pw.OFFRIG_MODEL, custom_manifest_sha, res_cut)
        assert res_cut["offrig_manifest"]["failed_blob"]["bytes_received"] in (1 << 20, 2 << 20)

        # 3. Slow but live stream with time left completes cleanly
        monkeypatch.setattr(pw, "OLLAMA_BLOB_READ_TIMEOUT_S", 0.25)
        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 600))
        mode_holder["mode"] = "slow_live"
        res_live: dict = {}
        pw.install_ollama_manifest(custom_manifest_path, pw.OFFRIG_MODEL, custom_manifest_sha, res_live)
        assert (work / "ollama-models" / "blobs" / f"sha256-{three_mib_sha}").stat().st_size == 3 << 20
        assert len(res_live["offrig_manifest"]["blobs"]) == 4
        assert "failed_blob" not in res_live["offrig_manifest"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_offrig_bench_manifest_records_progress_in_pin_result_json_on_mid_blob_cut_and_failure(tmp_path, monkeypatch):
    import hashlib
    import threading
    import time as _t
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)

    # Create a 3 MiB model blob so cutting mid-blob leaves partial bytes_received on the in-progress blob
    three_mib = b"M" * (3 << 20)
    three_mib_sha = hashlib.sha256(three_mib).hexdigest()
    manifest_obj = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    config_digest = manifest_obj["config"]["digest"]
    config_size = manifest_obj["config"]["size"]
    manifest_obj["layers"][0]["digest"] = f"sha256:{three_mib_sha}"
    manifest_obj["layers"][0]["size"] = len(three_mib)
    manifest_bytes = json.dumps(manifest_obj, separators=(",", ":")).encode("utf-8")
    manifest_file = content / "progress-manifest.json"
    manifest_file.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    base_blobs: dict[str, bytes] = {
        p.name.removeprefix("sha256-"): p.read_bytes()
        for p in sorted(FIXTURE_MANIFEST_DIR.glob("sha256-*"))
        if p.is_file()
    }
    base_blobs[three_mib_sha] = three_mib

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            for known_hex, data in base_blobs.items():
                if self.path.endswith(f"/blobs/sha256:{known_hex}"):
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    if known_hex == three_mib_sha:
                        self.wfile.write(data[: 1 << 20])
                        self.wfile.flush()
                        _t.sleep(0.05)
                        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() - 1))
                        try:
                            self.wfile.write(data[1 << 20 : 2 << 20])
                            self.wfile.flush()
                        except OSError:
                            pass
                        _t.sleep(0.2)
                        return
                    self.wfile.write(data)
                    self.wfile.flush()
                    return
            self.send_response(404)
            self.end_headers()

    srv = _Reuse(("127.0.0.1", 0), Handler)
    base_url = f"http://127.0.0.1:{srv.server_address[1]}"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("PIN_CONTENT", str(content))
        monkeypatch.setenv("PIN_WORK", str(work))
        monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(manifest_file))
        monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", manifest_sha)
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", base_url)
        monkeypatch.setenv("PIN_DEADLINE", str(_t.time() + 120))
        monkeypatch.setattr(pw, "take_job_lock", lambda _w: open(work / "job.lock", "w"))  # noqa: SIM115

        rc = pw.main()
        assert rc == 1
        pin_result_path = work / "pin-result.json"
        assert pin_result_path.is_file()
        pin_res = json.loads(pin_result_path.read_text(encoding="utf-8"))
        assert pin_res["ok"] is False
        assert "DeadlineReached" in pin_res["error"]

        om = pin_res["offrig_manifest"]
        assert om["registry"] == base_url
        assert om["path"] == str(manifest_file)
        assert om["sha256"] == manifest_sha
        assert om["manifest_dst"] == str(
            work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b"
        )
        # Config blob completed before the model layer was cut mid-blob
        assert len(om["blobs"]) == 1
        assert om["blobs"][0]["digest"] == config_digest
        assert om["blobs"][0]["size"] == config_size
        failed = om["failed_blob"]
        assert failed["digest"] == f"sha256:{three_mib_sha}"
        assert failed["bytes_received"] in (1 << 20, 2 << 20)
        assert failed["expected_size"] == 3 << 20
        assert "cut at the window's deadline" in failed["reason"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_offrig_bench_model_name_validation_rejects_invalid_parts_in_pin_window_and_stub_llama(tmp_path):
    import stub_llama

    assert pw._parse_ollama_model("gemma4:31b") == ("library", "gemma4", "31b")
    assert pw._parse_ollama_model("custom-ns/model_1.0:v2-q8") == ("custom-ns", "model_1.0", "v2-q8")

    invalid_models = [
        ".",
        ".:31b",
        "gemma4:.",
        "gemma4:31b:extra",
        "gemma4?x=1:31b",
        "gemma4:31b?x=1",
        "gemma4#frag:31b",
        "gemma4:31b#frag",
        "-gemma4:31b",
        "_gemma4:31b",
        ".gemma4:31b",
        "gemma4:-31b",
        "gemma4:.31b",
        "gemma 4:31b",
        "",
        "   ",
        "..",
        "gemma4:..",
        "../gemma4:31b",
        "library/../gemma4:31b",
        "library/gemma4/extra:31b",
        "gemma4:31b/extra",
        "gemma4\\evil:31b",
    ]
    for bad in invalid_models:
        with pytest.raises(RuntimeError, match="REFUSED: invalid PIN_OFFRIG_MODEL"):
            pw._parse_ollama_model(bad)
        with pytest.raises(RuntimeError, match="REFUSED: invalid PIN_OFFRIG_MODEL"):
            stub_llama._parse_ollama_model(bad)
        assert stub_llama._written_manifest_digest(bad, tmp_path) is None

    assert stub_llama._parse_ollama_model("gemma4:31b") == ("library", "gemma4", "31b")


def test_offrig_bench_stub_reused_work_dir_ignores_stale_manifest_when_pin_offrig_manifest_unset(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)

    # 1. Run 1 in manifest mode: writes fixture manifest (digest FIXTURE_MANIFEST_SHA256) into work/ollama-models/manifests/...
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", "stub")
    monkeypatch.delenv("PIN_OFFRIG_MODEL_DIGEST", raising=False)
    res_manifest: dict = {}
    pw.offrig_bench_window(res_manifest)
    written_manifest = work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b"
    assert written_manifest.is_file()
    assert res_manifest["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256

    # 2. Run 2 in the SAME reused WORK directory with PIN_OFFRIG_MANIFEST unset (expecting OFFRIG_MODEL_DIGEST):
    # Both StubLlama and _InlineStubOllama must report OFFRIG_MODEL_DIGEST, never the leftover manifest on disk.
    _unlock_offrig_in(work)
    monkeypatch.delenv("PIN_OFFRIG_MANIFEST", raising=False)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)

    res_unset_stub_llama: dict = {}
    pw.offrig_bench_window(res_unset_stub_llama)
    assert written_manifest.is_file()  # stale file was still on disk
    assert res_unset_stub_llama["offrig_model"]["digest"] == pw.OFFRIG_MODEL_DIGEST
    assert "offrig_manifest" not in res_unset_stub_llama

    # Also verify _InlineStubOllama on the same reused WORK directory with PIN_OFFRIG_MANIFEST unset
    _unlock_offrig_in(work)
    monkeypatch.setitem(sys.modules, "stub_llama", None)
    res_unset_inline: dict = {}
    pw.offrig_bench_window(res_unset_inline)
    assert res_unset_inline["offrig_model"]["digest"] == pw.OFFRIG_MODEL_DIGEST
    assert "offrig_manifest" not in res_unset_inline


def test_offrig_bench_manifest_mode_exercises_inline_stub_ollama_end_to_end(tmp_path, monkeypatch):
    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    # Force _start_stub_ollama onto _InlineStubOllama (as on a standalone Colab VM without tests/stub_llama.py)
    monkeypatch.setitem(sys.modules, "stub_llama", None)
    monkeypatch.setattr(pw, "_repo_root", lambda: content / "no-repo")
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", "stub")
    monkeypatch.delenv("PIN_OFFRIG_MODEL_DIGEST", raising=False)

    started_stubs: list = []
    orig_start = pw._start_stub_ollama

    def spy_start(model, digest, port, version=pw.OLLAMA_VERSION):
        srv = orig_start(model, digest, port, version)
        started_stubs.append(srv)
        return srv

    monkeypatch.setattr(pw, "_start_stub_ollama", spy_start)

    # 1. End-to-end manifest mode on _InlineStubOllama reads the written manifest's digest from disk
    res: dict = {}
    pw.offrig_bench_window(res)
    assert [type(s).__name__ for s in started_stubs] == ["_InlineStubOllama"]
    assert res["model_errors"] == 0
    assert res["offrig_manifest"]["sha256"] == pw.STUB_OLLAMA_MANIFEST_DIGEST
    inline_reg_parts = pw.urllib.parse.urlsplit(res["offrig_manifest"]["registry"])
    assert (inline_reg_parts.scheme, inline_reg_parts.hostname) == ("http", "127.0.0.1")
    assert len(res["offrig_manifest"]["blobs"]) == 4
    assert res["offrig_model"]["digest"] == pw.STUB_OLLAMA_MANIFEST_DIGEST
    assert res["ps"]["models"][0]["digest"] == pw.STUB_OLLAMA_MANIFEST_DIGEST
    assert res["ps_end"]["models"][0]["digest"] == pw.STUB_OLLAMA_MANIFEST_DIGEST
    assert res["offrig_bench"]["verdicts_count"] == 2

    # 2. If the written manifest file on disk is tampered when _InlineStubOllama serves /api/tags, pull_ollama_model refuses
    _unlock_offrig_in(work)
    expected_manifest_dst = (
        work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b"
    )
    orig_enter = pw.OllamaServed.__enter__

    def tamper_for_inline(self):
        expected_manifest_dst.parent.mkdir(parents=True, exist_ok=True)
        expected_manifest_dst.write_bytes(b'{"inline_tampered":true}')
        return orig_enter(self)

    monkeypatch.setattr(pw.OllamaServed, "__enter__", tamper_for_inline)
    with pytest.raises(
        RuntimeError,
        match=rf"REFUSED: model gemma4:31b digest .* != PIN_OFFRIG_MODEL_DIGEST {pw.STUB_OLLAMA_MANIFEST_DIGEST[:12]}",
    ):
        pw.offrig_bench_window({})
    # 3. With PIN_OFFRIG_MANIFEST unset on a reused WORK directory holding a stale manifest file, _InlineStubOllama
    #    must report OFFRIG_MODEL_DIGEST (never the stale manifest on disk)
    _unlock_offrig_in(work)
    monkeypatch.setattr(pw.OllamaServed, "__enter__", orig_enter)
    expected_manifest_dst.write_bytes(FIXTURE_MANIFEST_PATH.read_bytes())
    monkeypatch.delenv("PIN_OFFRIG_MANIFEST", raising=False)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)
    res_unset: dict = {}
    pw.offrig_bench_window(res_unset)
    assert len(started_stubs) == 3 and type(started_stubs[2]).__name__ == "_InlineStubOllama"
    assert res_unset["offrig_model"]["digest"] == pw.OFFRIG_MODEL_DIGEST
    assert "offrig_manifest" not in res_unset


# ---- PIN_OLLAMA_MIRROR (R&D, 2026-10-23): studio HF mirror of pinned Ollama blobs ----------------------------------------

MIRROR_SENTINEL = "hf_" + "SENTINELneverPERSIST9f3c2a71abcdEFGHijk"
PINNED_MIRROR_COMMIT = "9dfbad5aa9ecec08cd4275a1779ae2be3bcf2725"
PINNED_MIRROR_SHA256 = "sha256:9f104ae7d85df083b46d9087b04c9bf44d47f447455d4e1583ebd6d0a66369bb"
PINNED_MIRROR_BASE = f"https://huggingface.co/studio/role-set-ollama-pins/resolve/{PINNED_MIRROR_COMMIT}"


def _start_test_blob_mirror(
    *,
    expected_token: str = MIRROR_SENTINEL,
    revision: str = PINNED_MIRROR_COMMIT,
    blob_overrides: dict[str, bytes] | None = None,
    status_overrides: dict[str, int] | None = None,
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    overrides_bytes = blob_overrides or {}
    overrides_status = status_overrides or {}
    blobs: dict[str, bytes] = {
        p.name.removeprefix("sha256-"): p.read_bytes()
        for p in sorted(FIXTURE_MANIFEST_DIR.glob("sha256-*"))
        if p.is_file()
    }
    blobs.update(overrides_bytes)
    resolve_prefix = f"/studio/role-set-ollama-pins/resolve/{revision}"

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class _TestMirror:
        def __init__(self):
            self.requests: list[str] = []
            self.auth_matched: list[bool] = []
            mir = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_GET(self):  # noqa: N802
                    mir.requests.append(self.path)
                    auth = self.headers.get("Authorization")
                    matched = auth == f"Bearer {expected_token}"
                    mir.auth_matched.append(matched)
                    if not matched:
                        self.send_response(401)
                        self.end_headers()
                        return
                    for known_hex, code in overrides_status.items():
                        if self.path == f"{resolve_prefix}/blobs/sha256-{known_hex}":
                            self.send_response(code)
                            self.end_headers()
                            return
                    for known_hex, data in blobs.items():
                        if self.path == f"{resolve_prefix}/blobs/sha256-{known_hex}":
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                    self.send_response(404)
                    self.end_headers()

            self.server = _Reuse(("127.0.0.1", 0), Handler)
            self.host_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            self.base_url = f"{self.host_url}{resolve_prefix}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self) -> None:
            self.server.shutdown()
            self.server.server_close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    return _TestMirror()


def _find_git_bash() -> str | None:
    if os.name == "nt":
        for p in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ):
            if p.is_file():
                return str(p)
    b = shutil.which("bash")
    if b and os.name == "nt" and "system32" in b.lower():
        return None
    return b


def test_offrig_bench_ollama_mirror_end_to_end_on_fixture_with_auth_header(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_proc_cmds: list[list[str]] = []
    orig_run_proc = pw.run_proc

    def spy_run_proc(cmd, **kw):
        if isinstance(cmd, list):
            seen_proc_cmds.append(list(cmd))
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", spy_run_proc)
    orig_urlopen = pw.urllib.request.urlopen

    with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:
        mir_parts = pw.urllib.parse.urlsplit(mir.base_url)

        def guard_blob_urlopen(req_or_url, *a, **kw):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
            parts = pw.urllib.parse.urlsplit(url)
            if "/blobs/" in parts.path and (
                (parts.scheme, parts.hostname, parts.port) != (mir_parts.scheme, mir_parts.hostname, mir_parts.port)
                or not parts.path.startswith(f"{mir_parts.path}/blobs/sha256-")
            ):
                raise AssertionError(f"blob request went to {url} instead of mirror {mir.base_url}")
            return orig_urlopen(req_or_url, *a, **kw)

        monkeypatch.setattr(pw.urllib.request, "urlopen", guard_blob_urlopen)
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", mir.base_url)
        res: dict = {}
        pw.offrig_bench_window(res)

        assert len(mir.requests) == 4
        assert mir.auth_matched == [True, True, True, True]
        assert all(
            r.startswith(f"/studio/role-set-ollama-pins/resolve/{PINNED_MIRROR_COMMIT}/blobs/sha256-")
            for r in mir.requests
        )

    assert not any("pull" in c for c in seen_proc_cmds)
    assert res["model_errors"] == 0
    assert res["offrig_manifest"]["mirror"] == {
        "base": mir.base_url,
        "revision": PINNED_MIRROR_COMMIT,
    }
    assert "registry" not in res["offrig_manifest"]
    assert res["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256
    assert res["ps"]["models"][0]["digest"] == FIXTURE_MANIFEST_SHA256
    assert res["ps_end"]["models"][0]["digest"] == FIXTURE_MANIFEST_SHA256
    assert res["offrig_bench"]["verdicts_count"] == 2

    parsed = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    for desc in [parsed["config"], *parsed["layers"]]:
        bhex = desc["digest"].removeprefix("sha256:")
        written_blob = work / "ollama-models" / "blobs" / f"sha256-{bhex}"
        assert written_blob.read_bytes() == (FIXTURE_MANIFEST_DIR / f"sha256-{bhex}").read_bytes()

    # Without the right auth token (or when a mirror blob fails with 401/404), it refuses with no fallback to registry or tag
    _unlock_offrig_in(work)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    seen_proc_cmds.clear()
    with (
        _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir,
        _start_test_blob_registry() as reg,
    ):
        monkeypatch.setattr(pw.urllib.request, "urlopen", orig_urlopen)
        pw.TOKEN_PATH.write_text("hf_" + "WRONGtoken0000000000000000000000000000\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", mir.base_url)
        with pytest.raises(RuntimeError, match="HTTP 401"):
            pw.offrig_bench_window({})
        assert mir.auth_matched == [False]
        assert reg.requests == []
    assert not any("pull" in c for c in seen_proc_cmds)

    # Stub fixture mirror shorthand (`PIN_OLLAMA_MIRROR=stub`) also exercises the local fixture mirror with auth header
    _unlock_offrig_in(work)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
    monkeypatch.setenv("PIN_OLLAMA_MIRROR", "stub")
    res_stub_mir: dict = {}
    pw.offrig_bench_window(res_stub_mir)
    assert res_stub_mir["offrig_manifest"]["mirror"]["revision"] == PINNED_MIRROR_COMMIT
    stub_mir_parts = pw.urllib.parse.urlsplit(res_stub_mir["offrig_manifest"]["mirror"]["base"])
    assert (stub_mir_parts.scheme, stub_mir_parts.hostname) == ("http", "127.0.0.1")
    assert stub_mir_parts.path == f"/studio/role-set-ollama-pins/resolve/{PINNED_MIRROR_COMMIT}"


def _run_mirror_base_mutation_suite(
    base_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Run the single-mutation refusal matrix and positive controls against `base_url` without ever printing `base_url`."""
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_urls: list[str] = []
    orig_urlopen = pw.urllib.request.urlopen

    def fail_on_network(req_or_url, *a, **kw):
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        if "/blobs/" in pw.urllib.parse.urlsplit(url).path:
            seen_urls.append(url)
            raise AssertionError("network call must not happen on disallowed mirror")
        return orig_urlopen(req_or_url, *a, **kw)

    monkeypatch.setattr(pw.urllib.request, "urlopen", fail_on_network)

    prefix_before_resolve, orig_rev = base_url.rsplit("/resolve/", 1)
    assert orig_rev == PINNED_MIRROR_COMMIT
    alt_rev = "a621a793661c9cdbced5bd76318b14e8f5a86ed7"
    base_parts = pw.urllib.parse.urlsplit(base_url)
    assert (base_parts.scheme, base_parts.hostname, base_parts.port) == ("https", "huggingface.co", None)
    after_host = base_parts.path.lstrip("/")
    acct_part, repo_part, _rest = after_host.split("/", 2)
    assert repo_part == "role-set-ollama-pins"
    host_prefix = f"{base_parts.scheme}://{base_parts.hostname}/"

    mutations: list[tuple[str, str, str]] = [
        (
            "http_scheme",
            "http://" + base_url.removeprefix("https://"),
            r"REFUSED: PIN_OLLAMA_MIRROR=.* must use https:// on a paid window",
        ),
        (
            "other_40hex_revision",
            f"{prefix_before_resolve}/resolve/{alt_rev}",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* is not in OLLAMA_MIRROR_ALLOWLIST on a paid window",
        ),
        (
            "main_revision",
            f"{prefix_before_resolve}/resolve/main",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* revision 'main' must be a 40-char lowercase hex commit sha",
        ),
        (
            "trailing_path_blobs",
            f"{base_url}/blobs",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* revision '.*/blobs' must be a 40-char lowercase hex commit sha",
        ),
        (
            "trailing_path_extra",
            f"{base_url}/extra",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* revision '.*/extra' must be a 40-char lowercase hex commit sha",
        ),
        (
            "other_repo_name",
            f"{host_prefix}{acct_part}/other-repo/resolve/{orig_rev}",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* is not in OLLAMA_MIRROR_ALLOWLIST on a paid window",
        ),
        (
            "userinfo",
            "https://user@" + base_url.removeprefix("https://"),
            r"REFUSED: PIN_OLLAMA_MIRROR=.* must not include userinfo on a paid window",
        ),
        (
            "port_443",
            f"https://huggingface.co:443/{after_host}",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* must not specify an explicit port on a paid window",
        ),
        (
            "port_8443",
            f"https://huggingface.co:8443/{after_host}",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* must not specify an explicit port on a paid window",
        ),
        (
            "path_case_change",
            f"{host_prefix}{acct_part}/Role-Set-Ollama-Pins/resolve/{orig_rev}",
            r"REFUSED: PIN_OLLAMA_MIRROR=.* is not in OLLAMA_MIRROR_ALLOWLIST on a paid window",
        ),
    ]

    refusal_passes = 0
    for label, mutated_url, expected_match in mutations:
        assert mutated_url != base_url, label
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", mutated_url)
        try:
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        except RuntimeError as exc:
            msg = str(exc)
            if not pw.re.search(expected_match, msg):
                raise AssertionError(
                    f"mutation {label!r} raised wrong message (expected {expected_match!r})"
                ) from None
        else:
            raise AssertionError(f"mutation {label!r} was not refused")
        assert seen_urls == [], label
        assert not pw.TOKEN_PATH.exists() and pw._TOKEN is None
        refusal_passes += 1

    # Positive controls on a paid window: exact base and base with trailing slash
    positive_passes = 0
    with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:

        def intercept_mirror_https_to_local(req_or_url, *a, **kw):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
            parts = pw.urllib.parse.urlsplit(url)
            if (
                (parts.scheme, parts.hostname, parts.port) == (base_parts.scheme, base_parts.hostname, base_parts.port)
                and parts.path.startswith(f"{base_parts.path}/")
            ):
                seen_urls.append(url)
                rel_path = parts.path[len(base_parts.path) :]
                local_url = f"{mir.base_url}{rel_path}" + (f"?{parts.query}" if parts.query else "")
                headers = dict(req_or_url.header_items()) if hasattr(req_or_url, "header_items") else {}
                return orig_urlopen(pw.urllib.request.Request(local_url, headers=headers), *a, **kw)
            return orig_urlopen(req_or_url, *a, **kw)

        monkeypatch.setattr(pw.urllib.request, "urlopen", intercept_mirror_https_to_local)
        for pos_input in (base_url, f"{base_url}/"):
            shutil.rmtree(work / "ollama-models", ignore_errors=True)
            seen_urls.clear()
            mir.auth_matched.clear()
            pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
            monkeypatch.setenv("PIN_OLLAMA_MIRROR", pos_input)
            res_paid: dict = {}
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_paid)
            assert res_paid["offrig_manifest"]["mirror"] == {
                "base": base_url,
                "revision": PINNED_MIRROR_COMMIT,
            }
            assert len(seen_urls) == 4 and all(
                ((p := pw.urllib.parse.urlsplit(u)).scheme, p.hostname, p.port)
                == (base_parts.scheme, base_parts.hostname, base_parts.port)
                and p.path.startswith(f"{base_parts.path}/blobs/sha256-")
                for u in seen_urls
            )
            assert mir.auth_matched == [True, True, True, True]
            positive_passes += 1

    return {"refusal_passes": refusal_passes, "positive_passes": positive_passes}


def test_offrig_bench_ollama_mirror_refuses_unlisted_main_repo_http_trailing_path_and_commit_before_network(
    tmp_path, monkeypatch
):
    assert pw.OLLAMA_MIRROR_ALLOWLIST == (PINNED_MIRROR_SHA256,)
    pinned_test_sha = f"sha256:{pw.hashlib.sha256(PINNED_MIRROR_BASE.encode('utf-8')).hexdigest()}"
    monkeypatch.setattr(pw, "OLLAMA_MIRROR_ALLOWLIST", (pinned_test_sha,))

    counts = _run_mirror_base_mutation_suite(PINNED_MIRROR_BASE, tmp_path, monkeypatch)
    assert counts == {"refusal_passes": 10, "positive_passes": 2}

    # Literal https:// allow-list entry with scheme-case and multi-slash normalisation also works on a paid window
    _content, work, _port = _setup_offrig_manifest_env(tmp_path / "literal-allow", monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)
    orig_urlopen = pw.urllib.request.urlopen
    seen_urls: list[str] = []
    pinned_parts = pw.urllib.parse.urlsplit(PINNED_MIRROR_BASE)
    with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:

        def intercept_mirror_https_to_local(req_or_url, *a, **kw):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
            parts = pw.urllib.parse.urlsplit(url)
            if (
                (parts.scheme, parts.hostname, parts.port)
                == (pinned_parts.scheme, pinned_parts.hostname, pinned_parts.port)
                and parts.path.startswith(f"{pinned_parts.path}/")
            ):
                seen_urls.append(url)
                rel_path = parts.path[len(pinned_parts.path) :]
                local_url = f"{mir.base_url}{rel_path}" + (f"?{parts.query}" if parts.query else "")
                headers = dict(req_or_url.header_items()) if hasattr(req_or_url, "header_items") else {}
                return orig_urlopen(pw.urllib.request.Request(local_url, headers=headers), *a, **kw)
            return orig_urlopen(req_or_url, *a, **kw)

        monkeypatch.setattr(pw.urllib.request, "urlopen", intercept_mirror_https_to_local)
        monkeypatch.setattr(pw, "OLLAMA_MIRROR_ALLOWLIST", (PINNED_MIRROR_BASE,))
        shutil.rmtree(work / "ollama-models", ignore_errors=True)
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", f"  HTTPS://{PINNED_MIRROR_BASE.removeprefix('https://')}///  ")
        res_paid: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_paid)
        assert res_paid["offrig_manifest"]["mirror"] == {
            "base": PINNED_MIRROR_BASE,
            "revision": PINNED_MIRROR_COMMIT,
        }
        assert len(seen_urls) == 4
        assert mir.auth_matched == [True, True, True, True]


def test_offrig_bench_ollama_mirror_real_pin_allowlist_and_mutations(tmp_path, monkeypatch):
    real_base = (os.environ.get("ROLE_SET_TEST_MIRROR_BASE") or "").strip()
    if not real_base:
        pytest.skip("ROLE_SET_TEST_MIRROR_BASE unset")
    computed_entry = f"sha256:{pw.hashlib.sha256(real_base.encode('utf-8')).hexdigest()}"
    assert pw.OLLAMA_MIRROR_ALLOWLIST == (PINNED_MIRROR_SHA256,)
    assert computed_entry == PINNED_MIRROR_SHA256
    counts = _run_mirror_base_mutation_suite(real_base, tmp_path, monkeypatch)
    assert counts == {"refusal_passes": 10, "positive_passes": 2}
    print(
        f"REAL_PIN_MUTATION_SUITE refusal_passes={counts['refusal_passes']} "
        f"positive_passes={counts['positive_passes']} "
        f"total_passes={counts['refusal_passes'] + counts['positive_passes']} "
        f"checked_sha256={computed_entry}"
    )


def _start_redirect_test_servers(
    *,
    expected_token: str = MIRROR_SENTINEL,
    revision: str = PINNED_MIRROR_COMMIT,
    redirect_mode: str = "cross_host",
    redirect_target_override: str | None = None,
    cdn_blob_overrides: dict[str, bytes] | None = None,
    cdn_reject_auth: bool = True,
):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    blobs: dict[str, bytes] = {
        p.name.removeprefix("sha256-"): p.read_bytes()
        for p in sorted(FIXTURE_MANIFEST_DIR.glob("sha256-*"))
        if p.is_file()
    }
    cdn_blobs = {**blobs, **(cdn_blob_overrides or {})}
    resolve_prefix = f"/studio/role-set-ollama-pins/resolve/{revision}"

    class _Reuse(ThreadingHTTPServer):
        allow_reuse_address = True

    class _TestCDN:
        def __init__(self):
            self.requests: list[str] = []
            self.saw_auth: list[bool] = []
            cdn = self

            class CDNHandler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_GET(self):  # noqa: N802
                    cdn.requests.append(self.path)
                    has_auth = any(k.lower() == "authorization" for k in self.headers.keys())
                    cdn.saw_auth.append(has_auth)
                    if cdn_reject_auth and has_auth:
                        self.send_response(400)
                        self.end_headers()
                        return
                    for known_hex, data in cdn_blobs.items():
                        if self.path == f"/cdn-blobs/sha256-{known_hex}":
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                    self.send_response(404)
                    self.end_headers()

            self.server = _Reuse(("127.0.0.1", 0), CDNHandler)
            self.host_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self) -> None:
            self.server.shutdown()
            self.server.server_close()

    cdn_srv = _TestCDN()

    class _TestOrigin:
        def __init__(self):
            self.requests: list[str] = []
            self.auth_matched: list[bool] = []
            self.cdn = cdn_srv
            orig = self

            class OriginHandler(BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_GET(self):  # noqa: N802
                    orig.requests.append(self.path)
                    auth = self.headers.get("Authorization")
                    matched = auth == f"Bearer {expected_token}"
                    orig.auth_matched.append(matched)
                    if not matched:
                        self.send_response(401)
                        self.end_headers()
                        return
                    req_path = self.path.split("?", 1)[0]
                    for idx, (known_hex, data) in enumerate(blobs.items()):
                        if req_path == f"{resolve_prefix}/blobs/sha256-{known_hex}":
                            if redirect_target_override is not None:
                                self.send_response(302)
                                self.send_header(
                                    "Location", redirect_target_override.format(hex=known_hex, base=orig.host_url)
                                )
                                self.end_headers()
                                return
                            if redirect_mode == "cross_host":
                                code = (301, 302, 303, 307, 308)[idx % 5]
                                self.send_response(code)
                                self.send_header("Location", f"{cdn_srv.host_url}/cdn-blobs/sha256-{known_hex}")
                                self.end_headers()
                                return
                            if redirect_mode == "same_host_then_cdn":
                                loc = (
                                    f"/same-host-hop/blobs/sha256-{known_hex}"
                                    if idx % 2 == 0
                                    else f"{orig.host_url}/same-host-hop/blobs/sha256-{known_hex}"
                                )
                                self.send_response(302)
                                self.send_header("Location", loc)
                                self.end_headers()
                                return
                            if redirect_mode == "signed_hop_missing_location":
                                self.send_response(302)
                                self.send_header(
                                    "Location",
                                    f"{orig.host_url}/same-host-hop/blobs/sha256-{known_hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
                                )
                                self.end_headers()
                                return
                            if redirect_mode == "loop":
                                self.send_response(302)
                                self.send_header("Location", f"{resolve_prefix}/blobs/sha256-{known_hex}")
                                self.end_headers()
                                return
                        if req_path == f"/same-host-hop/blobs/sha256-{known_hex}":
                            if redirect_mode == "signed_hop_missing_location":
                                self.send_response(302)
                                self.end_headers()
                                return
                            if idx == len(blobs) - 1:
                                self.send_response(302)
                                self.send_header("Location", f"{cdn_srv.host_url}/cdn-blobs/sha256-{known_hex}")
                                self.end_headers()
                                return
                            self.send_response(200)
                            self.send_header("Content-Type", "application/octet-stream")
                            self.send_header("Content-Length", str(len(data)))
                            self.end_headers()
                            self.wfile.write(data)
                            return
                    self.send_response(404)
                    self.end_headers()

            self.server = _Reuse(("127.0.0.1", 0), OriginHandler)
            self.host_url = f"http://127.0.0.1:{self.server.server_address[1]}"
            self.base_url = f"{self.host_url}{resolve_prefix}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def close(self) -> None:
            try:
                self.server.shutdown()
                self.server.server_close()
            finally:
                self.cdn.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

    return _TestOrigin()


def test_offrig_bench_ollama_mirror_cross_host_redirect_strips_auth_header(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    with _start_redirect_test_servers(expected_token=MIRROR_SENTINEL, redirect_mode="cross_host") as srv:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
        res: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res)

        assert len(srv.requests) == 4
        assert srv.auth_matched == [True, True, True, True]
        assert len(srv.cdn.requests) == 4
        assert srv.cdn.saw_auth == [False, False, False, False]

    parsed = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    for desc in [parsed["config"], *parsed["layers"]]:
        bhex = desc["digest"].removeprefix("sha256:")
        written_blob = work / "ollama-models" / "blobs" / f"sha256-{bhex}"
        assert written_blob.read_bytes() == (FIXTURE_MANIFEST_DIR / f"sha256-{bhex}").read_bytes()

    # Paid window (`PIN_STUB` unset): `https://huggingface.co/...` redirects 302 to `https://us.aws.cdn.hf.co:443/...`
    # and strips `Authorization` on the CDN hop
    monkeypatch.delenv("PIN_STUB", raising=False)
    pinned_test_sha = f"sha256:{pw.hashlib.sha256(PINNED_MIRROR_BASE.encode('utf-8')).hexdigest()}"
    monkeypatch.setattr(pw, "OLLAMA_MIRROR_ALLOWLIST", (pinned_test_sha,))
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    orig_urlopen = pw.urllib.request.urlopen
    cdn_https_headers: list[dict[str, str]] = []
    pinned_parts = pw.urllib.parse.urlsplit(PINNED_MIRROR_BASE)
    with _start_redirect_test_servers(
        expected_token=MIRROR_SENTINEL,
        redirect_target_override="https://us.aws.cdn.hf.co:443/cdn-blobs/sha256-{hex}",
    ) as srv_paid:

        def route_paid_https_to_local(req_or_url, *a, _srv=srv_paid, **kw):
            url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
            hdrs = dict(req_or_url.header_items()) if hasattr(req_or_url, "header_items") else {}
            parts = pw.urllib.parse.urlsplit(url)
            if (
                (parts.scheme, parts.hostname, parts.port)
                == (pinned_parts.scheme, pinned_parts.hostname, pinned_parts.port)
                and parts.path.startswith(f"{pinned_parts.path}/")
            ):
                rel_path = parts.path[len(pinned_parts.path) :]
                local_url = f"{_srv.base_url}{rel_path}" + (f"?{parts.query}" if parts.query else "")
                return orig_urlopen(pw.urllib.request.Request(local_url, headers=hdrs), *a, **kw)
            if (parts.scheme, parts.hostname, parts.port) == ("https", "us.aws.cdn.hf.co", 443):
                cdn_https_headers.append(hdrs)
                local_cdn_url = f"{_srv.cdn.host_url}{parts.path}" + (f"?{parts.query}" if parts.query else "")
                return orig_urlopen(pw.urllib.request.Request(local_cdn_url, headers=hdrs), *a, **kw)
            return orig_urlopen(req_or_url, *a, **kw)

        monkeypatch.setattr(pw.urllib.request, "urlopen", route_paid_https_to_local)
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", PINNED_MIRROR_BASE)
        res_paid: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_paid)
        assert srv_paid.auth_matched == [True, True, True, True]
        assert srv_paid.cdn.saw_auth == [False, False, False, False]
        assert len(cdn_https_headers) == 4
        assert all(not any(k.lower() == "authorization" for k in h) for h in cdn_https_headers)


def test_offrig_bench_ollama_mirror_same_host_redirect_keeps_auth_header(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    with _start_redirect_test_servers(expected_token=MIRROR_SENTINEL, redirect_mode="same_host_then_cdn") as srv:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
        res: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res)

        # 4 initial origin requests + 4 same-host redirected requests all keep Authorization
        assert len(srv.requests) == 8
        assert srv.auth_matched == [True] * 8
        # The final blob's second hop goes cross-host to the CDN and strips Authorization
        assert len(srv.cdn.requests) == 1
        assert srv.cdn.saw_auth == [False]
    assert (work / "ollama-models" / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "31b").is_file()


def test_offrig_bench_ollama_mirror_refuses_non_hf_redirect_before_network(tmp_path, monkeypatch):
    _content, _work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    orig_opener_open = pw.urllib.request.OpenerDirector._open
    leaked_targets: list[str] = []

    for bad_target in (
        "https://evil.example.com/blobs/sha256-{hex}",
        "https://huggingface.co.evil.example.com/blobs/sha256-{hex}",
        "https://evil-hf.co/blobs/sha256-{hex}",
        "https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:{hex}",
        "http://huggingface.co/blobs/sha256-{hex}",
        "https://user@huggingface.co/blobs/sha256-{hex}",
        "https://huggingface.co:8443/blobs/sha256-{hex}",
    ):
        leaked_targets.clear()
        with _start_redirect_test_servers(
            expected_token=MIRROR_SENTINEL,
            redirect_target_override=bad_target,
        ) as srv:
            srv_parts = pw.urllib.parse.urlsplit(srv.host_url)

            def guard_opener_open(self_opener, req, data=None, _sp=srv_parts):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                parts = pw.urllib.parse.urlsplit(url)
                if (parts.scheme, parts.hostname, parts.port) != (_sp.scheme, _sp.hostname, _sp.port):
                    leaked_targets.append(url)
                    raise AssertionError(f"request must never be sent to disallowed redirect target {url}")
                return orig_opener_open(self_opener, req, data)

            monkeypatch.setattr(pw.urllib.request.OpenerDirector, "_open", guard_opener_open)
            pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
            monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
            with pytest.raises(RuntimeError, match="REFUSED: mirror redirect target"):
                pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
            assert leaked_targets == []

    # Signed URLs (?X-Amz-Signature=SECRET#FRAG_SECRET) in a refused redirect target or missing-Location redirect never leak into RuntimeError, say(), or pin-result fields
    say_lines: list[str] = []
    monkeypatch.setattr(pw, "say", say_lines.append)
    for signed_bad_target in (
        "https://evil.example.com/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "https://registry.ollama.ai/v2/library/gemma4/blobs/sha256:{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "http://huggingface.co/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "https://user@huggingface.co/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "https://huggingface.co:8443/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "https://huggingface.co:bad/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
        "https://huggingface.co:/blobs/sha256-{hex}?X-Amz-Signature=SECRET#FRAG_SECRET",
    ):
        say_lines.clear()
        with _start_redirect_test_servers(
            expected_token=MIRROR_SENTINEL,
            redirect_target_override=signed_bad_target,
        ) as srv_signed:
            srv_parts = pw.urllib.parse.urlsplit(srv_signed.host_url)

            def guard_signed_open(self_opener, req, data=None, _sp=srv_parts):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                parts = pw.urllib.parse.urlsplit(url)
                if (parts.scheme, parts.hostname, parts.port) != (_sp.scheme, _sp.hostname, _sp.port):
                    raise AssertionError(f"request must never be sent to disallowed redirect target {url}")
                return orig_opener_open(self_opener, req, data)

            monkeypatch.setattr(pw.urllib.request.OpenerDirector, "_open", guard_signed_open)
            pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
            monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv_signed.base_url)
            res_signed: dict = {}
            with pytest.raises(RuntimeError, match="REFUSED: mirror redirect target") as exc_info:
                pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_signed)
            err_msg = str(exc_info.value)
            assert "X-Amz-Signature" not in err_msg
            assert "SECRET" not in err_msg
            assert "FRAG_SECRET" not in err_msg
            res_dump = json.dumps(res_signed)
            assert "X-Amz-Signature" not in res_dump and "SECRET" not in res_dump
            assert all("X-Amz-Signature" not in ln and "SECRET" not in ln for ln in say_lines)

    # Hop-2 redirect missing Location when cur_url carries ?X-Amz-Signature=SECRET#FRAG_SECRET also strips query and fragment
    monkeypatch.setattr(pw.urllib.request.OpenerDirector, "_open", orig_opener_open)
    say_lines.clear()
    with _start_redirect_test_servers(
        expected_token=MIRROR_SENTINEL,
        redirect_mode="signed_hop_missing_location",
    ) as srv_missing_loc:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv_missing_loc.base_url)
        res_missing_loc: dict = {}
        with pytest.raises(RuntimeError, match="REFUSED: .* redirect HTTP 302 .* missing Location") as exc_loc:
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_missing_loc)
        loc_msg = str(exc_loc.value)
        assert "X-Amz-Signature" not in loc_msg
        assert "SECRET" not in loc_msg
        assert "FRAG_SECRET" not in loc_msg
        loc_dump = json.dumps(res_missing_loc)
        assert "X-Amz-Signature" not in loc_dump and "SECRET" not in loc_dump
        assert all("X-Amz-Signature" not in ln and "SECRET" not in ln for ln in say_lines)

    # Paid window (`PIN_STUB` unset) also refuses a redirect to loopback `127.0.0.1` and refuses redirect loops (> 5 hops)
    monkeypatch.setattr(pw.urllib.request.OpenerDirector, "_open", orig_opener_open)
    with _start_redirect_test_servers(expected_token=MIRROR_SENTINEL, redirect_mode="loop") as srv_loop:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv_loop.base_url)
        with pytest.raises(RuntimeError, match="REFUSED: .* exceeded max redirects"):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})


def test_offrig_bench_ollama_mirror_cross_host_redirect_enforces_size_and_sha256(tmp_path, monkeypatch):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    parsed = json.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    cfg_hex = parsed["config"]["digest"].removeprefix("sha256:")
    cfg_size = int(parsed["config"]["size"])

    # 1. Cross-host CDN returns wrong byte size -> refused with size mismatch
    with _start_redirect_test_servers(
        expected_token=MIRROR_SENTINEL,
        redirect_mode="cross_host",
        cdn_blob_overrides={cfg_hex: (FIXTURE_MANIFEST_DIR / f"sha256-{cfg_hex}").read_bytes() + b"extra"},
    ) as srv:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
        with pytest.raises(RuntimeError, match="size mismatch"):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        assert srv.cdn.saw_auth == [False]
    assert not (work / "ollama-models" / "blobs" / f"sha256-{cfg_hex}").exists()
    assert not (work / "ollama-models" / "blobs" / f"sha256-{cfg_hex}.part").exists()

    # 2. Cross-host CDN returns exact expected byte size but wrong sha256 -> refused with sha256 mismatch
    with _start_redirect_test_servers(
        expected_token=MIRROR_SENTINEL,
        redirect_mode="cross_host",
        cdn_blob_overrides={cfg_hex: b"X" * cfg_size},
    ) as srv:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
        with pytest.raises(RuntimeError, match=r"sha256 mismatch: [0-9a-f]{12} != [0-9a-f]{12}"):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        assert srv.cdn.saw_auth == [False]
    assert not (work / "ollama-models" / "blobs" / f"sha256-{cfg_hex}").exists()
    assert not (work / "ollama-models" / "blobs" / f"sha256-{cfg_hex}.part").exists()


def test_offrig_bench_ollama_mirror_missing_token_non_hex_revision_and_window_prechecks(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    seen_urls: list[str] = []
    orig_urlopen = pw.urllib.request.urlopen

    def fail_on_network(req_or_url, *a, **kw):
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        if "/blobs/" in pw.urllib.parse.urlsplit(url).path:
            seen_urls.append(url)
            raise AssertionError(f"unexpected network call to {url}")
        return orig_urlopen(req_or_url, *a, **kw)

    monkeypatch.setattr(pw.urllib.request, "urlopen", fail_on_network)

    # 1. PIN_OLLAMA_MIRROR set with no token file -> refused before network
    pinned_test_sha = f"sha256:{pw.hashlib.sha256(PINNED_MIRROR_BASE.encode('utf-8')).hexdigest()}"
    monkeypatch.setattr(pw, "OLLAMA_MIRROR_ALLOWLIST", (pinned_test_sha,))
    pw.TOKEN_PATH.unlink(missing_ok=True)
    pw.take_token()
    monkeypatch.setenv("PIN_OLLAMA_MIRROR", PINNED_MIRROR_BASE)
    res_no_tok: dict = {}
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OLLAMA_MIRROR is set, but no token file reached the VM"):
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_no_tok)
    assert seen_urls == []
    assert all(res_no_tok["token_clean"].values())

    # 2. Non-40-hex /resolve/ revision refused both when listed in OLLAMA_MIRROR_ALLOWLIST (paid) and in stub mode
    for bad_rev in ("main", "9dfbad5", f"{PINNED_MIRROR_COMMIT}00", PINNED_MIRROR_COMMIT.upper()):
        bad_paid_url = f"https://huggingface.co/studio/role-set-ollama-pins/resolve/{bad_rev}"
        bad_paid_sha = f"sha256:{pw.hashlib.sha256(bad_paid_url.encode('utf-8')).hexdigest()}"
        monkeypatch.setattr(pw, "OLLAMA_MIRROR_ALLOWLIST", (bad_paid_url, bad_paid_sha))
        monkeypatch.delenv("PIN_STUB", raising=False)
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", bad_paid_url)
        with pytest.raises(
            RuntimeError,
            match=r"REFUSED: PIN_OLLAMA_MIRROR=.* revision .* must be a 40-char lowercase hex commit sha",
        ):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        assert seen_urls == []

        monkeypatch.setenv("PIN_STUB", "1")
        bad_stub_url = f"http://127.0.0.1:9999/studio/role-set-ollama-pins/resolve/{bad_rev}"
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", bad_stub_url)
        with pytest.raises(
            RuntimeError,
            match=r"REFUSED: PIN_OLLAMA_MIRROR=.* revision .* must be a 40-char lowercase hex commit sha",
        ):
            pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, {})
        assert seen_urls == []

    # 3. offrig_bench_window refuses when both PIN_OLLAMA_MIRROR and PIN_OLLAMA_REGISTRY are set, and records token_clean in finally
    monkeypatch.setenv("PIN_STUB", "1")
    pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
    monkeypatch.setenv("PIN_OLLAMA_MIRROR", PINNED_MIRROR_BASE)
    monkeypatch.setenv("PIN_OLLAMA_REGISTRY", "https://registry.ollama.ai")
    res_both: dict = {}
    with pytest.raises(RuntimeError, match="REFUSED: PIN_OLLAMA_MIRROR and PIN_OLLAMA_REGISTRY cannot both be set"):
        pw.offrig_bench_window(res_both)
    assert seen_urls == []
    assert pw._TOKEN is None and not pw.TOKEN_PATH.exists()
    assert res_both.get("token_clean") == {
        "file_gone": True,
        "value_dropped": True,
        "env_clear": True,
        "no_cached_token": True,
        "no_colab_secret": True,
    }
    monkeypatch.delenv("PIN_OLLAMA_REGISTRY", raising=False)

    # 4. offrig_bench_window's pre-wave token-residue check refuses before any blob download starts
    for residue_kind, expected_key in (
        ("env", "env_clear"),
        ("cache", "no_cached_token"),
        ("colab", "no_colab_secret"),
        ("file", "file_gone"),
    ):
        _unlock_offrig_in(work)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delitem(sys.modules, "huggingface_hub.utils._auth", raising=False)
        clean_home = tmp_path / f"hf-prewave-{residue_kind}"
        clean_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HF_HOME", str(clean_home))
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)
        if residue_kind == "env":
            monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "residue")
        elif residue_kind == "cache":
            (clean_home / "token").write_text("residue\n", encoding="utf-8")
        elif residue_kind == "colab":
            monkeypatch.setitem(
                sys.modules, "huggingface_hub.utils._auth", SimpleNamespace(_GOOGLE_COLAB_SECRET="residue")
            )
        else:
            pw.TOKEN_PATH.write_text("residue\n", encoding="utf-8")

        res_pre: dict = {}
        with pytest.raises(RuntimeError, match=rf"a token residue is left before any wave \(.*{expected_key}"):
            pw.offrig_bench_window(res_pre)
        assert seen_urls == []
        assert res_pre["token_clean"][expected_key] is False
        assert res_pre["token_clean"]["value_dropped"] is True
        assert pw._TOKEN is None
        pw.TOKEN_PATH.unlink(missing_ok=True)


def test_offrig_bench_ollama_mirror_token_travels_only_in_header_never_in_url_log_or_pin_result(
    tmp_path, monkeypatch
):
    content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_CONTENT", str(content))
    monkeypatch.setenv("PIN_WORK", str(work))
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)
    monkeypatch.setattr(pw, "take_job_lock", lambda _w: open(work / "job.lock", "w"))  # noqa: SIM115

    # Restore real `say` so job.log captures every log line written during `pw.main()`
    monkeypatch.setattr(
        pw,
        "say",
        lambda msg: print(f"[{__import__('time').strftime('%H:%M:%S')}] {msg}", flush=True),
    )

    seen_request_urls: list[str] = []
    seen_request_headers: list[dict[str, str]] = []
    orig_urlopen = pw.urllib.request.urlopen

    def inspect_requests(req_or_url, *a, **kw):
        url = req_or_url.full_url if hasattr(req_or_url, "full_url") else str(req_or_url)
        hdrs = dict(req_or_url.header_items()) if hasattr(req_or_url, "header_items") else {}
        if "/blobs/" in pw.urllib.parse.urlsplit(url).path:
            seen_request_urls.append(url)
            seen_request_headers.append(hdrs)
        return orig_urlopen(req_or_url, *a, **kw)

    monkeypatch.setattr(pw.urllib.request, "urlopen", inspect_requests)

    with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", mir.base_url)
        rc = pw.main()

    assert rc == 0
    assert len(seen_request_urls) == 4
    # Token is in the Authorization header on every mirror blob request
    assert all(
        h.get("Authorization") == f"Bearer {MIRROR_SENTINEL}" or h.get("authorization") == f"Bearer {MIRROR_SENTINEL}"
        for h in seen_request_headers
    )
    # Token is never in any URL
    assert all(MIRROR_SENTINEL not in u and not pw.TOKEN_SHAPE_RE.search(u) for u in seen_request_urls)

    # Token is never in job.log, pin-result.json, or any file under content / work
    job_log_text = (work / "job.log").read_text(encoding="utf-8")
    pin_result_text = (work / "pin-result.json").read_text(encoding="utf-8")
    assert MIRROR_SENTINEL not in job_log_text and not pw.TOKEN_SHAPE_RE.search(job_log_text)
    assert MIRROR_SENTINEL not in pin_result_text and not pw.TOKEN_SHAPE_RE.search(pin_result_text)
    for fpath in content.rglob("*"):
        if fpath.is_file():
            raw = fpath.read_bytes()
            assert MIRROR_SENTINEL.encode("utf-8") not in raw, fpath
            assert not pw.re.search(rb"hf_[A-Za-z0-9]{30,}", raw), fpath

    pin_res = json.loads(pin_result_text)
    assert pin_res["ok"] is True
    assert pin_res["offrig_manifest"]["mirror"] == {
        "base": mir.base_url,
        "revision": PINNED_MIRROR_COMMIT,
    }
    assert set(pin_res["offrig_manifest"]["mirror"].keys()) == {"base", "revision"}

    # In registry mode (PIN_OLLAMA_MIRROR unset), even if a token file was present on disk, NO Authorization header is ever sent
    _unlock_offrig_in(work)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    seen_request_urls.clear()
    seen_request_headers.clear()
    monkeypatch.delenv("PIN_OLLAMA_MIRROR", raising=False)
    with _start_test_blob_registry() as reg:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        monkeypatch.setenv("PIN_OLLAMA_REGISTRY", reg.base_url)
        monkeypatch.setattr(pw, "take_job_lock", lambda _w: open(work / "job.lock", "w"))  # noqa: SIM115
        assert pw.main() == 0
    assert len(seen_request_headers) == 4
    assert all("Authorization" not in h and "authorization" not in h for h in seen_request_headers)


def test_offrig_bench_ollama_mirror_token_clean_is_true_after_fetch_and_refuses_residue(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)

    # 1. After read_token_file(needed=True) + install_ollama_manifest on the mirror, token_clean is all True, TOKEN_PATH is gone, and _TOKEN is None
    with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)
        monkeypatch.setenv("PIN_OLLAMA_MIRROR", mir.base_url)
        res_ok: dict = {}
        pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_ok)

    assert not pw.token_file().exists()
    assert pw._TOKEN is None
    assert pw.token_clean() == {
        "file_gone": True,
        "value_dropped": True,
        "env_clear": True,
        "no_cached_token": True,
        "no_colab_secret": True,
    }
    assert res_ok["token_clean"] == pw.token_clean()
    assert all(res_ok["token_clean"].values())
    assert res_ok["offrig_manifest"]["mirror"]["revision"] == PINNED_MIRROR_COMMIT

    # 2. Any token residue (env, Hub token cache, Colab secret, or leftover file) fails token_clean
    for residue_kind, expected_key in (
        ("env", "env_clear"),
        ("cache", "no_cached_token"),
        ("colab", "no_colab_secret"),
        ("file", "file_gone"),
    ):
        _unlock_offrig_in(work)
        shutil.rmtree(work / "ollama-models", ignore_errors=True)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delitem(sys.modules, "huggingface_hub.utils._auth", raising=False)
        clean_home = tmp_path / f"hf-clean-{residue_kind}"
        clean_home.mkdir(exist_ok=True)
        monkeypatch.setenv("HF_HOME", str(clean_home))
        pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
        pw.read_token_file(needed=True)

        if residue_kind == "env":
            monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "residue")
        elif residue_kind == "cache":
            (clean_home / "token").write_text("residue\n", encoding="utf-8")
        elif residue_kind == "colab":
            monkeypatch.setitem(
                sys.modules, "huggingface_hub.utils._auth", SimpleNamespace(_GOOGLE_COLAB_SECRET="residue")
            )
        else:
            pw.TOKEN_PATH.write_text("residue\n", encoding="utf-8")

        with _start_test_blob_mirror(expected_token=MIRROR_SENTINEL) as mir:
            monkeypatch.setenv("PIN_OLLAMA_MIRROR", mir.base_url)
            res_res: dict = {}
            with pytest.raises(RuntimeError, match=rf"token residue is left before any wave \(.*{expected_key}"):
                pw.install_ollama_manifest(FIXTURE_MANIFEST_PATH, pw.OFFRIG_MODEL, FIXTURE_MANIFEST_SHA256, res_res)
            assert res_res["token_clean"][expected_key] is False
        assert pw._TOKEN is None
        pw.TOKEN_PATH.unlink(missing_ok=True)


def test_smoke_window_accepts_offrig_bench_ollama_pins_read_and_refuses_any_other_token(tmp_path):
    bash_bin = _find_git_bash()
    if not bash_bin:
        pytest.skip("bash not available")
    script = str(Path(__file__).resolve().parents[1] / "scripts" / "colab" / "smoke_window.sh")
    binfile = tmp_path / "offrig"
    binfile.write_bytes(b"x" * 64)

    # 1. smoke_window.sh accepts offrig-bench:HF_TOKEN_OLLAMA_PINS_READ (when empty on the rig, reaches the empty-var check at exit 3)
    out_empty = tmp_path / "out-empty"
    env_empty = {
        **os.environ,
        "PIN_GPU": "A100",
        "PIN_GATE_MIN_DISK_GB": "80",
        "PIN_ROLE": "offrig-bench",
        "PIN_HF_TOKEN_FROM": "HF_TOKEN_OLLAMA_PINS_READ",
        "HF_TOKEN_OLLAMA_PINS_READ": "",
    }
    r_empty = subprocess.run([bash_bin, script, str(binfile), str(out_empty)], env=env_empty, capture_output=True, text=True, timeout=30)
    combined_empty = r_empty.stdout + r_empty.stderr
    assert r_empty.returncode == 3, (r_empty.returncode, combined_empty)
    assert "PIN_HF_TOKEN_FROM names HF_TOKEN_OLLAMA_PINS_READ, which is empty on the rig" in combined_empty

    # 2. smoke_window.sh refuses offrig-bench with any token other than HF_TOKEN_OLLAMA_PINS_READ before any VM call
    for bad_tok in ("HF_TOKEN_GGUF_READ", "HF_TOKEN_GGUF_WRITE", "HF_TOKEN", "HF_TOKEN_OTHER"):
        out_dir = tmp_path / f"out-{bad_tok}"
        env = {
            **os.environ,
            "PIN_GPU": "A100",
            "PIN_GATE_MIN_DISK_GB": "80",
            "PIN_ROLE": "offrig-bench",
            "PIN_HF_TOKEN_FROM": bad_tok,
            bad_tok: MIRROR_SENTINEL,
        }
        r = subprocess.run([bash_bin, script, str(binfile), str(out_dir)], env=env, capture_output=True, text=True, timeout=30)
        combined = r.stdout + r.stderr
        assert r.returncode == 2, (bad_tok, r.returncode, combined)
        assert f"REFUSED: PIN_ROLE=offrig-bench may not send {bad_tok}" in combined
        assert "offrig-bench: HF_TOKEN_OLLAMA_PINS_READ" in combined

    # 3. smoke_window.sh refuses HF_TOKEN_OLLAMA_PINS_READ for any role other than offrig-bench
    for other_role in ("pilot", "critic-eval", "prebuild", "smoke"):
        out_dir = tmp_path / f"out-role-{other_role}"
        env = {
            **os.environ,
            "PIN_GPU": "A100",
            "PIN_GATE_MIN_DISK_GB": "80",
            "PIN_ROLE": other_role,
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_OLLAMA_PINS_READ",
            "HF_TOKEN_OLLAMA_PINS_READ": MIRROR_SENTINEL,
        }
        r = subprocess.run([bash_bin, script, str(binfile), str(out_dir)], env=env, capture_output=True, text=True, timeout=30)
        combined = r.stdout + r.stderr
        assert r.returncode == 2, (other_role, r.returncode, combined)
        assert f"REFUSED: PIN_ROLE={other_role} may not send HF_TOKEN_OLLAMA_PINS_READ" in combined


def test_offrig_bench_with_mirror_unset_behaviour_unchanged_from_a621a793(tmp_path, monkeypatch):
    _content, work, free_port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=False)
    monkeypatch.delenv("PIN_OLLAMA_MIRROR", raising=False)

    # 1. Manifest mode with PIN_OLLAMA_MIRROR unset uses registry.ollama.ai (or PIN_OLLAMA_REGISTRY), needs no token, and records `registry`
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)
    with _start_test_blob_registry() as reg:
        monkeypatch.setattr(pw, "_resolve_ollama_registry_base", lambda _raw=None: (reg.base_url, None))
        res_reg: dict = {}
        pw.offrig_bench_window(res_reg)
        assert len(reg.requests) == 8  # 4 v2 blob requests + 4 redirected _blobs requests
    assert res_reg["offrig_manifest"]["registry"] == reg.base_url
    assert "mirror" not in res_reg["offrig_manifest"]
    assert all(res_reg["token_clean"].values())
    assert res_reg["offrig_model"]["digest"] == FIXTURE_MANIFEST_SHA256

    # 2. Non-manifest mode with PIN_OLLAMA_MIRROR unset runs `ollama pull` and records no `offrig_manifest`
    _unlock_offrig_in(work)
    shutil.rmtree(work / "ollama-models", ignore_errors=True)
    monkeypatch.delenv("PIN_OFFRIG_MANIFEST", raising=False)
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", pw.OFFRIG_MODEL_DIGEST)
    seen_proc_cmds: list[list[str]] = []
    orig_run_proc = pw.run_proc

    def spy_run_proc(cmd, **kw):
        if isinstance(cmd, list):
            seen_proc_cmds.append(list(cmd))
            if cmd[1:3] == ["pull", "gemma4:31b"]:
                assert kw["env"]["OLLAMA_HOST"] == f"127.0.0.1:{free_port}"
                return subprocess.CompletedProcess(cmd, 0, "pulling manifest\nsuccess\n", "")
        return orig_run_proc(cmd, **kw)

    monkeypatch.setattr(pw, "run_proc", spy_run_proc)
    res_pull: dict = {}
    pw.offrig_bench_window(res_pull)
    assert [c[1] for c in seen_proc_cmds] == ["pull", "verify"]
    assert "offrig_manifest" not in res_pull
    assert res_pull["offrig_model"]["digest"] == pw.OFFRIG_MODEL_DIGEST


def test_offrig_bench_mirror_non_http_exception_strips_signed_url_query_and_fragment(tmp_path, monkeypatch, capsys):
    _content, work, _port = _setup_offrig_manifest_env(tmp_path, monkeypatch, stub=True)
    monkeypatch.setenv("PIN_ROLE", "offrig-bench")
    monkeypatch.setenv("PIN_OFFRIG_MANIFEST", str(FIXTURE_MANIFEST_PATH))
    monkeypatch.setenv("PIN_OFFRIG_MODEL_DIGEST", FIXTURE_MANIFEST_SHA256)
    monkeypatch.setenv("PIN_HF_TOKEN_FROM", "HF_TOKEN_OLLAMA_PINS_READ")

    signed_cdn_url = (
        "https://cas-bridge.xethub.hf.co/xet-bridge-us/blobs/sha256-deadbeef"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=SECRET#FRAG_SECRET"
    )
    for exc_factory in (
        lambda: pw.urllib.error.URLError(f"TLS EOF on {signed_cdn_url}"),
        lambda: OSError(f"connection reset by peer for {signed_cdn_url} ({MIRROR_SENTINEL})"),
        lambda: ValueError(f"invalid chunk header from {signed_cdn_url}"),
        lambda: RuntimeError(f"proxy error fetching {signed_cdn_url}"),
    ):
        with _start_redirect_test_servers(expected_token=MIRROR_SENTINEL) as srv:
            monkeypatch.setenv("PIN_OLLAMA_MIRROR", srv.base_url)
            orig_open_no_redirect = pw._open_no_redirect
            hop_count = 0

            def fail_on_cdn_hop(req, *, timeout, _make_exc=exc_factory, _orig=orig_open_no_redirect):
                nonlocal hop_count
                hop_count += 1
                if hop_count >= 2:
                    raise _make_exc()
                return _orig(req, timeout=timeout)

            monkeypatch.setattr(pw, "_open_no_redirect", fail_on_cdn_hop)
            pw.TOKEN_PATH.write_text(MIRROR_SENTINEL + "\n", encoding="utf-8")
            capsys.readouterr()
            rc = pw._main()
            out_log = capsys.readouterr().out
            assert rc == 1
            pin_res = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
            assert pin_res["ok"] is False
            failed_reason = pin_res["offrig_manifest"]["failed_blob"]["reason"]
            err_field = pin_res["error"]
            assert "https://cas-bridge.xethub.hf.co/xet-bridge-us/blobs/sha256-deadbeef" in failed_reason
            assert "https://cas-bridge.xethub.hf.co/xet-bridge-us/blobs/sha256-deadbeef" in err_field
            for surface in (failed_reason, err_field, json.dumps(pin_res), out_log):
                assert "X-Amz-Signature" not in surface
                assert "SECRET" not in surface
                assert "FRAG_SECRET" not in surface
                assert MIRROR_SENTINEL not in surface


def test_pilot_window_wires_per_arm_t10_settings_records_in_pin_result_and_refuses_malformed(tmp_path, monkeypatch):
    """FIX 5: PIN_SIDE_PAIRING, PIN_DOMAIN_SCOPE, PIN_CANNOT_TELL_CONSTRUCTION, and PIN_POST_REFUSAL_SPAN
    are wired for pilot_window's write, critique, and rewrite waves; unset is default and byte-identical to today;
    each value is recorded in pin-result.json; and malformed values are refused."""
    from subprocess import CompletedProcess

    for var in ("PIN_SIDE_PAIRING", "PIN_DOMAIN_SCOPE", "PIN_CANNOT_TELL_CONSTRUCTION", "PIN_POST_REFUSAL_SPAN", "PIN_VERDICT_PROMPT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("PIN_CRITIQUE_SCOPE", raising=False)
    monkeypatch.delenv("PIN_DEADLINE", raising=False)

    seen_cmds: list[list[str]] = []

    def fake_run_proc(cmd, **kw):
        seen_cmds.append(list(cmd))
        return CompletedProcess(cmd, 0, '{"dropped": 0, "states": {"awaiting_critique": 0, "needs_rewrite": 0}}\n', "")

    monkeypatch.setattr(pw, "run_proc", fake_run_proc)
    rs = {"role_set": "role-set", "src": str(tmp_path)}
    writer_a = {"served_as": "qwen3.6-35b-a3b", "gguf_sha256": "a" * 64}
    writer_b = {"served_as": "muse-glimmer-30b", "gguf_sha256": "b" * 64}
    critic = {"served_as": "nemotron-3.5-lightning-30b-a3b", "gguf_sha256": "c" * 64}

    # 1. Unset means default and byte-identical to today across write, critique, and rewrite waves.
    assert pw.side_pairing() == "default"
    assert pw.domain_scope() == "default"
    assert pw.cannot_tell_construction() == "default"
    assert pw.post_refusal_span() is False
    pw.write_wave(rs, ["--plan", "p.json"], "write", tmp_path / "qwen3.6-35b-a3b" / "w0", writer=writer_a)
    pw.write_wave(rs, ["--plan", "p.json"], "critique", tmp_path / "qwen3.6-35b-a3b" / "w1", critic=critic, drafts=tmp_path / "d.jsonl")
    pw.write_wave(rs, ["--plan", "p.json"], "rewrite", tmp_path / "qwen3.6-35b-a3b" / "w2", writer=writer_a, drafts=tmp_path / "d.jsonl")
    for cmd in seen_cmds:
        assert "--side-pairing" not in cmd
        assert "--domain-scope" not in cmd
        assert "--cannot-tell-construction" not in cmd
        assert "--post-refusal-span" not in cmd

    # 2. Per-arm settings pass only on the configured arm across write, critique, and rewrite waves.
    seen_cmds.clear()
    monkeypatch.setenv("PIN_SIDE_PAIRING", "qwen3.6-35b-a3b=default,muse-glimmer-30b=verdict_safe")
    monkeypatch.setenv("PIN_DOMAIN_SCOPE", "qwen3.6-35b-a3b=default,muse-glimmer-30b=framing_only")
    monkeypatch.setenv("PIN_CANNOT_TELL_CONSTRUCTION", "qwen3.6-35b-a3b=default,muse-glimmer-30b=coarse_vs_sharp")
    monkeypatch.setenv("PIN_POST_REFUSAL_SPAN", "qwen3.6-35b-a3b=0,muse-glimmer-30b=1")

    arms_list = ["qwen3.6-35b-a3b", "muse-glimmer-30b"]
    assert pw.side_pairing(writer_names=arms_list) == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "verdict_safe"}
    assert pw.domain_scope(writer_names=arms_list) == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "framing_only"}
    assert pw.cannot_tell_construction(writer_names=arms_list) == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "coarse_vs_sharp"}
    assert pw.post_refusal_span(writer_names=arms_list) == {"qwen3.6-35b-a3b": False, "muse-glimmer-30b": True}

    for wave_idx, wave_name in enumerate(("write", "critique", "rewrite")):
        pw.write_wave(
            rs,
            ["--plan", "p.json"],
            wave_name,
            tmp_path / "qwen3.6-35b-a3b" / f"w{wave_idx}",
            writer=writer_a if wave_name != "critique" else None,
            critic=critic if wave_name == "critique" else None,
            drafts=tmp_path / "d.jsonl" if wave_name != "write" else None,
        )
        pw.write_wave(
            rs,
            ["--plan", "p.json"],
            wave_name,
            tmp_path / "muse-glimmer-30b" / f"w{wave_idx}",
            writer=writer_b if wave_name != "critique" else None,
            critic=critic if wave_name == "critique" else None,
            drafts=tmp_path / "d.jsonl" if wave_name != "write" else None,
        )

    for i in range(0, 6, 2):
        cmd_a, cmd_b = seen_cmds[i], seen_cmds[i + 1]
        assert "--side-pairing" not in cmd_a
        assert "--domain-scope" not in cmd_a
        assert "--cannot-tell-construction" not in cmd_a
        assert "--post-refusal-span" not in cmd_a
        assert cmd_b[cmd_b.index("--side-pairing") + 1] == "verdict_safe"
        assert cmd_b[cmd_b.index("--domain-scope") + 1] == "framing_only"
        assert cmd_b[cmd_b.index("--cannot-tell-construction") + 1] == "coarse_vs_sharp"
        assert "--post-refusal-span" in cmd_b

    # 3. Malformed values are refused before check_pins / prefetch / serve in pilot_window.
    content = tmp_path / "content"
    work = tmp_path / "work"
    content.mkdir()
    work.mkdir()
    (content / "pack").mkdir()
    (content / "pack" / "pack.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "_disk_target", lambda: tmp_path)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs)
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(pw, "sh", lambda cmd, **kw: "ok")
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: pytest.fail("check_pins must not run"))
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "b" * 40)
    monkeypatch.setenv("PIN_PILOT_WRITERS", "qwen3.6-35b-a3b,muse-glimmer-30b")

    for bad_var, bad_val in (
        ("PIN_SIDE_PAIRING", "bogus"),
        ("PIN_SIDE_PAIRING", "unknown-arm=verdict_safe"),
        ("PIN_SIDE_PAIRING", "default,verdict_safe"),
        ("PIN_DOMAIN_SCOPE", "material_only"),
        ("PIN_CANNOT_TELL_CONSTRUCTION", "sharp_only"),
        ("PIN_POST_REFUSAL_SPAN", "maybe"),
    ):
        for v in ("PIN_SIDE_PAIRING", "PIN_DOMAIN_SCOPE", "PIN_CANNOT_TELL_CONSTRUCTION", "PIN_POST_REFUSAL_SPAN"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv(bad_var, bad_val)
        res_bad: dict = {"machine": {"gpu": "A100"}}
        with pytest.raises(RuntimeError, match=f"REFUSED:.*{bad_var}"):
            pw.pilot_window({}, res_bad)

    # 4. pilot_window records each value in result (pin-result.json) both when unset and when set per-arm.
    root_repo = Path(__file__).resolve().parents[1]
    rs_real = {"role_set": "role-set", "src": str(root_repo)}
    monkeypatch.setattr(pw, "role_set_env", lambda commit: rs_real)
    monkeypatch.setattr(pw, "check_pins", lambda *a, **kw: None)
    monkeypatch.setattr(pw, "prefetch_hosted", lambda *a, **kw: None)
    monkeypatch.setattr(pw, "server_sha256", lambda: "d" * 64)
    monkeypatch.setattr(pw, "server_pin_entry", lambda sha, wp=None: {"commit": "abc", "sha256": sha, "archs": "90"})
    monkeypatch.setattr(
        pw,
        "pilot_batch",
        lambda rs, common, plan, base, writer_names, critic_for, pool, **kw: {
            n: {"waves": [{"exit": 0}], "last": base / n / "w0"} for n in writer_names
        },
    )

    def fake_plan_sh(cmd: str, **kw) -> str:
        if " role-set plan " in f" {cmd} " or cmd.startswith("role-set plan "):
            out_flag = cmd.split("--out", 1)[1].strip().split()[0]
            Path(out_flag).write_text(
                json.dumps({"slots": [{"id": "s1"}], "exclusion_state_sha256": "e" * 64}),
                encoding="utf-8",
            )
        return "ok"

    monkeypatch.setattr(pw, "sh", fake_plan_sh)
    monkeypatch.setenv("PIN_PILOT_CRITIC", "nemotron-3.5-lightning-30b-a3b")
    monkeypatch.setenv("PIN_PILOT_TWINS", "0")
    for v in ("PIN_SIDE_PAIRING", "PIN_DOMAIN_SCOPE", "PIN_CANNOT_TELL_CONSTRUCTION", "PIN_POST_REFUSAL_SPAN"):
        monkeypatch.delenv(v, raising=False)

    res_unset: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, res_unset)
    assert res_unset["side_pairing"] == "default"
    assert res_unset["domain_scope"] == "default"
    assert res_unset["cannot_tell_construction"] == "default"
    assert res_unset["post_refusal_span"] is False
    for arm_name in ("qwen3.6-35b-a3b", "muse-glimmer-30b"):
        assert res_unset["arms"][arm_name]["side_pairing"] == "default"
        assert res_unset["arms"][arm_name]["domain_scope"] == "default"
        assert res_unset["arms"][arm_name]["cannot_tell_construction"] == "default"
        assert res_unset["arms"][arm_name]["post_refusal_span"] is False

    monkeypatch.setenv("PIN_SIDE_PAIRING", "qwen3.6-35b-a3b=default,muse-glimmer-30b=verdict_safe")
    monkeypatch.setenv("PIN_DOMAIN_SCOPE", "qwen3.6-35b-a3b=default,muse-glimmer-30b=framing_only")
    monkeypatch.setenv("PIN_CANNOT_TELL_CONSTRUCTION", "qwen3.6-35b-a3b=default,muse-glimmer-30b=coarse_vs_sharp")
    monkeypatch.setenv("PIN_POST_REFUSAL_SPAN", "qwen3.6-35b-a3b=false,muse-glimmer-30b=true")
    res_set: dict = {"machine": {"gpu": "A100"}}
    pw.pilot_window({}, res_set)
    assert res_set["side_pairing"] == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "verdict_safe"}
    assert res_set["domain_scope"] == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "framing_only"}
    assert res_set["cannot_tell_construction"] == {"qwen3.6-35b-a3b": "default", "muse-glimmer-30b": "coarse_vs_sharp"}
    assert res_set["post_refusal_span"] == {"qwen3.6-35b-a3b": False, "muse-glimmer-30b": True}
    assert res_set["arms"]["qwen3.6-35b-a3b"]["side_pairing"] == "default"
    assert res_set["arms"]["muse-glimmer-30b"]["side_pairing"] == "verdict_safe"
    assert res_set["arms"]["muse-glimmer-30b"]["domain_scope"] == "framing_only"
    assert res_set["arms"]["muse-glimmer-30b"]["cannot_tell_construction"] == "coarse_vs_sharp"
    assert res_set["arms"]["muse-glimmer-30b"]["post_refusal_span"] is True



