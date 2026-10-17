"""The Colab window script's own rules that run without a VM."""

import os
import sys
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


class _Reclaimed(BaseException):
    """Colab taking the VM back: nothing in the window catches it."""


def _fake_window(monkeypatch, tmp_path, *, die_after_write_of=None):
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
        return {"exit": 0, "result": {"dropped": 0, "states": {"awaiting_critique": 0, "needs_rewrite": 0}}}

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


@pytest.mark.skipif(os.name == "nt", reason="fcntl.flock is Linux, where the window runs")
def test_the_job_lock_lets_one_job_run_per_vm(tmp_path):
    first = pw.take_job_lock(tmp_path)
    assert first is not None
    assert pw.take_job_lock(tmp_path) is None  # a second exec, even in the same process, finds it held
    assert "JOB ALREADY RUNNING" in (tmp_path / "job.log").read_text(encoding="utf-8")
    first.close()
    assert pw.take_job_lock(tmp_path) is not None  # released with its file
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
    with pytest.raises(RuntimeError, match="no deepseek-r1-distill-qwen-32b-q6k"):
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

    slots_pilot = "T02-A-01,T02-C-01,T02-D-01,T02-E-01,T02-B-01,T02-A-02,T02-C-08,T02-D-05"
    monkeypatch.setenv("PIN_STUB", "1")
    monkeypatch.setenv("PIN_ROLE", "pilot")
    monkeypatch.setenv("PIN_CONTENT", str(content))
    monkeypatch.setenv("PIN_WORK", str(work))
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "d12bae86a22a82567f98e7f2697f1fddb9b1f300")
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
    assert res["ok"] is True and res["stub"] is True
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
