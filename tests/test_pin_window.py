"""The Colab window script's own rules that run without a VM."""

import os
import subprocess
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
    "sample": "GPU sampling during a serve; no wave runs inside it",
    "smoke": "the server's startup poll; no wave runs inside it",
    "start_server": "the server's startup poll; no wave runs inside it",
    "critic_window": "re-raises DeadlineReached first",
    "q6k_window": "re-raises DeadlineReached first",
    "pilot_window": "the tar in finally, after the waves",
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
