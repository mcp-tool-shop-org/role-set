"""Tests for the batching-drift study (DESIGN-v1.md §3, §4, §5 checks 1-8, §8)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "colab",
    REPO_ROOT / "scripts" / "batchdrift",
    REPO_ROOT / "tests",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_requests60 as br60  # noqa: E402
import compare_batch as cb  # noqa: E402
import pin_window as pw  # noqa: E402
import replay as rp  # noqa: E402
from role_set import callcache, cards, critic, gates, pack  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _critic_proxy_cfg() -> dict:
    return {
        "candidate": "muse-glimmer-30b",
        "config": {
            "candidate": "muse-glimmer-30b",
            "revision": "r1",
            "sampling": dict(rp.EXPECTED_CARD_SAMPLING),
            "chat_template_kwargs": dict(rp.EXPECTED_CARD_KWARGS),
            "transforms": ["reattach_thinking"],
        },
    }


def _critic_model_info() -> dict:
    m = next(x for x in pw.MODELS if x["served_as"] == "muse-glimmer-30b")
    return {
        "served_as": m["served_as"],
        "gguf_sha256": m["expect_sha256"],
        "path": "/dev/null",
    }


def _configure_stub_pw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, int]:
    content = tmp_path / "content"
    work = content / "pin"
    content.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    shm = tmp_path / "shm"
    shm.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "WORK", work)
    monkeypatch.setattr(pw, "JOB_LOG", work / "job.log")
    monkeypatch.setattr(pw, "TOKEN_PATH", shm / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.setattr(pw, "PILOT_PORT", port)
    monkeypatch.setattr(pw, "SMI_INTERVAL_S", 0.05)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setenv("PIN_STUB", "1")
    for k in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "PIN_DEADLINE",
        "PIN_STUB_CALL_S",
        "PIN_STUB_OUT_OF_ORDER",
        "PIN_GGUF_REPO",
        *rp.RESUME_ENV_VARS,
    ):
        monkeypatch.delenv(k, raising=False)
    return content, work, port


# ---------------------------------------------------------------------------
# Build 1: requests-60.jsonl and build_requests60.py
# ---------------------------------------------------------------------------


def test_committed_requests60_matches_pin_and_schema():
    rows, sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    assert sha == rp.REQUESTS60_SHA256
    assert len(rows) == 60
    assert [r["request_index"] for r in rows] == list(range(60))

    verdict_rows = [r for r in rows if r["check"] == "verdict"]
    twin_rows = [r for r in rows if r["check"] == "twin_verdict"]
    flaw_rows = [r for r in rows if r["check"] == "flaw"]
    assert len(verdict_rows) == 40
    assert len(twin_rows) == 12
    assert len(flaw_rows) == 8
    assert all(r["kind"] == "C" for r in flaw_rows)
    assert sum(1 for r in flaw_rows if r["split"] != "apply") == 5
    assert sum(1 for r in flaw_rows if r["split"] == "apply") == 3

    for idx, r in enumerate(rows):
        assert r["seed"] == 0 and isinstance(r["seed"], int) and not isinstance(r["seed"], bool)
        assert r["attempt"] == 0
        assert callcache.messages_sha256(r["messages"]) == r["messages_sha256"]
        assert callcache.request_key("critic", (r["messages"],), {}) == r["key"]
        assert br60.canonical_payload_sha256(r["messages"], r["seed"]) == r["payload_sha256"]
        assert r["request_index"] == idx


@pytest.mark.skipif(
    not br60.P1R_RECEIPTS_DEFAULT.is_dir(),
    reason="P1r local receipts directory not present on this machine",
)
def test_build_requests60_from_real_p1r_receipts_reproduces_committed_file(tmp_path):
    out_file = tmp_path / "requests-60.jsonl"
    rc = br60.main(["--receipts-dir", str(br60.P1R_RECEIPTS_DEFAULT), "--out", str(out_file)])
    assert rc == 0
    assert br60.file_sha256(out_file) == rp.REQUESTS60_SHA256
    assert out_file.read_bytes() == rp.REQUESTS60_DEFAULT_PATH.read_bytes()


def _build_synthetic_p1r_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create a synthetic pack, card-values, wave-drafts, and journal that exercises every branch of build_requests60.py."""

    pack_dir = tmp_path / "pack"
    trait_files = {
        "role.md": "# Lesson 1\n\nthe role\n",
        "traits/T10.md": "## 10. Calibrate confidence\n\nCalibrate.\n",
        "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
        "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
    }
    for rel, content in trait_files.items():
        p = pack_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="\n")
    shas, whole = pack.pack_hash(trait_files)
    (pack_dir / "pack.json").write_text(
        json.dumps(pack.PackManifest("verifier", "abc1234", shas, whole).as_dict()),
        encoding="utf-8",
    )

    cards_path = tmp_path / "card-values.json"
    cards_obj = {
        "candidates": [
            {
                "candidate": "muse-glimmer-30b",
                "revision": "r1",
                "sampling": dict(rp.EXPECTED_CARD_SAMPLING),
                "chat_template_kwargs": dict(rp.EXPECTED_CARD_KWARGS),
                "transforms": ["reattach_thinking"],
            }
        ]
    }
    cards_path.write_text(json.dumps(cards_obj), encoding="utf-8")
    card_set, _ = cards.load(cards_path)
    critic_card = card_set["muse-glimmer-30b"]

    drafts: list[dict] = [{"type": "header", "verdict_prompt": "old", "flaw_prompt": "old"}]
    journal_rows: list[dict] = []

    def _add_journal(slot_id: str, check_name: str, msgs: list[dict], text: str) -> None:
        sent = cards.apply(critic_card, msgs)
        journal_rows.append({
            "role": "critic",
            "slot": slot_id,
            "check": check_name,
            "attempt": 0,
            "key": callcache.request_key("critic", (msgs,), {}),
            "messages_sha256": callcache.messages_sha256(sent),
            "reused": False,
            "completion": {"text": text, "seconds": 1.25},
        })

    # 12 Kind C slots (8 train, 4 apply) + 8 Kind D apply slots with r10_twin -> 12 twin_verdict, 12 flaw
    for i in range(12):
        split = None if i < 8 else "apply"
        slot_id = f"T10-C-{i + 1:02d}" if split is None else f"T10-C-AP-{i - 7:02d}"
        mat_text = f"LineA{i} states alpha is 10.\nLineB{i} states beta is 20."
        stmt = f"Alpha is 10 in case {i}."
        twin_stmt = f"Alpha is 99 in case {i}."
        trace = f"Checker trace for case {i} says unsupported."
        slot = {
            "id": slot_id,
            "lesson": "T10",
            "kind": "C",
            "side": "guess",
            "tier": 1,
            "content_kind": "notice",
            "verdict": "supported",
            "split": split,
            "domain": {"code": "51", "title": "Info"},
        }
        cand_fields = {
            "framing": f"Check the notice for case {i}.",
            "statement": stmt,
            "deciding": [f"LineA{i} states alpha is 10."],
            "checker_verdict": "unsupported",
            "checker_trace": trace,
            "planted_flaw": "guess",
            "thinking": f"Thinking for case {i}.",
            "reply": f"The notice directly supports the claim in case {i}.",
            **({"r10_twin": {"statement": twin_stmt, "verdict": "unsupported"}} if split == "apply" else {}),
        }
        drafts.append({
            "type": "draft",
            "slot": slot,
            "attempt": 1,
            "material": {"text": mat_text, "source": {}},
            "candidate": {"text": json.dumps(cand_fields)},
            "messages": [],
        })
        v_msgs = critic.verdict_messages(mat_text, stmt, verdict_prompt="old")
        _add_journal(slot_id, "verdict", v_msgs, f"VERDICT: supported\nDECIDING: LineA{i} states alpha is 10.\nESCALATE: no")
        if split == "apply":
            tv_msgs = critic.verdict_messages(mat_text, twin_stmt, verdict_prompt="old")
            _add_journal(slot_id, "twin_verdict", tv_msgs, f"VERDICT: unsupported\nDECIDING: LineA{i} states alpha is 10.\nESCALATE: no")
        f_msgs = critic.flaw_messages(mat_text, stmt, trace, "unsupported", flaw_prompt="old")
        _add_journal(slot_id, "flaw", f_msgs, "guess")

    # 28 Kind D single-turn slots (first 8 are apply with r10_twin -> 8 more twin_verdict, total twin_verdict = 12, total verdict = 40)
    for j in range(28):
        split = "apply" if j < 8 else None
        slot_id = f"T10-D-AP-{j + 1:02d}" if split == "apply" else f"T10-D-{j - 7:02d}"
        mat_text = f"FactD{j} says gamma is 5.\nOtherD{j} says delta is 6."
        stmt = f"Gamma is 5 in D{j}."
        twin_stmt = f"Gamma is 99 in D{j}."
        slot = {
            "id": slot_id,
            "lesson": "T10",
            "kind": "D",
            "side": "direct match",
            "tier": 1,
            "content_kind": "notice",
            "verdict": "supported",
            "split": split,
            "domain": {"code": "51", "title": "Info"},
        }
        cand_fields = {
            "framing": f"Review the record for D{j}.",
            "statement": stmt,
            "deciding": [f"FactD{j} says gamma is 5."],
            "thinking": f"Thinking for D{j}.",
            "reply": f"FactD directly supports the claim in D{j}.",
            **({"r10_twin": {"statement": twin_stmt, "verdict": "unsupported"}} if split == "apply" else {}),
        }
        # Put candidate in assistant message on j==8 to cover _candidate_text_from_draft fallback
        if j == 8:
            drafts.append({
                "type": "draft",
                "slot": slot,
                "attempt": 1,
                "material": {"text": mat_text, "source": {}},
                "messages": [{"role": "assistant", "content": json.dumps(cand_fields)}],
            })
        else:
            drafts.append({
                "type": "draft",
                "slot": slot,
                "attempt": 1,
                "material": {"text": mat_text, "source": {}},
                "candidate": {"text": json.dumps(cand_fields)},
                "messages": [],
            })
        v_msgs = critic.verdict_messages(mat_text, stmt, verdict_prompt="old")
        _add_journal(slot_id, "verdict", v_msgs, f"VERDICT: supported\nDECIDING: FactD{j} says gamma is 5.\nESCALATE: no")
        if split == "apply":
            tv_msgs = critic.verdict_messages(mat_text, twin_stmt, verdict_prompt="old")
            _add_journal(slot_id, "twin_verdict", tv_msgs, f"VERDICT: unsupported\nDECIDING: FactD{j} says gamma is 5.\nESCALATE: no")

    # Sort journal_rows into train-then-apply order matching reconstruct_p1r_non_trait_calls
    train_ids = {d["slot"]["id"] for d in drafts[1:] if d["slot"].get("split") != "apply"}
    journal_rows = [r for r in journal_rows if r["slot"] in train_ids] + [r for r in journal_rows if r["slot"] not in train_ids]

    drafts_path = tmp_path / "wave-drafts.jsonl"
    drafts_path.write_text("".join(json.dumps(d) + "\n" for d in drafts), encoding="utf-8", newline="\n")
    journal_path = tmp_path / "calls.jsonl"
    journal_path.write_text("".join(json.dumps(r) + "\n" for r in journal_rows), encoding="utf-8", newline="\n")
    return drafts_path, pack_dir, journal_path, cards_path


def test_build_requests60_synthetic_end_to_end_and_refusals(tmp_path):
    drafts_path, pack_dir, journal_path, cards_path = _build_synthetic_p1r_fixture(tmp_path)
    out_path = tmp_path / "out-requests-60.jsonl"

    rc = br60.main([
        "--drafts", str(drafts_path),
        "--pack", str(pack_dir),
        "--journal", str(journal_path),
        "--cards", str(cards_path),
        "--no-pin-check",
        "--out", str(out_path),
    ])
    assert rc == 0
    rows, _ = rp.load_and_verify_requests60(out_path, expected_sha256=None, expected_count=60)
    assert len(rows) == 60

    # Also test build_from_receipts with synthetic tarballs
    rec_dir = tmp_path / "synth-receipts"
    rec_dir.mkdir()
    (rec_dir / "card-values.json").write_bytes(cards_path.read_bytes())
    with tarfile.open(rec_dir / "pack.tar", "w") as tf:
        for p in pack_dir.rglob("*"):
            if p.is_file():
                tf.add(p, arcname=str(p.relative_to(pack_dir)))
    with tarfile.open(rec_dir / "pilot-out.tar", "w") as tf:
        tf.add(drafts_path, arcname="pilot/pilot/gemma-4-31b-it/w1/wave-drafts.jsonl")
        tf.add(journal_path, arcname="pilot/pilot/gemma-4-31b-it/calls-w1-muse-glimmer-30b.jsonl")

    out_from_tar = tmp_path / "from-tar-60.jsonl"
    summary = br60.build_from_receipts(rec_dir, out_from_tar, verify_pins=False)
    assert summary["verified"] == 60
    assert out_from_tar.read_bytes() == out_path.read_bytes()

    # Refusal checks on build_requests60:
    with pytest.raises(ValueError, match="non-negative integer"):
        br60.canonical_payload_sha256([], -1)
    with pytest.raises(ValueError, match="non-negative integer"):
        br60.canonical_payload_sha256([], True)
    with pytest.raises(RuntimeError, match="neither candidate.text nor an assistant message"):
        br60._candidate_text_from_draft({"slot": {"id": "S1"}, "messages": []})
    assert br60._completion_text({"completion": "raw string"}) == "raw string"

    # Pin check refusals
    with pytest.raises(RuntimeError, match="REFUSED: wave-drafts.jsonl sha256"):
        br60.reconstruct_p1r_non_trait_calls(
            drafts_path=drafts_path,
            pack_dir=pack_dir,
            journal_path=journal_path,
            cards_path=cards_path,
            verify_pins=True,
        )

    # Partial CLI args refusal
    with pytest.raises(RuntimeError, match="provide all of --drafts, --pack, --journal, and --cards"):
        br60.main(["--drafts", str(drafts_path)])

    # Missing receipt file refusal
    with pytest.raises(RuntimeError, match="missing P1r receipt file"):
        br60.build_from_receipts(tmp_path / "empty-dir", tmp_path / "out.jsonl")

    # Select/permute and journal verification refusals
    bad_calls = [dict(r) for r in rows]
    bad_calls[0]["kind"] = "D"
    bad_calls[0]["check"] = "flaw"
    with pytest.raises(RuntimeError, match="expected all flaw calls to come from Kind C"):
        br60.select_and_permute_60(bad_calls)

    with pytest.raises(RuntimeError, match="insufficient flaw calls"):
        br60.select_and_permute_60([r for r in rows if r["check"] != "flaw"])

    with pytest.raises(RuntimeError, match="expected 60 selected calls"):
        br60.select_and_permute_60(rows[:-1])

    mut_seed = [dict(r) for r in rows]
    mut_seed[0]["seed"] = None
    with pytest.raises(RuntimeError, match="invalid seed"):
        br60.verify_against_journal(mut_seed, journal_path)

    mut_jidx = [dict(r) for r in rows]
    mut_jidx[0]["p1r_journal_index"] = 9999
    with pytest.raises(RuntimeError, match="invalid p1r_journal_index"):
        br60.verify_against_journal(mut_jidx, journal_path)

    mut_sha = [dict(r) for r in rows]
    mut_sha[0]["messages_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="recomputed messages_sha256"):
        br60.verify_against_journal(mut_sha, journal_path)


# ---------------------------------------------------------------------------
# Build 2 & 4: CPU Stub Checks 1, 2, 3, 4, 6, 8 on replay.py and pin_window.py
# ---------------------------------------------------------------------------


def test_stub_check_1_and_2_concurrency_and_props_across_all_widths(tmp_path, monkeypatch):
    """DESIGN-v1.md §5 checks 1 & 2:
    1. Concurrency reaches StubLlama (`max_in_flight == N` and `/slots` busy count `== N` at N = 1, 2, 4, 8).
    2. `/props` reports `total_slots == N` and per-slot `n_ctx == 32768`, and launch args carry
       `--cache-ram 0 --slot-prompt-similarity 0.0`.
    """
    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_8 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:8])]

    model_info = _critic_model_info()
    proxy_cfg = _critic_proxy_cfg()

    for arm_name, width in (("A1", 1), ("B2", 2), ("B4", 4), ("B8", 8)):
        res = rp.replay_arm(
            arm_name=arm_name,
            parallel=width,
            requests=subset_8,
            requests_sha256=req_sha,
            built={},
            model_info=model_info,
            proxy_cfg=proxy_cfg,
            out_dir=work / "bd",
            pw_mod=pw,
            port=port,
        )
        assert res["parallel"] == width
        assert res["ctx"] == 32768 * width
        assert res["n_ctx_per_slot"] == 32768
        assert res["extra_args"] == ["--cache-ram", "0", "--slot-prompt-similarity", "0.0"]
        assert res["props_slots"] == {"total_slots": width, "n_ctx_per_slot": 32768, "ok": True}
        assert res["stub"]["max_in_flight"] == width
        assert res["slots_busy_peak"] == width
        assert res["stub"]["slot_log_count"] == 8
        assert res["cached_prompt_tokens"] == 0
        assert res["reused_calls"] == 0


def test_start_server_command_line_and_slots_validation(tmp_path, monkeypatch):
    """Verify `pin_window.start_server` passes `--parallel N -c (32768*N) --cache-ram 0 --slot-prompt-similarity 0.0`."""
    _content, _work, port = _configure_stub_pw(tmp_path, monkeypatch)
    seen_popen: list[list[str]] = []

    class FakeProc:
        def poll(self):
            return None

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_popen(cmd, **kw):
        seen_popen.append(list(cmd))
        return FakeProc()

    monkeypatch.setattr(pw.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pw.urllib.request, "urlopen", lambda url, timeout=5: FakeResp())

    built = {"path": str(tmp_path / "llama-server")}
    m = {"served_as": "muse-glimmer-30b", "path": str(tmp_path / "model.gguf")}
    proc, log_fh, server_log = pw.start_server(
        built,
        m,
        131072,
        port,
        parallel=4,
        extra_args=list(rp.BATCHDRIFT_EXTRA_ARGS),
    )
    log_fh.close()
    assert proc is not None
    assert server_log.name == "muse-glimmer-30b-critic-server.log"
    cmd = seen_popen[0]
    assert cmd[cmd.index("-c") + 1] == "131072"
    assert cmd[cmd.index("--parallel") + 1] == "4"
    assert "--cache-ram" in cmd and cmd[cmd.index("--cache-ram") + 1] == "0"
    assert "--slot-prompt-similarity" in cmd and cmd[cmd.index("--slot-prompt-similarity") + 1] == "0.0"


def test_stub_check_3_zero_cached_tokens_and_no_journal_replay(tmp_path, monkeypatch):
    """DESIGN-v1.md §5 check 3:
    - Any resume env var is refused immediately.
    - Non-zero cached prompt tokens on any reply is refused by replay_arm and flagged void by compare_batch.
    """
    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_2 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:2])]

    for var in rp.RESUME_ENV_VARS:
        monkeypatch.setenv(var, "1")
        with pytest.raises(RuntimeError, match=f"REFUSED: batching-drift forbids journal replay.*{var}"):
            rp.require_no_resume_env()
        monkeypatch.delenv(var, raising=False)

    # Plant cached_tokens > 0 in StubLlama and verify replay_arm refuses
    orig_init = pw.Served.__enter__

    def enter_with_cached_tokens(self):
        s = orig_init(self)
        if s.stub is not None:
            s.stub.cached_tokens = 128
        return s

    monkeypatch.setattr(pw.Served, "__enter__", enter_with_cached_tokens)
    with pytest.raises(RuntimeError, match="REFUSED: arm A1 had 256 cached prompt tokens"):
        rp.replay_arm(
            arm_name="A1",
            parallel=1,
            requests=subset_2,
            requests_sha256=req_sha,
            built={},
            model_info=_critic_model_info(),
            proxy_cfg=_critic_proxy_cfg(),
            out_dir=work / "bd-cache",
            pw_mod=pw,
            port=port,
        )


def test_stub_check_4_card_sampling_and_seed_reach_server_and_dropped_seed_detected(tmp_path, monkeypatch):
    """DESIGN-v1.md §5 check 4:
    Card sampling parameters and explicit integer seed reach StubLlama in its received POST body per request_index,
    and a planted row where seed is dropped (or sampling mutated) is detected by both verify_stub_received_bodies
    and compare_batch.py.
    """
    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_4 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:4])]

    res = rp.replay_arm(
        arm_name="A1",
        parallel=1,
        requests=subset_4,
        requests_sha256=req_sha,
        built={},
        model_info=_critic_model_info(),
        proxy_cfg=_critic_proxy_cfg(),
        out_dir=work / "bd-seed",
        pw_mod=pw,
        port=port,
    )
    assert res["stub"]["received_check"]["ok"] is True
    stub_req_file = Path(res["stub"]["stub_requests_path"])
    received = [json.loads(ln) for ln in stub_req_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(received) == 4
    for idx, body in enumerate(received):
        assert body["seed"] == 0 and isinstance(body["seed"], int)
        for k, v in rp.EXPECTED_CARD_SAMPLING.items():
            assert body[k] == v
        assert body["chat_template_kwargs"] == rp.EXPECTED_CARD_KWARGS
        assert callcache.messages_sha256(body["messages"]) == subset_4[idx]["messages_sha256"]

    # Plant a dropped seed on request_index 2 in the received bodies
    dropped = [dict(b) for b in received]
    del dropped[2]["seed"]
    with pytest.raises(RuntimeError, match=r"REFUSED: seed missing or mismatched.*\[2\]"):
        rp.verify_stub_received_bodies(dropped, subset_4)

    # Write the dropped-seed POST bodies to stub_requests_path and verify compare_batch flags it void
    stub_req_file.write_text("".join(json.dumps(b) + "\n" for b in dropped), encoding="utf-8", newline="\n")
    loaded = cb.load_and_validate_arm(
        arm_name="A1",
        calls_path=Path(res["calls_path"]),
        generations_path=Path(res["generations_path"]),
        requests=subset_4,
        expected_parallel=1,
        stub_requests_path=stub_req_file,
    )
    assert loaded["valid"] is False
    assert loaded["stub_dropped_seeds"] == [2]
    assert "stub_dropped_seeds:[2]" in loaded["void_reasons"]

    # Plant a sampling mismatch on request_index 1
    bad_samp = [dict(b) for b in received]
    bad_samp[1]["temperature"] = 0.7
    with pytest.raises(RuntimeError, match="REFUSED: card sampling mismatch"):
        rp.verify_stub_received_bodies(bad_samp, subset_4)


def test_stub_check_6_join_survives_out_of_order_completion(tmp_path, monkeypatch):
    """DESIGN-v1.md §5 check 6:
    When StubLlama completes requests out of arrival order at width 4, calls and generations join accurately by
    `call_index == request_index + 1` and match A1 100% byte-for-byte.
    """
    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_8 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:8])]
    out_dir = work / "bd-ooo"

    res_a1 = rp.replay_arm(
        arm_name="A1",
        parallel=1,
        requests=subset_8,
        requests_sha256=req_sha,
        built={},
        model_info=_critic_model_info(),
        proxy_cfg=_critic_proxy_cfg(),
        out_dir=out_dir,
        pw_mod=pw,
        port=port,
        out_of_order_stub=False,
    )
    res_b4 = rp.replay_arm(
        arm_name="B4",
        parallel=4,
        requests=subset_8,
        requests_sha256=req_sha,
        built={},
        model_info=_critic_model_info(),
        proxy_cfg=_critic_proxy_cfg(),
        out_dir=out_dir,
        pw_mod=pw,
        port=port,
        out_of_order_stub=True,
    )
    assert res_b4["out_of_order_completions"] > 0

    loaded_a1 = cb.load_and_validate_arm(
        arm_name="A1",
        calls_path=Path(res_a1["calls_path"]),
        generations_path=Path(res_a1["generations_path"]),
        requests=subset_8,
        expected_parallel=1,
    )
    loaded_b4 = cb.load_and_validate_arm(
        arm_name="B4",
        calls_path=Path(res_b4["calls_path"]),
        generations_path=Path(res_b4["generations_path"]),
        requests=subset_8,
        expected_parallel=4,
    )
    assert loaded_b4["valid"] is True
    assert loaded_b4["out_of_order_calls"] > 0
    assert loaded_b4["out_of_order_generations"] > 0

    comp = cb.compare_two_arms(loaded_a1, loaded_b4, subset_8)
    assert comp["compared"] == 8
    assert comp["exact_matches"] == 8
    assert comp["group4_whole_reply_drifts"] == 0


def test_stub_in_flight_decremented_before_response_sent_under_post_response_interleaving(
    tmp_path, monkeypatch
):
    """Deterministic regression test for Run 40967817297 (c2dbcf4c):
    Force Handler 1 (serving request 0 at parallel=1) to pause immediately after flushing the HTTP response bytes
    to `wfile` until Request 1 arrives at `StubLlama.do_POST` AND increments `stub.in_flight`, using threading.Events
    (no sleeps).
    Before the fix, `stub.in_flight` was decremented in `finally:` after `_send_json(200, out)` wrote the response,
    so Request 1 incremented `stub.in_flight` to 2 (`max_in_flight=2`) while Handler 1 was still in post-response
    teardown, causing `replay_arm` to raise `RuntimeError: REFUSED: stub arm A1 max_in_flight=2, expected 1`.
    """
    import threading
    import stub_llama

    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_2 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:2])]

    resp0_written = threading.Event()
    req1_counted = threading.Event()
    orig_reuse_srv = stub_llama._ReuseHTTPServer

    class _ControlledReuseHTTPServer(orig_reuse_srv):
        def __init__(self, server_address, RequestHandlerClass):
            orig_do_post = RequestHandlerClass.do_POST
            orig_send_json = RequestHandlerClass._send_json
            post_count = 0
            post_lock = threading.Lock()
            stub_obj = next(
                c.cell_contents
                for c in (orig_do_post.__closure__ or ())
                if isinstance(c.cell_contents, stub_llama.StubLlama)
            )
            orig_requests = stub_obj.requests

            class _NotifyingList(list):
                def append(self, item):
                    super().append(item)
                    if len(self) == 2:
                        req1_counted.set()

            stub_obj.requests = _NotifyingList(orig_requests)

            def controlled_send_json(handler_self, status: int, payload: dict) -> None:
                orig_send_json(handler_self, status, payload)
                handler_self.wfile.flush()
                if getattr(handler_self, "_controlled_post_idx", None) == 0:
                    resp0_written.set()
                    req1_counted.wait(timeout=5.0)

            def controlled_do_post(handler_self):
                nonlocal post_count
                if not handler_self.path.startswith("/api/"):
                    with post_lock:
                        idx = post_count
                        post_count += 1
                    handler_self._controlled_post_idx = idx
                    if idx == 1:
                        resp0_written.wait(timeout=5.0)
                return orig_do_post(handler_self)

            RequestHandlerClass._send_json = controlled_send_json
            RequestHandlerClass.do_POST = controlled_do_post
            super().__init__(server_address, RequestHandlerClass)

    monkeypatch.setattr(stub_llama, "_ReuseHTTPServer", _ControlledReuseHTTPServer)

    res_a1 = rp.replay_arm(
        arm_name="A1",
        parallel=1,
        requests=subset_2,
        requests_sha256=req_sha,
        built={},
        model_info=_critic_model_info(),
        proxy_cfg=_critic_proxy_cfg(),
        out_dir=work / "bd-race1",
        pw_mod=pw,
        port=port,
    )
    assert resp0_written.is_set() and req1_counted.is_set()
    assert res_a1["stub"]["max_in_flight"] == 1



def test_stub_check_8_cap_watcher_cuts_cleanly_and_preserves_checkpoint_rows(tmp_path, monkeypatch, capsys):
    """DESIGN-v1.md §5 check 8:
    When PIN_DEADLINE expires mid-arm, the watcher cuts the arm cleanly, raises DeadlineReached, records pw.CUT_AT,
    and preserves the finished rows in `calls-{arm}.jsonl` and `CALL_CHECKPOINT` lines.
    """
    _content, work, port = _configure_stub_pw(tmp_path, monkeypatch)
    all_reqs, req_sha = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    subset_12 = [dict(r, request_index=i) for i, r in enumerate(all_reqs[:12])]

    said_lines: list[str] = []
    monkeypatch.setattr(pw, "say", lambda msg: said_lines.append(str(msg)))
    t_deadline = time.monotonic() + 0.45
    monkeypatch.setattr(pw, "deadline_left", lambda: max(0, int(round((t_deadline - time.monotonic()) * 10))))

    with pytest.raises(pw.DeadlineReached, match="cut at the window's deadline"):
        rp.replay_arm(
            arm_name="A1",
            parallel=1,
            requests=subset_12,
            requests_sha256=req_sha,
            built={},
            model_info=_critic_model_info(),
            proxy_cfg=_critic_proxy_cfg(),
            out_dir=work / "bd-cut",
            pw_mod=pw,
            port=port,
            stub_call_delay_s=0.10,
        )

    assert pw.CUT_AT is not None
    assert pw.CUT_AT["wave"] == "batching-drift"
    assert pw.CUT_AT["arm"] == "A1"
    assert 1 <= pw.CUT_AT["completed_calls"] < 12

    calls_file = Path(pw.CUT_AT["journal"])
    assert calls_file.is_file()
    saved_rows = [json.loads(ln) for ln in calls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(saved_rows) == pw.CUT_AT["completed_calls"]
    captured_out = capsys.readouterr().out
    assert "CALL_CHECKPOINT A1 batchdrift-A1" in captured_out
    assert any("CUT AT THE DEADLINE" in ln for ln in said_lines)


# ---------------------------------------------------------------------------
# Build 3 & 4: CPU Stub Checks 5 & 7 — full 6-arm run + compare_batch.py
# ---------------------------------------------------------------------------


def test_stub_check_5_and_7_full_6_arm_run_and_non_zero_exit_if_arm_missing(tmp_path, monkeypatch):
    """DESIGN-v1.md §5 checks 5 & 7:
    - Full 6-arm run (A1, A2, B2, B4, B8, B4r) over all 60 requests in `requests-60.jsonl` through `pin_window._main()`
      with `PIN_ROLE=batching-drift` and `PIN_STUB=1`.
    - `A1` vs `A2` is byte-identical (`exact_matches == 60`), `B4r` vs `B4` is byte-identical, and all arms classify
      as `"pinned-safe"`.
    - If any of the 6 arms did not run, `run_all_arms` raises `RuntimeError` (non-zero exit).
    """
    content, work, _port = _configure_stub_pw(tmp_path, monkeypatch)
    cards_obj = {"candidates": [_critic_proxy_cfg()["config"]]}
    (content / "card-values.json").write_text(json.dumps(cards_obj), encoding="utf-8")

    monkeypatch.setenv("PIN_ROLE", "batching-drift")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "d29499f77a81bda144ebd51885f3c167a04667aa")
    monkeypatch.setenv("PIN_RND_COMMIT", "04dccfad9419cd79c4068703ac6cf1442850c9de")
    monkeypatch.setattr(pw, "role_set_env", lambda commit: {"role_set": "role-set", "src": str(REPO_ROOT), "commit": commit})
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})

    rc = pw._main()
    assert rc == 0
    pin_res = json.loads((work / "pin-result.json").read_text(encoding="utf-8"))
    assert pin_res["ok"] is True
    assert pin_res["role"] == "batching-drift"
    bd = pin_res["batching_drift"]
    assert bd["all_six_arms_completed"] is True
    assert bd["requests_sha256"] == rp.REQUESTS60_SHA256
    assert bd["requests_count"] == 60
    assert list(bd["arms"].keys()) == ["A1", "A2", "B2", "B4", "B8", "B4r"]
    for arm_name, width in rp.REQUIRED_ARMS:
        arm_meta = bd["arms"][arm_name]
        assert arm_meta["calls_completed"] == 60
        assert arm_meta["parallel"] == width
        assert arm_meta["stub"]["max_in_flight"] == width
        assert arm_meta["cached_prompt_tokens"] == 0
        assert arm_meta["reused_calls"] == 0

    comp = bd["comparison"]
    assert comp["validity_gate_a2"]["a2_clean"] is True
    assert comp["b4_reproducibility"]["reproducible"] is True
    assert comp["arm_tiers"] == {
        "A1": "reference",
        "A2": "pinned-safe",
        "B2": "pinned-safe",
        "B4": "pinned-safe",
        "B8": "pinned-safe",
        "B4r": "pinned-safe",
    }
    assert (work / "batchdrift-out.tar").is_file()

    # CLI compare_batch.main(--dir ...) and (--arm ...) produce identical report
    cli_out = tmp_path / "cli-comp.json"
    assert cb.main(["--dir", str(work / "batchdrift"), "--out", str(cli_out)]) == 0
    cli_report = json.loads(cli_out.read_text(encoding="utf-8"))
    assert cli_report["arm_tiers"] == comp["arm_tiers"]

    arm_specs = [
        f"{a}={(work / 'batchdrift' / f'calls-{a}-muse-glimmer-30b.jsonl').as_posix()}:{(work / 'batchdrift' / f'generations-{a}-muse-glimmer-30b.jsonl').as_posix()}"
        for a, _ in rp.REQUIRED_ARMS
    ]
    cli_arm_args = ["--requests", str(rp.REQUESTS60_DEFAULT_PATH)]
    for s in arm_specs:
        cli_arm_args.extend(["--arm", s])
    assert cb.main(cli_arm_args) == 0
    with pytest.raises(RuntimeError, match="invalid --arm spec"):
        cb.main(["--requests", str(rp.REQUESTS60_DEFAULT_PATH), "--arm", "A1=/tmp/only_calls.jsonl"])

    # Check 7 refusal: if fewer than 6 arms run, run_all_arms raises RuntimeError
    with pytest.raises(RuntimeError, match="REFUSED: batching-drift did not complete all six required arms; missing or incomplete: B8, B4r"):
        rp.run_all_arms(
            built={},
            model_info=_critic_model_info(),
            proxy_cfg=_critic_proxy_cfg(),
            requests_path=rp.REQUESTS60_DEFAULT_PATH,
            out_dir=tmp_path / "incomplete-arms",
            pw_mod=pw,
            arms=(("A1", 1), ("A2", 1), ("B2", 2), ("B4", 4)),
        )


def test_compare_batch_planted_drifts_and_statistics(tmp_path):
    """DESIGN-v1.md §5 check 5:
    Verify compare_batch.py accurately classifies:
    - Group 1 decision changes (verdict flip, deciding verbatim break, twin_verdict flip, flaw flip, trait, answer)
    - Group 2 auxiliary changes (raw DECIDING quote punctuation, ESCALATE, CONTRASTED) without Group 1 drift
    - Group 3 reasoning changes (`reasoning_content` changed while `response` is identical)
    - Group 4 token/logprob changes (`first_token_diff_index`, `max_abs_logprob_diff`)
    - B4r != B4 non-reproducibility downgrading B4/B4r from `pinned-safe` to `decision-safe-only`
    - A2 != A1 noise floor shift
    - Known-bad input detection (payload mismatch, reused rows, cached tokens, duplicate/missing indices, text mismatch)
    - Exact Clopper-Pearson 95% CIs and McNemar's exact test
    """
    # 1. Clopper-Pearson 95% CI exact values (DESIGN-v1.md §4 lines 206-210)
    cp_60_60 = cb.clopper_pearson_95(60, 60)
    assert cp_60_60["rate"] == 1.0
    assert cp_60_60["two_sided_95"] == [0.940371, 1.0]
    assert cp_60_60["one_sided_95_lower"] == 0.951297

    cp_0_60 = cb.clopper_pearson_95(0, 60)
    assert cp_0_60["rate"] == 0.0
    assert cp_0_60["two_sided_95"] == [0.0, 0.059629]
    assert cp_0_60["one_sided_95_upper"] == 0.048703

    cp_1_60 = cb.clopper_pearson_95(1, 60)
    assert cp_1_60["rate"] == round(1 / 60, 6)
    assert round(cp_1_60["two_sided_95"][1], 3) == 0.089

    with pytest.raises(ValueError, match="invalid"):
        cb.clopper_pearson_95(-1, 60)

    # 2. McNemar's exact test
    mc_deg = cb.mcnemar_exact([True] * 60, [True] * 60)
    assert mc_deg["discordant"] == 0 and mc_deg["p_value"] == 1.0 and mc_deg["degenerate"] is True

    mc_disc = cb.mcnemar_exact([True] * 54 + [False] * 6, [True] * 48 + [False] * 12)
    assert mc_disc["n10"] == 6 and mc_disc["n01"] == 0 and mc_disc["discordant"] == 6
    assert mc_disc["p_value"] == 0.03125 and mc_disc["degenerate"] is False

    with pytest.raises(ValueError, match="equal length"):
        cb.mcnemar_exact([True], [True, False])

    # 3. Build synthetic arm journals from requests-60.jsonl (covering verdict with key_verdict=supported, twin_verdict, flaw)
    all_reqs, _ = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    verdict_req = next(r for r in all_reqs if r["check"] == "verdict" and r.get("key_verdict") == "supported")
    twin_req = next(r for r in all_reqs if r["check"] == "twin_verdict" and r.get("key_verdict") != "cannot_tell")
    flaw_req = next(r for r in all_reqs if r["check"] == "flaw")
    sample_reqs = [
        dict(verdict_req, request_index=0),
        dict(twin_req, request_index=1),
        dict(flaw_req, request_index=2),
    ]

    def _baseline_reply(req: dict) -> str:
        if req["check"] in ("verdict", "twin_verdict"):
            kv = req.get("key_verdict") or "supported"
            kd = list(req.get("key_deciding") or [])
            if not kd and kv != "cannot_tell":
                kd = [(req.get("material") or "").splitlines()[0]]
            if kv == "cannot_tell":
                return "VERDICT: cannot_tell\nDECIDING: none\nESCALATE: no"
            joined_kd = " | ".join(kd)
            return f"VERDICT: {kv}\nDECIDING: {joined_kd}\nESCALATE: no"
        return req.get("key_flaw") or "guess"

    def _write_arm(
        name: str,
        text_overrides: dict[int, str] | None = None,
        reasoning_overrides: dict[int, str] | None = None,
        top_overrides: dict[int, list] | None = None,
        tokens_overrides: dict[int, list[str]] | None = None,
    ) -> dict:
        c_path = tmp_path / f"calls-{name}.jsonl"
        g_path = tmp_path / f"generations-{name}.jsonl"
        c_rows = []
        g_rows = []
        for req in sample_reqs:
            idx = req["request_index"]
            txt = (text_overrides or {}).get(idx, _baseline_reply(req))
            rc_txt = (reasoning_overrides or {}).get(idx, f"reasoning-{idx}")
            toks = (tokens_overrides or {}).get(idx, ["tok0", "tok1"])
            top = (top_overrides or {}).get(idx, [[["tok0", -0.05], ["alt0", -3.0]], [["tok1", -0.02], ["alt1", -4.0]]])
            c_rows.append({
                "role": "critic",
                "arm": name,
                "key": req["key"],
                "request_index": idx,
                "slot": req["slot"],
                "check": req["check"],
                "attempt": 0,
                "seed": req["seed"],
                "messages_sha256": req["messages_sha256"],
                "payload_sha256": req["payload_sha256"],
                "reused": False,
                "completion": {"text": txt, "messages_sha256": req["messages_sha256"]},
            })
            g_rows.append({
                "kind": "result",
                "call_index": idx + 1,
                "response": txt,
                "reasoning_content": rc_txt,
                "tokens": toks,
                "top": top,
                "timings": {"cache_n": 0},
                "usage": {"prompt_tokens_details": {"cached_tokens": 0}},
            })
        c_path.write_text("".join(json.dumps(r) + "\n" for r in c_rows), encoding="utf-8", newline="\n")
        g_path.write_text("".join(json.dumps(r) + "\n" for r in g_rows), encoding="utf-8", newline="\n")
        return cb.load_and_validate_arm(
            arm_name=name,
            calls_path=c_path,
            generations_path=g_path,
            requests=sample_reqs,
        )

    arm_a1 = _write_arm("A1")
    arm_a2 = _write_arm("A2")
    # B2: Group 2 auxiliary drift only (ESCALATE: no -> ESCALATE: check_owner)
    b2_txt = _baseline_reply(sample_reqs[0]).replace("ESCALATE: no", "ESCALATE: check_owner")
    arm_b2 = _write_arm("B2", text_overrides={0: b2_txt})
    # B4: Group 3 reasoning drift only
    arm_b4 = _write_arm("B4", reasoning_overrides={1: "different chain of thought"})
    # B4r: identical to A1 (so B4r != B4 -> non-reproducible at width 4)
    arm_b4r = _write_arm("B4r")
    # B8: Group 1 decision drift (verdict flip on idx 0 and flaw flip on idx 2) + logprob diff on idx 1
    arm_b8 = _write_arm(
        "B8",
        text_overrides={
            0: "VERDICT: cannot_tell\nDECIDING: none\nESCALATE: no",
            2: "none" if _baseline_reply(sample_reqs[2]) != "none" else "guess",
        },
        top_overrides={
            1: [[["tok0", -0.25], ["alt0", -1.5]], [["tok1", -0.02], ["alt1", -4.0]]],
        },
    )

    study = cb.classify_study(
        loaded_arms={"A1": arm_a1, "A2": arm_a2, "B2": arm_b2, "B4": arm_b4, "B4r": arm_b4r, "B8": arm_b8},
        requests=sample_reqs,
    )
    assert study["validity_gate_a2"]["a2_clean"] is True
    assert study["b4_reproducibility"]["reproducible"] is False
    assert study["arm_tiers"] == {
        "A1": "reference",
        "A2": "pinned-safe",
        "B2": "decision-safe-only",
        "B4": "decision-safe-only",
        "B4r": "decision-safe-only",  # downgraded because B4r != B4
        "B8": "not-safe",
    }

    b2_comp = study["comparisons_vs_a1"]["B2"]
    assert b2_comp["group1_decision_drifts"] == 0
    assert b2_comp["group2_auxiliary_drifts"] == 1
    assert b2_comp["group2_field_changes"]["ESCALATE"] == 1

    b4_comp = study["comparisons_vs_a1"]["B4"]
    assert b4_comp["group1_decision_drifts"] == 0
    assert b4_comp["group2_auxiliary_drifts"] == 0
    assert b4_comp["group3_reasoning_drifts"] == 1
    assert b4_comp["group4_whole_reply_drifts"] == 1

    b8_comp = study["comparisons_vs_a1"]["B8"]
    assert b8_comp["group1_decision_drifts"] == 2
    assert b8_comp["verdict_drifts"] == 1
    assert b8_comp["flaw_drifts"] == 1
    assert b8_comp["max_abs_logprob_diff_identical_tokens"] == 1.5

    # Also test trait and answer Group 1 evaluators in compare_batch.py
    g = gates.load_gates()
    trait_req = {"request_index": 10, "check": "trait", "key_trait": "T10"}
    g1_t_a = cb.evaluate_group1(trait_req, "TRAIT: T10\nCONTRASTED: T02", g)
    g1_t_b = cb.evaluate_group1(trait_req, "TRAIT: T02\nCONTRASTED: T10", g)
    diff_t, reasons_t = cb.group1_differs(trait_req, g1_t_a, g1_t_b, g)
    assert diff_t is True and reasons_t == ["TRAIT"]

    ans_req = {"request_index": 11, "check": "answer", "key_answer": "42"}
    g1_a_a = cb.evaluate_group1(ans_req, "ANSWER: 42", g)
    g1_a_b = cb.evaluate_group1(ans_req, "ANSWER: 99", g)
    diff_a, reasons_a = cb.group1_differs(ans_req, g1_a_a, g1_a_b, g)
    assert diff_a is True and reasons_a == ["ANSWER"]


def test_compare_batch_detects_known_bad_inputs(tmp_path):
    """Verify load_and_validate_arm detects reused rows, seed errors, payload mismatches, text mismatches, and props mismatches."""
    all_reqs, _ = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    req0 = dict(all_reqs[0], request_index=0)

    c_path = tmp_path / "bad-calls.jsonl"
    g_path = tmp_path / "bad-gens.jsonl"
    c_path.write_text(
        json.dumps({
            "role": "critic",
            "arm": "B2",
            "key": "wrong-key",
            "request_index": 0,
            "slot": req0["slot"],
            "check": req0["check"],
            "attempt": 0,
            "seed": None,
            "messages_sha256": "0" * 64,
            "payload_sha256": "1" * 64,
            "reused": True,
            "completion": {"text": "VERDICT: supported"},
        })
        + "\n",
        encoding="utf-8",
    )
    g_path.write_text(
        json.dumps({
            "kind": "result",
            "call_index": 1,
            "response": "DIFFERENT TEXT",
            "reasoning_content": "",
            "tokens": [],
            "top": [],
            "timings": {"cache_n": 16},
        })
        + "\n",
        encoding="utf-8",
    )

    loaded = cb.load_and_validate_arm(
        arm_name="B2",
        calls_path=c_path,
        generations_path=g_path,
        requests=[req0],
        expected_parallel=2,
        props_slots={"total_slots": 1, "n_ctx_per_slot": 16384},
        extra_args=["--cache-ram", "1"],
    )
    assert loaded["valid"] is False
    assert loaded["reused_rows"] == [0]
    assert loaded["seed_errors"] == [0]
    assert loaded["payload_mismatches"] == [0]
    assert loaded["join_text_mismatches"] == [0]
    assert loaded["cached_token_rows"] == [0]
    assert any("props_total_slots" in r for r in loaded["void_reasons"])
    assert any("props_n_ctx_per_slot" in r for r in loaded["void_reasons"])
    assert any("extra_args_mismatch" in r for r in loaded["void_reasons"])


def test_replay_and_compare_edge_case_refusals(tmp_path):
    """Cover validation and refusal branches in replay.py and compare_batch.py."""
    all_reqs, _ = rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH)
    req0 = dict(all_reqs[0], request_index=0)

    # validate_request_row refusals
    with pytest.raises(RuntimeError, match="request_index=99, expected 0"):
        rp.validate_request_row({**req0, "request_index": 99}, 0)
    with pytest.raises(RuntimeError, match="missing or invalid seed"):
        rp.validate_request_row({**req0, "seed": -1}, 0)
    with pytest.raises(RuntimeError, match="empty or non-list messages"):
        rp.validate_request_row({**req0, "messages": []}, 0)
    with pytest.raises(RuntimeError, match="messages_sha256 mismatch"):
        rp.validate_request_row({**req0, "messages_sha256": "0" * 64}, 0)
    with pytest.raises(RuntimeError, match="key mismatch"):
        rp.validate_request_row({**req0, "key": "bad"}, 0)
    with pytest.raises(RuntimeError, match="payload_sha256 mismatch"):
        rp.validate_request_row({**req0, "payload_sha256": "0" * 64}, 0)

    # load_and_verify_requests60 refusals
    with pytest.raises(RuntimeError, match="requests file not found"):
        rp.load_and_verify_requests60(tmp_path / "missing.jsonl")
    with pytest.raises(RuntimeError, match="!= pinned"):
        rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="has 60 rows, expected 10"):
        rp.load_and_verify_requests60(rp.REQUESTS60_DEFAULT_PATH, expected_sha256=None, expected_count=10)

    # join_calls_and_generations refusals
    with pytest.raises(RuntimeError, match="non-result row"):
        rp.join_calls_and_generations([], [{"kind": "refused", "call_index": 1}])
    with pytest.raises(RuntimeError, match="invalid call_index"):
        rp.join_calls_and_generations([], [{"kind": "result", "call_index": 0}])
    with pytest.raises(RuntimeError, match="duplicate call_index"):
        rp.join_calls_and_generations([], [{"kind": "result", "call_index": 1}, {"kind": "result", "call_index": 1}])
    with pytest.raises(RuntimeError, match="invalid request_index"):
        rp.join_calls_and_generations([{"request_index": -1}], [{"kind": "result", "call_index": 1}])
    with pytest.raises(RuntimeError, match="duplicate request_index"):
        rp.join_calls_and_generations(
            [{"request_index": 0}, {"request_index": 0}],
            [{"kind": "result", "call_index": 1}, {"kind": "result", "call_index": 2}],
        )
    with pytest.raises(RuntimeError, match="count mismatch"):
        rp.join_calls_and_generations([{"request_index": 0}], [])
    with pytest.raises(RuntimeError, match="no matching call_index"):
        rp.join_calls_and_generations([{"request_index": 0}], [{"kind": "result", "call_index": 5}])
    with pytest.raises(RuntimeError, match="joined text mismatch"):
        rp.join_calls_and_generations(
            [{"request_index": 0, "completion": {"text": "A"}}],
            [{"kind": "result", "call_index": 1, "response": "B"}],
        )

    # compare_logprobs token difference branches
    lp_diff = cb.compare_logprobs(["a", "b"], [[["a", -0.1]], [["b", -0.2]]], ["a", "c"], [[["a", -0.1]], [["c", -0.2]]])
    assert lp_diff["tokens_equal"] is False and lp_diff["first_token_diff_index"] == 1
    lp_len = cb.compare_logprobs(["a"], [[["a", -0.1]]], ["a", "b"], [[["a", -0.1]], [["b", -0.2]]])
    assert lp_len["tokens_equal"] is False and lp_len["first_token_diff_index"] == 1

    # classify_study missing A1 and main missing args
    with pytest.raises(RuntimeError, match="reference arm A1 is missing"):
        cb.classify_study(loaded_arms={}, requests=[])
    with pytest.raises(RuntimeError, match="pass either --dir"):
        cb.main([])
    with pytest.raises(ValueError, match="unsupported check"):
        cb.evaluate_group1({"request_index": 0, "check": "unknown"}, "txt", gates.load_gates())
    with pytest.raises(ValueError, match="unsupported check"):
        cb.group1_differs({"check": "unknown"}, {}, {}, gates.load_gates())


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash") or not shutil.which("setsid"),
    reason="needs bash, flock and setsid (Linux)",
)
def test_smoke_window_accepts_batching_drift_read_token_and_refuses_write_token():
    # Window BL1 was refused at start: the token allowlist had no batching-drift entry, though the critic GGUF
    # (muse-glimmer-30b) is read from the private role-set-gguf repo.
    from test_window_script import SENTINEL, _window

    rig_ok, job_ok, _ = _window(
        "normal",
        extra={
            "PIN_ROLE": "batching-drift",
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_READ",
            "HF_TOKEN_GGUF_READ": SENTINEL,
            "PIN_MOCK_S": "1",
        },
    )
    assert "uploaded the token file" in rig_ok and job_ok.count("PIN_RESULT") == 1, rig_ok

    rig_bad, job_bad, out_bad = _window(
        "normal",
        extra={
            "PIN_ROLE": "batching-drift",
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_WRITE",
            "HF_TOKEN_GGUF_WRITE": SENTINEL,
        },
    )
    assert "PIN_ROLE=batching-drift may not send HF_TOKEN_GGUF_WRITE" in rig_bad and not job_bad, rig_bad
    assert not (out_bad.parent / "st" / "news").exists()


def test_batchdrift_import_checks_role_set_out_before_importing_replay(tmp_path, monkeypatch):
    # Window BL1 failed on a fresh VM with `No module named 'replay'`: pin_window imported scripts/batchdrift before
    # role_set_env had cloned role-set. Here neither the VM checkout nor the repo root has scripts/batchdrift until
    # role_set_env runs, so the import only succeeds if the clone comes first.
    content = tmp_path / "content"
    content.mkdir()
    empty_root = tmp_path / "no-repo"
    empty_root.mkdir()
    src_dir = Path(__file__).resolve().parents[1] / "scripts" / "batchdrift"
    import aspire_si_gates  # the test venv has it; the VM's pin_window interpreter does not

    gates_pkg = Path(aspire_si_gates.__file__).resolve().parent
    calls = []

    def fake_role_set_env(commit):
        calls.append(commit)
        dst = content / "role-set" / "scripts" / "batchdrift"
        shutil.copytree(src_dir, dst)
        site = content / "rs-venv" / "lib" / "python3.15" / "site-packages"
        shutil.copytree(gates_pkg, site / gates_pkg.name)  # the venv role_set_env built holds the gates package
        return {"commit": commit}

    monkeypatch.setattr(pw, "CONTENT", content)
    monkeypatch.setattr(pw, "_repo_root", lambda: empty_root)
    monkeypatch.setattr(pw, "role_set_env", fake_role_set_env)
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "a" * 40)
    monkeypatch.delitem(sys.modules, "replay", raising=False)
    for entry in [x for x in sys.path if x.endswith(("batchdrift", "batchdrift/", "batchdrift\\"))]:
        monkeypatch.setattr(sys, "path", [x for x in sys.path if x != entry])
    # As on the VM: aspire_si_gates is nowhere on this interpreter's path until the venv's site-packages is added.
    for name in [m for m in sys.modules if m == "aspire_si_gates" or m.startswith("aspire_si_gates.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", [x for x in sys.path if "site-packages" not in x])
    monkeypatch.setattr(pw, "_TOKEN", "sentinel-token")
    seen_token = []
    real_fake = fake_role_set_env

    def watching_role_set_env(commit):
        seen_token.append(pw._TOKEN)
        return real_fake(commit)

    monkeypatch.setattr(pw, "role_set_env", watching_role_set_env)
    result = {}
    mod = pw.import_batchdrift_replay(result)
    assert calls == ["a" * 40]
    assert result["role_set"] == {"commit": "a" * 40}
    assert Path(mod.__file__).resolve().parent == (content / "role-set" / "scripts" / "batchdrift").resolve()
    assert seen_token == [None], "role_set_env ran with the token in memory"
    assert pw._TOKEN == "sentinel-token", "the token was not handed back after the clone"
    from role_set import gates

    gates.load_gates()  # resolves aspire_si_gates from the venv's site-packages, as the window's comparison needs
