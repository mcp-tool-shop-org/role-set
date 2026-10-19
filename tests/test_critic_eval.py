"""Tests for `role-set critic-eval`, `role-set critic-rescore-offline`, and `PIN_ROLE=critic-eval`."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest
from stub_llama import StubLlama
from test_waves import pack_dir  # noqa: F401 - fixture

from role_set import waves
from role_set.cli import main
from role_set.critic_eval import (
    CriticEvalConfig,
    CriticEvalError,
    CriticRescoreOfflineConfig,
    run_critic_eval,
    run_critic_rescore_offline,
)
from role_set.mock import MockCritic
from role_set.transport import Completion
from role_set.write import WriteConfig, run

ROOT = Path(__file__).resolve().parents[1]
CARDS = Path(__file__).parent / "fixtures" / "card-values.json"

sys.path.insert(0, str(ROOT / "scripts" / "colab"))
sys.path.insert(0, str(ROOT / "scripts" / "critic_signal"))
import checkpoints  # noqa: E402
import perturb  # noqa: E402
import pin_window as pw  # noqa: E402

LIGHTNING = "nemotron-3.5-lightning-30b-a3b"
MUSE = "muse-glimmer-30b"


@pytest.fixture
def eval_items_file(tmp_path: Path, domain_table: Path, pack_dir: Path) -> Path:  # noqa: F811
    """Build a sealed items file containing kind C and kind D drafts with per-item keys and perturbation types."""
    plan = tmp_path / "plan.json"
    assert main(["plan", "--pilot", "--seed", "0", "--domains", str(domain_table), "--out", str(plan)]) == 0
    data = json.loads(plan.read_text(encoding="utf-8"))
    kept = []
    counts = {"C": 0, "D": 0}
    for s in data["slots"]:
        k = s["kind"]
        if k in counts and counts[k] < 2:
            kept.append(s)
            counts[k] += 1
    data["slots"] = kept
    plan.write_text(json.dumps(data, indent=1), encoding="utf-8")

    w0_out = tmp_path / "w0"
    run(WriteConfig(plan=plan, pack=pack_dir, out=w0_out, mock=True, partition="trial:j1-wave1", wave="write"))
    h0, d0 = waves.read(w0_out / "mock" / waves.DRAFTS_FILE)
    answers = waves.load_answers(h0)

    items = []
    ptypes = ["d", "b", "e", "f"]
    for idx, d in enumerate(d0):
        slot = d["slot"]
        cand = json.loads(d["candidate"]["text"])
        stmt = cand["statement"]
        first_line = d["material"]["text"].splitlines()[0]
        ans = answers.get((stmt, first_line))
        if isinstance(ans, tuple) and len(ans) == 2:
            verdict, deciding = ans[0], list(ans[1])
        else:
            verdict = slot.get("verdict") or "supported"
            deciding = list(cand.get("deciding") or [first_line])
        planted_flaw = cand.get("planted_flaw") or ("guess" if slot["kind"] == "C" else "none")
        item = {
            **d,
            "type": "draft",
            "item_id": f"{slot['id']}:{ptypes[idx % len(ptypes)]}",
            "source_row_id": slot["id"],
            "perturbation_type": ptypes[idx % len(ptypes)],
            "writer_model": "qwen3.8-27b",
            "writer_family": "qwen",
            "window_tag": "release-window13-tag",
            "key": {
                "verdict": verdict,
                "deciding": deciding,
                "planted_flaw": planted_flaw,
            },
        }
        items.append(item)

    items_path = tmp_path / "sealed-items.jsonl"
    header = {
        "type": "header",
        "format": waves.FORMAT,
        "kind": "critic-eval-items",
        "mock_answers": h0.get("mock_answers", []),
        "writer_pins": {"writer_model": "qwen3.8-27b", "writer_family": "qwen"},
        "writer_host": {"machine": {"colab_release_tag": "release-window13-tag"}},
    }
    waves.write(items_path, header, items)
    return items_path


def test_critic_eval_end_to_end_mock_records_checks_and_makes_no_trait_calls(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    h, items = waves.read(eval_items_file)
    base_critic = MockCritic(answers=waves.load_answers(h))
    calls_made: list[list[dict]] = []

    def counting_critic(messages: list[dict], **kwargs):
        calls_made.append(messages)
        return base_critic(messages, **kwargs)

    import role_set.critic_eval as ce_mod

    monkeypatch.setattr(ce_mod, "MockCritic", lambda answers: counting_critic)

    out_dir = tmp_path / "eval-out"
    rc = main([
        "critic-eval",
        "--items", str(eval_items_file),
        "--pack", str(pack_dir),
        "--writers", str(ROOT / "writers.json"),
        "--critic-model", LIGHTNING,
        "--critic-seed", "42",
        "--out", str(out_dir),
        "--mock",
    ])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (out_dir / "critic-eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = json.loads((out_dir / "critic-eval-report.json").read_text(encoding="utf-8"))

    assert len(rows) == len(items)
    expected_calls = sum(2 if it["slot"]["kind"] == "C" else 1 for it in items)
    # Exactly verdict (1) + flaw for kind C (1); zero trait calls.
    assert len(calls_made) == expected_calls
    assert report["calls"]["called"] == expected_calls
    assert report["input_sha256"] == waves.file_sha256(eval_items_file)
    assert report["gates"]["version"] and len(report["gates"]["sha256"]) == 64

    for row, src in zip(rows, items, strict=True):
        assert row["item_id"] == src["item_id"]
        assert row["perturbation_type"] == src["perturbation_type"]
        assert row["seed"] == 42
        verdict_chk = row["verdict_check"]
        assert set(verdict_chk) >= {"code", "detail", "raw_reply", "thinking_tokens", "output_tokens"}
        assert row["critic_verdict"] in ("supported", "unsupported", "cannot_tell")
        assert isinstance(row["critic_deciding"], list)
        if src["slot"]["kind"] == "C":
            assert row["flaw_check"] is not None
            assert set(row["flaw_check"]) >= {"code", "detail", "raw_reply", "thinking_tokens", "output_tokens"}
        else:
            assert row["flaw_check"] is None


def test_critic_eval_skips_same_family_writer_and_reports_it(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path  # noqa: F811
):
    h, items = waves.read(eval_items_file)
    # Stamp first item as nvidia family (same as nemotron-3.5-lightning-30b-a3b).
    items[0]["writer_model"] = LIGHTNING
    items[0]["writer_family"] = "nvidia"
    mixed_path = tmp_path / "mixed-items.jsonl"
    waves.write(mixed_path, h, items)

    out_dir = tmp_path / "eval-family"
    report = run_critic_eval(
        CriticEvalConfig(
            items=mixed_path,
            pack=pack_dir,
            out=out_dir,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=0,
            mock=True,
        )
    )
    assert report["skipped_count"] == 1
    assert report["skipped"][0]["item_id"] == items[0]["item_id"]
    assert report["items_scored"] == len(items) - 1


def test_critic_eval_refuses_unblinded_identifier_in_critic_messages(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path  # noqa: F811
):
    h, items = waves.read(eval_items_file)
    cand = json.loads(items[0]["candidate"]["text"])
    cand["statement"] = f"{cand['statement']} (written by qwen3.8-27b in release-window13-tag)"
    items[0]["candidate"]["text"] = json.dumps(cand)
    leaked_path = tmp_path / "leaked-items.jsonl"
    waves.write(leaked_path, h, items)

    with pytest.raises(CriticEvalError, match="REFUSED: unblinded identifier"):
        run_critic_eval(
            CriticEvalConfig(
                items=leaked_path,
                pack=pack_dir,
                out=tmp_path / "eval-leak",
                writers_file=ROOT / "writers.json",
                critic_model=LIGHTNING,
                critic_seed=0,
                mock=True,
            )
        )


def test_critic_eval_void_reply_retries_once_then_records(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    h, items = waves.read(eval_items_file)
    # Keep 1 kind-D item so we can test 1 VOID retry succeeding and 2 VOIDs staying VOID.
    d_item = next(it for it in items if it["slot"]["kind"] == "D")
    one_path = tmp_path / "one-item.jsonl"
    waves.write(one_path, h, [d_item])

    base_critic = MockCritic(answers=waves.load_answers(h))
    attempt_counter = {"n": 0}

    def void_once_critic(messages: list[dict], **kwargs):
        attempt_counter["n"] += 1
        if attempt_counter["n"] == 1:
            return Completion(
                text="not valid verdict lines at all", prompt_tokens=10, completion_tokens=5, model="mock", seconds=0.01
            )
        return base_critic(messages, **kwargs)

    import role_set.critic_eval as ce_mod

    monkeypatch.setattr(ce_mod, "MockCritic", lambda answers: void_once_critic)
    out1 = tmp_path / "void-once"
    run_critic_eval(
        CriticEvalConfig(
            items=one_path,
            pack=pack_dir,
            out=out1,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=5,
            mock=True,
        )
    )
    assert attempt_counter["n"] == 2
    row1 = json.loads((out1 / "critic-eval.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row1["verdict_check"]["code"] == "AGREE"
    assert row1["verdict_check"]["void_retries"] == 1
    assert row1["verdict_check"]["attempts"] == 2

    # Now always return VOID: retries once (2 calls total), then records VOID.
    attempt_counter["n"] = 0

    def always_void_critic(messages: list[dict], **kwargs):
        attempt_counter["n"] += 1
        return Completion(
            text="still not valid verdict lines", prompt_tokens=10, completion_tokens=5, model="mock", seconds=0.01
        )

    monkeypatch.setattr(ce_mod, "MockCritic", lambda answers: always_void_critic)
    out2 = tmp_path / "void-always"
    run_critic_eval(
        CriticEvalConfig(
            items=one_path,
            pack=pack_dir,
            out=out2,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=5,
            mock=True,
        )
    )
    assert attempt_counter["n"] == 2
    row2 = json.loads((out2 / "critic-eval.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row2["verdict_check"]["code"] == "VOID"
    assert row2["verdict_check"]["void_retries"] == 2
    assert row2["verdict_check"]["attempts"] == 2
    assert row2["critic_verdict"] is None


def test_critic_eval_journal_resume_replays_finished_calls_with_zero_model_calls(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    h, _ = waves.read(eval_items_file)
    base_critic = MockCritic(answers=waves.load_answers(h))
    live_calls = {"n": 0}

    def tracking_critic(messages: list[dict], **kwargs):
        live_calls["n"] += 1
        return base_critic(messages, **kwargs)

    import role_set.critic_eval as ce_mod

    monkeypatch.setattr(ce_mod, "MockCritic", lambda answers: tracking_critic)

    journal_1 = tmp_path / "run1-journal.jsonl"
    out_1 = tmp_path / "out1"
    run_critic_eval(
        CriticEvalConfig(
            items=eval_items_file,
            pack=pack_dir,
            out=out_1,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=11,
            call_journal=journal_1,
            mock=True,
        )
    )
    first_run_calls = live_calls["n"]
    assert first_run_calls > 0
    assert journal_1.exists()

    # Second run reusing journal_1: if any live critic call is made, fail immediately.
    def forbidden_critic(messages: list[dict], **kwargs):
        raise AssertionError("live critic should not be called when all calls are in --reuse-calls journal")

    monkeypatch.setattr(ce_mod, "MockCritic", lambda answers: forbidden_critic)
    journal_2 = tmp_path / "run2-journal.jsonl"
    out_2 = tmp_path / "out2"
    rep2 = run_critic_eval(
        CriticEvalConfig(
            items=eval_items_file,
            pack=pack_dir,
            out=out_2,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=11,
            call_journal=journal_2,
            reuse_calls=journal_1,
            mock=True,
        )
    )
    assert rep2["calls"]["reused"] == first_run_calls
    assert rep2["calls"]["called"] == 0
    assert (out_1 / "critic-eval.jsonl").read_text(encoding="utf-8") == (
        out_2 / "critic-eval.jsonl"
    ).read_text(encoding="utf-8")


def test_critic_eval_real_endpoint_passes_card_and_seed_and_records_tokens(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    import role_set.critic_eval as ce_mod

    monkeypatch.setattr(ce_mod, "load_gates", lambda closed_form=None: ce_mod.mock_gates(closed_form=closed_form))

    h, items = waves.read(eval_items_file)
    d_item = next(it for it in items if it["slot"]["kind"] == "D")
    one_path = tmp_path / "one-real.jsonl"
    waves.write(one_path, h, [d_item])

    writers_data = json.loads((ROOT / "writers.json").read_text(encoding="utf-8"))
    lightning_sha = next(c["gguf_sha256"] for c in writers_data["critics"] if c["served_as"] == LIGHTNING)

    srv = StubLlama("qwen3.8-27b", LIGHTNING, critic=MockCritic(answers=waves.load_answers(h)))
    try:
        out_dir = tmp_path / "real-out"
        rc = main([
            "critic-eval",
            "--items", str(one_path),
            "--pack", str(pack_dir),
            "--writers", str(ROOT / "writers.json"),
            "--cards", str(CARDS),
            "--critic-endpoint", srv.url,
            "--critic-model", LIGHTNING,
            "--critic-gguf-sha256", lightning_sha,
            "--critic-seed", "9",
            "--out", str(out_dir),
        ])
        assert rc == 0
        assert len(srv.requests) == 1
        req = srv.requests[0]
        assert req["seed"] == 9
        assert req["temperature"] == 1.0
    finally:
        srv.close()

    rows = [json.loads(x) for x in (out_dir / "critic-eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["verdict_check"]["code"] == "AGREE"
    assert rows[0]["output_tokens"] > 0
    report = json.loads((out_dir / "critic-eval-report.json").read_text(encoding="utf-8"))
    assert report["critic_card"]["card_candidate"] == LIGHTNING
    assert report["critic_seed"] == 9


def test_critic_rescore_offline_p0_p2_and_p3_rules(tmp_path: Path):
    """P0 vs P2 on hand-built replies covers each tightened rule:
    (i) flaw mismatch;
    (ii) cannot_tell with non-empty quote;
    (iii) quote longer than len(key_deciding) + 1;
    and P3 refuses if either seed refuses."""
    items = [
        {
            "type": "draft",
            "item_id": "item-flaw:c",
            "perturbation_type": "c",
            "slot": {"id": "S1", "kind": "C", "planted_flaw": "guess"},
            "material": {"text": "Line 1.\nLine 2.", "source": "s1"},
            "candidate": {"text": json.dumps({"statement": "Stmt 1.", "checker_trace": "Trace 1.", "checker_verdict": "supported"})},
            "key": {"verdict": "unsupported", "deciding": ["Line 1."], "planted_flaw": "guess"},
        },
        {
            "type": "draft",
            "item_id": "item-ct:e",
            "perturbation_type": "e",
            "slot": {"id": "S2", "kind": "D"},
            "material": {"text": "Line 1.\nLine 2.", "source": "s2"},
            "candidate": {"text": json.dumps({"statement": "Stmt 2."})},
            "key": {"verdict": "cannot_tell", "deciding": [], "planted_flaw": "none"},
        },
        {
            "type": "draft",
            "item_id": "item-long:g",
            "perturbation_type": "g",
            "slot": {"id": "S3", "kind": "D"},
            "material": {"text": "Line A.\nLine B.\nLine C.", "source": "s3"},
            "candidate": {"text": json.dumps({"statement": "Stmt 3."})},
            "key": {"verdict": "supported", "deciding": ["Line A."], "planted_flaw": "none"},
        },
        {
            "type": "draft",
            "item_id": "item-ctrl:d",
            "perturbation_type": "d",
            "slot": {"id": "S4", "kind": "D"},
            "material": {"text": "Line X.\nLine Y.", "source": "s4"},
            "candidate": {"text": json.dumps({"statement": "Stmt 4."})},
            "key": {"verdict": "supported", "deciding": ["Line X."], "planted_flaw": "none"},
        },
    ]
    items_path = tmp_path / "hand-items.jsonl"
    waves.write(items_path, {"type": "header", "format": waves.FORMAT, "kind": "test"}, items)

    # Seed 0 replies:
    # 1. item-flaw:c -> verdict matches, flaw reply names 'nitpick' (mismatch: P0 accepts, P2 refuses)
    # 2. item-ct:e -> verdict matches cannot_tell, deciding has ["Line 1."] (P0 accepts, P2 refuses)
    # 3. item-long:g -> verdict matches supported, deciding has 3 lines vs key's 1 line (P0 accepts, P2 refuses)
    # 4. item-ctrl:d -> verdict matches supported, deciding has ["Line X."] (P0 and P2 accept)
    eval_s0 = [
        {
            "item_id": "item-flaw:c",
            "perturbation_type": "c",
            "kind": "C",
            "seed": 0,
            "critic_verdict": "unsupported",
            "critic_deciding": ["Line 1."],
            "verdict_check": {
                "code": "AGREE",
                "detail": "agree",
                "raw_reply": "VERDICT: unsupported\nDECIDING: Line 1.\nESCALATE: no",
            },
            "flaw_check": {
                "code": "FLAW_MISMATCH",
                "detail": "wrong flaw",
                "raw_reply": "nitpick",
            },
        },
        {
            "item_id": "item-ct:e",
            "perturbation_type": "e",
            "kind": "D",
            "seed": 0,
            "critic_verdict": "cannot_tell",
            "critic_deciding": ["Line 1."],
            "verdict_check": {
                "code": "AGREE",
                "detail": "agree",
                "raw_reply": "VERDICT: cannot_tell\nDECIDING: Line 1.\nESCALATE: no",
            },
            "flaw_check": None,
        },
        {
            "item_id": "item-long:g",
            "perturbation_type": "g",
            "kind": "D",
            "seed": 0,
            "critic_verdict": "supported",
            "critic_deciding": ["Line A.", "Line B.", "Line C."],
            "verdict_check": {
                "code": "AGREE",
                "detail": "agree",
                "raw_reply": "VERDICT: supported\nDECIDING: Line A. | Line B. | Line C.\nESCALATE: no",
            },
            "flaw_check": None,
        },
        {
            "item_id": "item-ctrl:d",
            "perturbation_type": "d",
            "kind": "D",
            "seed": 0,
            "critic_verdict": "supported",
            "critic_deciding": ["Line X."],
            "verdict_check": {
                "code": "AGREE",
                "detail": "agree",
                "raw_reply": "VERDICT: supported\nDECIDING: Line X.\nESCALATE: no",
            },
            "flaw_check": None,
        },
    ]
    eval_s0_path = tmp_path / "eval-s0.jsonl"
    eval_s0_path.write_text("\n".join(json.dumps(r) for r in eval_s0) + "\n", encoding="utf-8")

    # Seed 1 replies: item-ctrl:d is refused on seed 1 so P3 refuses it while seed 0's P2 accepted it.
    eval_s1 = [
        {**r, "seed": 1} for r in eval_s0[:3]
    ] + [
        {
            **eval_s0[3],
            "seed": 1,
            "critic_verdict": "unsupported",
            "critic_deciding": ["Line X."],
            "verdict_check": {
                "code": "VERDICT_MISMATCH",
                "detail": "mismatch",
                "raw_reply": "VERDICT: unsupported\nDECIDING: Line X.\nESCALATE: no",
            },
            "flaw_check": None,
        },
    ]
    eval_s1_path = tmp_path / "eval-s1.jsonl"
    eval_s1_path.write_text("\n".join(json.dumps(r) for r in eval_s1) + "\n", encoding="utf-8")

    out_dir = tmp_path / "offline-out"
    rc = main([
        "critic-rescore-offline",
        "--eval", str(eval_s0_path), str(eval_s1_path),
        "--items", str(items_path),
        "--out", str(out_dir),
        "--mock",
    ])
    assert rc == 0

    rows = [
        json.loads(line)
        for line in (out_dir / "critic-rescore-offline.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_rule_item = {(r["rule_set"], r["item_id"]): r for r in rows}

    # Rule (i): flaw mismatch -> P0 accepts (advisory), P2 refuses with FLAW_MISMATCH
    assert by_rule_item[("P0", "item-flaw:c")]["outcome"] == "accept"
    assert by_rule_item[("P2", "item-flaw:c")]["outcome"] == "refuse"
    assert by_rule_item[("P2", "item-flaw:c")]["deciding_rule"] == "FLAW_MISMATCH"

    # Rule (ii): cannot_tell with non-empty deciding -> P0 accepts, P2 refuses with CANNOT_TELL_DECIDING
    assert by_rule_item[("P0", "item-ct:e")]["outcome"] == "accept"
    assert by_rule_item[("P2", "item-ct:e")]["outcome"] == "refuse"
    assert by_rule_item[("P2", "item-ct:e")]["deciding_rule"] == "CANNOT_TELL_DECIDING"

    # Rule (iii): quote length > len(key_deciding) + 1 -> P0 accepts, P2 refuses with QUOTE_TOO_LONG
    assert by_rule_item[("P0", "item-long:g")]["outcome"] == "accept"
    assert by_rule_item[("P2", "item-long:g")]["outcome"] == "refuse"
    assert by_rule_item[("P2", "item-long:g")]["deciding_rule"] == "QUOTE_TOO_LONG"

    # Control item: P2 accepts on seed 0, but seed 1 refuses -> P3 refuses!
    assert by_rule_item[("P2", "item-ctrl:d")]["outcome"] == "accept"
    assert by_rule_item[("P3", "item-ctrl:d")]["outcome"] == "refuse"
    assert by_rule_item[("P3", "item-ctrl:d")]["deciding_rule"] == "VERDICT_MISMATCH"


def test_critic_rescore_offline_scores_alternate_keys_a_c_g_without_any_model_call(
    eval_items_file: Path, pack_dir: Path, tmp_path: Path, monkeypatch  # noqa: F811
):
    # Generate perturbed items and alternate keys (a), (c), (g) from eval_items_file
    pert_dir = tmp_path / "perturbed"
    perturb.run_perturb(eval_items_file, pert_dir, seed=17)

    # Score items.jsonl once with MockCritic
    eval_out = tmp_path / "eval-out"
    run_critic_eval(
        CriticEvalConfig(
            items=pert_dir / "items.jsonl",
            pack=pack_dir,
            out=eval_out,
            writers_file=ROOT / "writers.json",
            critic_model=LIGHTNING,
            critic_seed=0,
            mock=True,
        )
    )

    # Now forbid any critic construction during offline rescoring of keys-a, keys-c, keys-g.
    import role_set.critic_eval as ce_mod
    import role_set.write as w_mod

    def no_critic(*args, **kwargs):
        raise AssertionError("No critic or model call may be constructed in critic-rescore-offline")

    monkeypatch.setattr(ce_mod, "MockCritic", no_critic)
    monkeypatch.setattr(ce_mod, "critic_caller", no_critic)
    monkeypatch.setattr(w_mod, "critic_caller", no_critic)

    for key_type, expected_rule in (("a", "VERDICT_MISMATCH"), ("c", "FLAW_MISMATCH"), ("g", "EVIDENCE_MISMATCH")):
        key_file = pert_dir / f"keys-{key_type}.jsonl"
        off_out = tmp_path / f"offline-{key_type}"
        report = run_critic_rescore_offline(
            CriticRescoreOfflineConfig(
                eval_files=(eval_out / "critic-eval.jsonl",),
                items=pert_dir / "items.jsonl",
                keys=(key_file,),
                out=off_out,
                mock=True,
            )
        )
        assert report["summary"]["P2"][key_type]["refused"] >= 1
        rows = [
            json.loads(x)
            for x in (off_out / "critic-rescore-offline.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        p2_rows = [r for r in rows if r["rule_set"] == "P2" and r["perturbation_type"] == key_type]
        assert p2_rows
        assert all(r["outcome"] == "refuse" and r["deciding_rule"] == expected_rule for r in p2_rows)


def test_critic_eval_and_offline_error_cases_and_extract_item_parts(tmp_path: Path):
    from role_set.critic_eval import extract_item_parts, score_single_item
    from role_set.mock import mock_gates

    # extract_item_parts on an assembled row with turns and quiz (kind E) and multi-turn new_material
    assembled = {
        "row": {
            "id": "E-ROW-1",
            "kind": "E",
            "material": "Line 1.",
            "turns": [
                {"user": "Check:\nStatement: The claim.\nChecker verdict: supported\nMy trace here."},
                {"new_material": ["Line 2."]},
            ],
            "quiz": {"verdict": "supported", "deciding": ["Line 1."]},
        },
        "fields": {"planted_flaw": "guess"},
    }
    parts = extract_item_parts(assembled)
    assert parts["item_id"] == "E-ROW-1"
    assert parts["statement"] == "The claim."
    assert parts["checker_verdict"] == "supported"
    assert parts["checker_trace"] == "My trace here."
    assert parts["key_verdict"] == "supported"
    assert parts["key_deciding"] == ["Line 1."]
    assert parts["key_flaw"] == "guess"
    assert "Line 2." in parts["seen_material"]

    # Unknown rule set in score_single_item raises CriticEvalError
    with pytest.raises(CriticEvalError, match="unknown rule set"):
        score_single_item(mock_gates(), {}, parts, "P99")

    # CLI error exit code 2 when input is missing
    assert main(["critic-eval", "--items", str(tmp_path / "missing.jsonl"), "--mock"]) == 2
    assert main(["critic-rescore-offline", "--eval", str(tmp_path / "missing.jsonl"), "--items", str(tmp_path / "missing.jsonl")]) == 2

    # P3 refuses if same seed is given twice, or >2 eval files, or no items/keys
    with pytest.raises(CriticEvalError, match="needs at least one --eval"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(), items=tmp_path / "x.jsonl", mock=True))
    with pytest.raises(CriticEvalError, match="takes one --eval file"):
        run_critic_rescore_offline(
            CriticRescoreOfflineConfig(
                eval_files=(tmp_path / "1.jsonl", tmp_path / "2.jsonl", tmp_path / "3.jsonl"),
                items=tmp_path / "x.jsonl",
                mock=True,
            )
        )
    with pytest.raises(CriticEvalError, match="needs --items"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(tmp_path / "1.jsonl",), mock=True))

    # P3 with both seeds accepting and both seeds refusing, plus .jsonl output path
    k_file = tmp_path / "plain-keys.jsonl"
    k_file.write_text(
        "\n".join([
            json.dumps({"item_id": "ok:d", "perturbation_type": "d", "kind": "D", "material": "Line 1.", "statement": "S1", "key": {"verdict": "supported", "deciding": ["Line 1."]}}),
            json.dumps({"item_id": "bad:b", "perturbation_type": "b", "kind": "D", "material": "Line 1.", "statement": "S2", "key": {"verdict": "supported", "deciding": ["Line 1."]}}),
        ]) + "\n",
        encoding="utf-8",
    )
    e0 = tmp_path / "e0.jsonl"
    e1 = tmp_path / "e1.jsonl"
    e0.write_text(
        "\n".join([
            json.dumps({"item_id": "ok:d", "seed": 0, "critic_verdict": "supported", "critic_deciding": ["Line 1."]}),
            json.dumps({"item_id": "bad:b", "seed": 0, "critic_verdict": "unsupported", "critic_deciding": ["Line 1."]}),
        ]) + "\n",
        encoding="utf-8",
    )
    e1.write_text(
        "\n".join([
            json.dumps({"item_id": "ok:d", "seed": 1, "critic_verdict": "supported", "critic_deciding": ["Line 1."]}),
            json.dumps({"item_id": "bad:b", "seed": 1, "critic_verdict": "cannot_tell", "critic_deciding": ["Line 1."]}),
        ]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CriticEvalError, match="different seeds"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(e0, e0), keys=(k_file,), mock=True))

    out_jsonl = tmp_path / "direct-out.jsonl"
    rep = run_critic_rescore_offline(
        CriticRescoreOfflineConfig(eval_files=(e0, e1), keys=(k_file,), out=out_jsonl, mock=True)
    )
    assert out_jsonl.exists()
    assert rep["by_item"]["ok:d"]["P3"]["outcome"] == "accept"
    assert rep["by_item"]["bad:b"]["P3"]["outcome"] == "refuse"

    # Cover seal mismatch, empty/invalid JSONL, missing cards/writers, and unsupported item kinds/keys
    bad_seal_file = tmp_path / "bad-seal.jsonl"
    bad_seal_file.write_text('{"item_id": "1"}\n', encoding="utf-8")
    waves.seal_path(bad_seal_file).write_text("0" * 64 + "  bad-seal.jsonl\n", encoding="utf-8")
    with pytest.raises(CriticEvalError, match="doesn't match its seal"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(e0,), keys=(bad_seal_file,), mock=True))

    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(CriticEvalError, match="is empty"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(e0,), keys=(empty_file,), mock=True))
    with pytest.raises(CriticEvalError, match="holds no critic-eval records"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(empty_file,), keys=(k_file,), mock=True))

    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(CriticEvalError, match="cannot parse"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(e0,), keys=(bad_json,), mock=True))
    with pytest.raises(CriticEvalError, match="cannot parse"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(bad_json,), keys=(k_file,), mock=True))

    unmatched_key = tmp_path / "unmatched.jsonl"
    unmatched_key.write_text(json.dumps({"item_id": "no-such-id", "source_row_id": "no-such-src", "material": "M", "statement": "S", "key": {"verdict": "supported", "deciding": ["M"]}}) + "\n", encoding="utf-8")
    with pytest.raises(CriticEvalError, match="no recorded critic reply found"):
        run_critic_rescore_offline(CriticRescoreOfflineConfig(eval_files=(e0,), keys=(unmatched_key,), mock=True))

    # Validation refusals in run_critic_eval (missing writers, missing cards, bad kind, missing statement/key)
    valid_sealed = tmp_path / "valid-sealed.jsonl"
    waves.write(
        valid_sealed,
        {"type": "header", "format": waves.FORMAT, "waves": [{"pins": {"machine": {"colab_release_tag": "window99-tag"}}}]},
        [{"type": "draft", "item_id": "i1", "kind": "A", "material": "M", "statement": "S", "key": {"verdict": "supported", "deciding": ["M"]}}],
    )
    with pytest.raises(CriticEvalError, match="cannot read writers file"):
        run_critic_eval(CriticEvalConfig(items=valid_sealed, out=tmp_path / "ce-out", writers_file=tmp_path / "no-writers.json", mock=False))
    with pytest.raises(CriticEvalError, match="pass --cards"):
        run_critic_eval(CriticEvalConfig(items=valid_sealed, out=tmp_path / "ce-out", writers_file=ROOT / "writers.json", critic_model=LIGHTNING, mock=False))
    with pytest.raises(CriticEvalError, match="unsupported kind 'A'"):
        run_critic_eval(CriticEvalConfig(items=valid_sealed, out=tmp_path / "ce-out", writers_file=ROOT / "writers.json", cards=CARDS, critic_model=LIGHTNING, mock=True))

    # Cover extract_item_parts fallback branches (row verdict/deciding/planted_flaw, deciding_lines, flaw)
    p_row = extract_item_parts({"row": {"id": "R2", "kind": "C", "verdict": "unsupported", "deciding": ["D1"], "planted_flaw": "loop"}})
    assert (p_row["key_verdict"], p_row["key_deciding"], p_row["key_flaw"]) == ("unsupported", ["D1"], "loop")
    p_alt = extract_item_parts({"slot": {"id": "S2", "verdict": "cannot_tell"}, "key": {"deciding_lines": ["D2"], "flaw": "fix"}})
    assert (p_alt["key_verdict"], p_alt["key_deciding"], p_alt["key_flaw"]) == ("cannot_tell", ["D2"], "fix")
    p_fld = extract_item_parts({"fields": {"deciding": ["D3"]}})
    assert p_fld["key_verdict"] == "" and p_fld["key_deciding"] == ["D3"]
    p_bad_cand = extract_item_parts({"candidate": {"text": "{bad json"}})
    assert p_bad_cand["key_deciding"] == []

    # Cover --items + --keys material inheritance (both same item_id override and control_item_id inheritance)
    base_items = tmp_path / "base-items.jsonl"
    base_items.write_text(
        "\n".join([
            json.dumps({"item_id": "ok:d", "source_row_id": "ok", "perturbation_type": "d", "kind": "D", "material": "Line 1.", "statement": "S1", "key": {"verdict": "supported", "deciding": ["Line 1."]}}),
            json.dumps({"item_id": "bad:b", "source_row_id": "bad", "perturbation_type": "b", "kind": "D", "material": "Line 1.", "statement": "S2", "key": {"verdict": "supported", "deciding": ["Line 1."]}}),
        ]) + "\n",
        encoding="utf-8",
    )
    override_keys = tmp_path / "override-keys.jsonl"
    override_keys.write_text(
        "\n".join([
            json.dumps({"item_id": "ok:d", "source_row_id": "ok", "perturbation_type": "d", "kind": "D", "material": "", "statement": "S1", "key": {"verdict": "supported", "deciding": ["Line 1."]}}),
            json.dumps({"item_id": "ok:a", "source_row_id": "ok", "control_item_id": "ok:d", "perturbation_type": "a", "kind": "D", "material": "", "statement": "S1", "key": {"verdict": "unsupported", "deciding": ["Line 1."]}}),
        ]) + "\n",
        encoding="utf-8",
    )
    rep_inh = run_critic_rescore_offline(
        CriticRescoreOfflineConfig(eval_files=(e0, e1), items=base_items, keys=(override_keys,), out=tmp_path / "inh-out", mock=True)
    )
    assert rep_inh["by_item"]["ok:a"]["P2"]["outcome"] == "refuse"


# ---- Window role: PIN_ROLE=critic-eval (Linux / POSIX tests, matching test_rescore_window.py) ----

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="tar, chmod and a POSIX shell, as on the VM")


def _tar_bytes(path: Path, files: dict[str, bytes], *, gz: bool = False) -> dict[str, str]:
    mode = "w:gz" if gz else "w"
    with tarfile.open(path, mode) as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


@pytest.fixture
def ce_vm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(pw, "CONTENT", tmp_path)
    monkeypatch.setattr(pw, "WORK", tmp_path / "pin")
    monkeypatch.setattr(pw, "say", lambda *a: None)
    monkeypatch.setattr(pw, "CUT_AT", None)
    monkeypatch.delenv("PIN_DEADLINE", raising=False)
    monkeypatch.delenv("PIN_GGUF_REPO", raising=False)
    monkeypatch.delenv("PIN_GGUF_REVISION", raising=False)
    monkeypatch.delenv("PIN_CRITIC_EVAL_RESUME", raising=False)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(pw, "TOKEN_PATH", tmp_path / "shm-hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    (tmp_path / "pin").mkdir()
    (tmp_path / "pack").mkdir()
    (tmp_path / "pack" / "pack.json").write_text("{}", encoding="utf-8")
    (tmp_path / "card-values.json").write_text("{}", encoding="utf-8")
    items_bytes = b'{"type":"header"}\n'
    shas = _tar_bytes(tmp_path / "critic-eval-inputs.tar", {"items.jsonl": items_bytes})
    monkeypatch.setenv("PIN_CRITIC_EVAL_INPUTS", str(tmp_path / "critic-eval-inputs.tar"))
    monkeypatch.setenv("PIN_CRITIC_EVAL_SHAS", ",".join(f"{k}={v}" for k, v in shas.items()))
    monkeypatch.setenv("PIN_CRITIC_EVAL_CRITIC", LIGHTNING)
    monkeypatch.setenv("PIN_CRITIC_SEEDS", "0,1")
    monkeypatch.setenv("PIN_ROLESET_COMMIT", "c" * 40)
    monkeypatch.setenv("PIN_RND_COMMIT", "d" * 40)
    monkeypatch.setattr(pw, "role_set_env", lambda commit: {"role_set": "role-set", "src": str(tmp_path)})
    monkeypatch.setattr(pw, "install_rnd", lambda commit: {"commit": commit})
    monkeypatch.setattr(
        pw,
        "sh",
        lambda cmd, **k: subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout,
    )
    return tmp_path


def _fake_ce_pool(monkeypatch, events: list):
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


def _fake_ce_eval(monkeypatch, events: list, *, card={"temperature": 0.6}, exit_code=0):
    def run_proc(cmd, **kw):
        events.append(("eval", cmd))
        out = Path(cmd[cmd.index("--out") + 1])
        journal = Path(cmd[cmd.index("--call-journal") + 1])
        seed = int(cmd[cmd.index("--critic-seed") + 1])
        journal.write_text(
            json.dumps({"key": f"k-{seed}", "model": LIGHTNING, "seed": seed, "completion": {"text": "{}"}}) + "\n",
            encoding="utf-8",
        )
        (out / "critic-eval.jsonl").write_text(json.dumps({"item_id": "i1", "seed": seed}) + "\n", encoding="utf-8")
        (out / "critic-eval-report.json").write_text(
            json.dumps({"critic_card": card, "items_scored": 1, "skipped_count": 0, "calls": 1}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, exit_code, "critic-eval ok", "")

    monkeypatch.setattr(pw, "run_proc", run_proc)


@posix_only
def test_window_wiring_order_prefetch_serve_eval_per_seed_stop_and_token_clean(ce_vm: Path, monkeypatch):
    events: list = []
    result: dict = {}
    token_val = "hf_" + "READtokenForCriticEvalWindow9876543210abc"
    pw.TOKEN_PATH.write_text(token_val, encoding="utf-8")
    pw.read_token_file(needed=True)
    monkeypatch.setenv("PIN_CRITIC_EVAL_CRITIC", MUSE)
    monkeypatch.setenv("PIN_GGUF_REPO", "org/private-ggufs")
    monkeypatch.setenv("PIN_GGUF_REVISION", "a" * 40)

    def fake_prefetch(names, by_name, res):
        tok = pw.take_token()
        assert tok == token_val
        events.append(("prefetch", list(names)))
        pw.check_token_clean(res)

    monkeypatch.setattr(pw, "prefetch_hosted", fake_prefetch)
    _fake_ce_pool(monkeypatch, events)
    _fake_ce_eval(monkeypatch, events)

    pw.critic_eval_window({}, result)

    assert [e if isinstance(e, str) else e[0] for e in events] == [
        "prefetch",
        "get",
        "serve",
        "eval",
        "eval",
        "stop",
    ]
    assert all(result["token_clean"].values()), result["token_clean"]
    assert not pw.TOKEN_PATH.exists()
    assert pw._TOKEN is None
    assert result["critic_card"] == {"temperature": 0.6}
    assert [r["seed"] for r in result["critic_eval"]["runs"]] == [0, 1]
    names = tarfile.open(ce_vm / "pin" / "critic-eval-out.tar").getnames()
    assert "critic-eval/s0/critic-eval.jsonl" in names
    assert "critic-eval/s1/critic-eval.jsonl" in names


@posix_only
def test_window_byte_match_refusal_before_serve(ce_vm: Path, monkeypatch):
    monkeypatch.delenv("PIN_STUB", raising=False)
    served, ran = [], []
    monkeypatch.setattr(
        pw,
        "fetch_model",
        lambda spec, built, keep_snapshot=False: {
            "served_as": spec["served_as"],
            "gguf_sha256": "f" * 64,
            "path": "/dev/null",
        },
    )
    monkeypatch.setattr(pw, "Served", lambda *a, **k: served.append(a))
    monkeypatch.setattr(pw, "run_proc", lambda *a, **k: ran.append(a))
    with pytest.raises(RuntimeError, match="doesn't reproduce the pin"):
        pw.critic_eval_window({}, {})
    assert not served and not ran


@posix_only
def test_window_input_sha_mismatch_refuses_before_setup(ce_vm: Path, monkeypatch):
    events: list = []
    _fake_ce_pool(monkeypatch, events)
    _fake_ce_eval(monkeypatch, events)
    monkeypatch.setenv("PIN_CRITIC_EVAL_SHAS", "items.jsonl=" + "0" * 64)
    with pytest.raises(RuntimeError, match="REFUSED: critic-eval input sha256 mismatch"):
        pw.critic_eval_window({}, {})
    assert not events


@posix_only
def test_window_accepts_tar_gz_input(ce_vm: Path, monkeypatch):
    events: list = []
    result: dict = {}
    _fake_ce_pool(monkeypatch, events)
    _fake_ce_eval(monkeypatch, events)

    gz_path = ce_vm / "critic-eval-inputs.tar.gz"
    shas = _tar_bytes(gz_path, {"items.jsonl": b'{"type":"header"}\n'}, gz=True)
    monkeypatch.setenv("PIN_CRITIC_EVAL_INPUTS", str(gz_path))
    monkeypatch.setenv("PIN_CRITIC_EVAL_SHAS", f"items.jsonl={shas['items.jsonl']}")
    monkeypatch.setenv("PIN_CRITIC_SEEDS", "0")

    pw.critic_eval_window({}, result)
    assert result["critic_eval_inputs"]["files"]["items.jsonl"] == shas["items.jsonl"]
    assert (ce_vm / "pin" / "critic-eval-in" / "items.jsonl.sha256").exists()


@posix_only
def test_window_deadline_cuts_run_streams_journal_and_returns_tar(ce_vm: Path, monkeypatch, capsys):
    events: list = []
    _fake_ce_pool(monkeypatch, events)
    monkeypatch.setenv("PIN_DEADLINE", str(int(time.time() + 30)))
    monkeypatch.setenv("PIN_CRITIC_SEEDS", "0,1")

    journal_line = json.dumps({"key": "k-cut", "model": LIGHTNING, "seed": 0, "completion": {"text": "{}"}})

    def run_proc(cmd, **kw):
        journal = Path(cmd[cmd.index("--call-journal") + 1])
        journal.write_text(journal_line + "\n", encoding="utf-8")
        time.sleep(0.05)  # allow stream_calls thread to flush
        assert kw["term_first"] is True and kw["timeout"] <= 30
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(pw, "run_proc", run_proc)
    with pytest.raises(pw.DeadlineReached, match="critic-eval seed 0 was cut"):
        pw.critic_eval_window({}, {})

    assert pw.CUT_AT["wave"] == "critic-eval" and pw.CUT_AT["seed"] == 0
    names = tarfile.open(ce_vm / "pin" / "critic-eval-out.tar").getnames()
    assert "critic-eval/s0/calls-journal.jsonl" in names

    # Verify CALL_CHECKPOINT was streamed and checkpoints.extract recovers the journal!
    stdout = capsys.readouterr().out
    assert "CALL_CHECKPOINT " in stdout
    job_log = ce_vm / "job.log"
    job_log.write_text(stdout, encoding="utf-8")
    extracted_dir = ce_vm / "extracted"
    checkpoints.extract(job_log, extracted_dir)
    recovered = extracted_dir / "critic-eval-s0" / "calls" / f"{LIGHTNING}.jsonl"
    assert recovered.exists()
    assert recovered.read_text(encoding="utf-8").strip() == journal_line


@posix_only
def test_window_resumes_from_pin_critic_eval_resume_tar(ce_vm: Path, monkeypatch):
    events: list = []
    result: dict = {}
    _fake_ce_pool(monkeypatch, events)
    _fake_ce_eval(monkeypatch, events)

    resume_tar = ce_vm / "earlier-journals.tar"
    _tar_bytes(
        resume_tar,
        {
            f"critic-eval-s0/calls/{LIGHTNING}.jsonl": b'{"key":"k0"}\n',
            f"critic-eval-s1/calls/{LIGHTNING}.jsonl": b'{"key":"k1"}\n',
        },
    )
    monkeypatch.setenv("PIN_CRITIC_EVAL_RESUME", str(resume_tar))
    monkeypatch.setenv("PIN_CRITIC_SEEDS", "0,1")

    pw.critic_eval_window({}, result)
    eval_cmds = [e[1] for e in events if isinstance(e, tuple) and e[0] == "eval"]
    assert len(eval_cmds) == 2
    for seed, cmd in enumerate(eval_cmds):
        assert "--reuse-calls" in cmd
        reuse_arg = cmd[cmd.index("--reuse-calls") + 1]
        assert f"critic-eval-s{seed}" in reuse_arg


@pytest.mark.skipif(
    os.name == "nt" or not shutil.which("bash") or not shutil.which("setsid"),
    reason="needs bash, flock and setsid (Linux)",
)
def test_smoke_window_accepts_critic_eval_read_token_and_refuses_write_token():
    from test_window_script import SENTINEL, _window

    # critic-eval + HF_TOKEN_GGUF_READ is accepted
    rig_ok, job_ok, _ = _window(
        "normal",
        extra={
            "PIN_ROLE": "critic-eval",
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_READ",
            "HF_TOKEN_GGUF_READ": SENTINEL,
            "PIN_MOCK_S": "1",
        },
    )
    assert "uploaded the token file" in rig_ok and job_ok.count("PIN_RESULT") == 1, rig_ok

    # critic-eval + HF_TOKEN_GGUF_WRITE is refused before any VM is opened
    rig_bad, job_bad, out_bad = _window(
        "normal",
        extra={
            "PIN_ROLE": "critic-eval",
            "PIN_HF_TOKEN_FROM": "HF_TOKEN_GGUF_WRITE",
            "HF_TOKEN_GGUF_WRITE": SENTINEL,
        },
    )
    assert "PIN_ROLE=critic-eval may not send HF_TOKEN_GGUF_WRITE" in rig_bad and not job_bad, rig_bad
    assert not (out_bad.parent / "st" / "news").exists()
