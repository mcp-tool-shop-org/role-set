"""Tests for `role-set rescore` (R&D specification, 2026-10-17 / 2026-10-18)."""

from __future__ import annotations

import json
import tarfile
from functools import partial
from pathlib import Path

import pytest
from stub_llama import StubLlama
from test_waves import pack_dir  # noqa: F401 - fixture

from role_set import waves
from role_set.cli import main
from role_set.mock import MockCritic
from role_set.rescore import VERBATIM_NOTE, RescoreConfig, RescoreError, run_rescore
from role_set.write import WriteConfig, run

ROOT = Path(__file__).resolve().parents[1]
CARDS = Path(__file__).parent / "fixtures" / "card-values.json"
OLD_GRANITE_W0_HEADER = Path(__file__).parent / "fixtures" / "window12b-granite-w0-header.json"


@pytest.fixture
def screen_plan(tmp_path: Path, domain_table: Path) -> Path:
    plan = tmp_path / "pilot-plan.json"
    assert main(["plan", "--pilot", "--seed", "0", "--domains", str(domain_table), "--out", str(plan)]) == 0
    data = json.loads(plan.read_text(encoding="utf-8"))
    # Keep 8 slots across kinds (A, B, C, D, E) with at least 4 verdict-bearing C/D slots so 2 can be refused by Muse.
    kept = []
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    caps = {"A": 1, "B": 1, "C": 3, "D": 2, "E": 1}
    for s in data["slots"]:
        k = s["kind"]
        if counts[k] < caps[k]:
            kept.append(s)
            counts[k] += 1
    data["slots"] = kept
    plan.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return plan


_PHONETIC = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "november", "oscar", "papa", "quebec",
    "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
)
_STMT_SEQ = 0


def _make_arm_waves(
    plan: Path,
    pack: Path,
    arm_root: Path,
    *,
    writer_model: str = "qwen3.8-27b",
    writer_family: str = "qwen",
    window_tag: str = "release-window12f-tag",
    refuse_count: int = 2,
    drop_at_w0_slot: str | None = None,
    monkeypatch=None,
) -> tuple[Path, Path, set[str]]:
    """Run mock write (w0) and critique (w1) waves for one arm, with Muse refusing `refuse_count` C/D drafts."""

    global _STMT_SEQ
    import role_set.write as w_mod

    w0_out = arm_root / "w0"
    w1_out = arm_root / "w1"
    run(WriteConfig(plan=plan, pack=pack, out=w0_out, mock=True, partition="trial:j1-wave1", wave="write"))
    w0_drafts_path = w0_out / "mock" / waves.DRAFTS_FILE
    h0, d0 = waves.read(w0_drafts_path)

    # Stamp the arm's writer pins and window tag as a window's write wave records them.
    h0["writer_pins"] = {
        "writer_model": writer_model,
        "writer_family": writer_family,
        "card_candidate": writer_model,
        "writer_repo": f"org/{writer_model}",
        "writer_gguf_file": f"{writer_model}-Q8_0.gguf",
    }
    h0["writer_host"] = {"machine": {"colab_release_tag": window_tag}}
    answers = waves.load_answers(h0)
    for d in d0:
        if d.get("candidate"):
            d["candidate"]["model"] = writer_model
            if d["slot"]["kind"] in ("C", "D") and d["state"] == "awaiting_critique":
                _STMT_SEQ += 1
                w1 = _PHONETIC[_STMT_SEQ % len(_PHONETIC)]
                w2 = _PHONETIC[(_STMT_SEQ // len(_PHONETIC)) % len(_PHONETIC)]
                fields = json.loads(d["candidate"]["text"])
                old_stmt = fields["statement"]
                new_stmt = f"At the {w1} {w2} counter, {old_stmt[0].lower() + old_stmt[1:]}"
                first_line = (d.get("material") or {}).get("text", "").splitlines()[0]
                if (old_stmt, first_line) in answers:
                    answers[(new_stmt, first_line)] = answers[(old_stmt, first_line)]
                if ("turn2", old_stmt, first_line) in answers:
                    answers[("turn2", new_stmt, first_line)] = answers[("turn2", old_stmt, first_line)]
                fields["statement"] = new_stmt
                if d["slot"]["kind"] == "C" and "checker_trace" in fields:
                    old_trace = fields["checker_trace"]
                    new_trace = old_trace.replace(old_stmt, new_stmt)
                    if ("flaw", old_trace) in answers:
                        answers[("flaw", new_trace)] = answers[("flaw", old_trace)]
                    fields["checker_trace"] = new_trace
                d["candidate"]["text"] = json.dumps(fields)
    h0["mock_answers"] = waves.dump_answers(answers)

    if drop_at_w0_slot:
        for i, d in enumerate(d0):
            if d["slot"]["id"] == drop_at_w0_slot:
                d0[i] = {
                    **d,
                    "state": "dropped",
                    "attempt": 3,
                    "candidate": None,
                    "refusals": [
                        {"code": "GATE", "detail": "gate refusal 1", "suggestion": "fix 1"},
                        {"code": "GATE", "detail": "gate refusal 2", "suggestion": "fix 2"},
                        {"code": "ASSEMBLY", "detail": "gate refusal 3", "suggestion": "fix 3"},
                    ],
                }
    waves.write(w0_drafts_path, h0, d0)

    # Choose C/D statements for Muse to refuse in w1.
    refuse_stmts: list[str] = []
    for d in d0:
        if d["state"] == "awaiting_critique" and d["slot"]["kind"] in ("C", "D"):
            stmt = json.loads(d["candidate"]["text"])["statement"]
            if stmt not in refuse_stmts:
                refuse_stmts.append(stmt)
            if len(refuse_stmts) >= refuse_count:
                break
    refuse_set = set(refuse_stmts[:refuse_count])

    if monkeypatch is not None:
        monkeypatch.setattr(w_mod, "MockCritic", partial(MockCritic, disagree=set(refuse_set)))
    run(WriteConfig(plan=plan, pack=pack, out=w1_out, mock=True, partition="trial:j1-wave1", wave="critique", drafts=w0_drafts_path))
    if monkeypatch is not None:
        monkeypatch.setattr(w_mod, "MockCritic", MockCritic)
    w1_drafts_path = w1_out / "mock" / waves.DRAFTS_FILE
    return w0_drafts_path, w1_drafts_path, refuse_set


def _pack_tar(tar_path: Path, pilot_root: Path) -> Path:
    with tarfile.open(tar_path, "w") as tf:
        tf.add(pilot_root, arcname="pilot")
    return tar_path


def test_end_to_end_rescore_on_mock_waves_for_two_arms(screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch, capsys):  # noqa: F811
    pilot_dir = tmp_path / "window12f" / "pilot" / "pilot"
    _, _, ref_qwen = _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "qwen3.8-27b",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    _, _, ref_gemma = _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "gemma-4-31b-it",
        writer_model="gemma-4-31b-it", writer_family="gemma", refuse_count=2, drop_at_w0_slot="T02-E-01", monkeypatch=monkeypatch,
    )
    tar_path = _pack_tar(tmp_path / "window12f" / "pilot-out.tar", tmp_path / "window12f" / "pilot")

    # Second critic is MockCritic that also refuses the same C/D statements Muse refused.
    import role_set.rescore as r_mod

    monkeypatch.setattr(r_mod, "MockCritic", lambda answers: MockCritic(answers=answers, disagree=ref_qwen | ref_gemma))

    out_dir = tmp_path / "rescore-out"
    rc = main([
        "rescore",
        "--runs", str(tar_path),
        "--pack", str(pack_dir),
        "--writers", str(ROOT / "writers.json"),
        "--critic-model", "nemotron-3.5-lightning-30b-a3b",
        "--seed", "7",
        "--out", str(out_dir),
        "--mock",
    ])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert VERBATIM_NOTE in stdout
    assert "qwen3.8-27b" in stdout and "gemma-4-31b-it" in stdout

    manifest = json.loads((out_dir / "sample-manifest.json").read_text(encoding="utf-8"))
    rescore = json.loads((out_dir / "rescore.json").read_text(encoding="utf-8"))
    report = json.loads((out_dir / "rescore-report.json").read_text(encoding="utf-8"))

    assert manifest["seed"] == 7
    assert len(manifest["items"]) == 8
    # Every manifest item records arm, slot, w0_state, refusal_code, and which check refused it (gate or critic).
    for item in manifest["items"]:
        assert set(item) >= {"arm", "slot", "w0_state", "refusal_code", "refused_by", "check", "group"}
        if item["group"] == "accepted":
            assert item["w0_state"] == "accepted" and item["refusal_code"] is None and item["check"] is None
        else:
            assert item["w0_state"] in ("needs_rewrite", "dropped")
            assert item["refusal_code"] == "VERDICT_MISMATCH"
            assert item["refused_by"] == "critic" and item["check"] == "critic"

    # The slot dropped by a mechanical gate in w0 (T02-E-01 on gemma) is recorded as gate-refused and never drawn.
    gemma_slots = {s["slot"]: s for s in manifest["arms"]["gemma-4-31b-it"]["slots"]}
    assert gemma_slots["T02-E-01"]["w0_state"] == "dropped"
    assert gemma_slots["T02-E-01"]["refusal_code"] == "ASSEMBLY"
    assert gemma_slots["T02-E-01"]["check"] == "gate"
    assert all(i["slot"] != "T02-E-01" for i in manifest["items"] if i["arm"] == "gemma-4-31b-it")

    assert len(rescore["items"]) == 8
    assert report["note"] == VERBATIM_NOTE
    for arm in ("qwen3.8-27b", "gemma-4-31b-it"):
        arm_rep = report["arms"][arm]
        assert arm_rep["n"] == 4
        assert arm_rep["verdict_agreements"] == 4
        assert arm_rep["group_split"] == "accepted 2/2, refused 2/2"
        assert arm_rep["flagged"] is False
        # C items both refused report flaw agreement.
        assert arm_rep["c_both_refused"] >= 1
        assert arm_rep["flaw_agreements"] == arm_rep["c_both_refused"]


def test_blinding_strips_arm_writer_served_as_and_window_from_critic_messages(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    pilot_dir = tmp_path / "window12f" / "pilot" / "pilot"
    _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "arm-alpha",
        writer_model="qwen3.8-27b", writer_family="qwen", window_tag="window12f-secret-tag", refuse_count=2, monkeypatch=monkeypatch,
    )
    _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "arm-beta",
        writer_model="granite-4.2-30b", writer_family="granite", window_tag="window12g-secret-tag", refuse_count=2, monkeypatch=monkeypatch,
    )
    tar_path = _pack_tar(tmp_path / "window12f" / "pilot-out.tar", tmp_path / "window12f" / "pilot")

    captured_messages: list[list[dict]] = []
    answers: dict = {}
    for arm_name in ("arm-alpha", "arm-beta"):
        h0, _ = waves.read(pilot_dir / arm_name / "w0" / "mock" / waves.DRAFTS_FILE)
        answers.update(waves.load_answers(h0))
    base_critic = MockCritic(answers=answers)

    def capturing_critic(messages: list[dict], **kwargs):
        captured_messages.append(messages)
        return base_critic(messages, **kwargs)

    cfg = RescoreConfig(
        runs=(tar_path,),
        pack=pack_dir,
        writers_file=ROOT / "writers.json",
        critic_model="nemotron-3.5-lightning-30b-a3b",
        seed=11,
        out=tmp_path / "blind-out",
        mock=True,
    )
    run_rescore(cfg, critic=capturing_critic)
    assert captured_messages, "expected the second critic to be called"
    blob = json.dumps(captured_messages)
    forbidden = (
        "arm-alpha",
        "arm-beta",
        "qwen3.8-27b",
        "granite-4.2-30b",
        "Qwen/Qwen3.8-27B",
        "ibm-granite/granite-4.2-30b",
        "window12f",
        "window12g",
        "window12f-secret-tag",
        "window12g-secret-tag",
    )
    for token in forbidden:
        assert token not in blob, f"leaked {token!r} into critic messages"


def test_fixed_seed_reproduces_sample_and_queue_order_and_another_seed_differs(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    pilot_dir = tmp_path / "pilot" / "pilot"
    _make_arm_waves(screen_plan, pack_dir, pilot_dir / "qwen3.8-27b", writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=3, monkeypatch=monkeypatch)
    _make_arm_waves(screen_plan, pack_dir, pilot_dir / "gemma-4-31b-it", writer_model="gemma-4-31b-it", writer_family="gemma", refuse_count=3, monkeypatch=monkeypatch)
    tar_path = _pack_tar(tmp_path / "pilot-out.tar", tmp_path / "pilot")

    for name, seed in (("run-a", 42), ("run-b", 42), ("run-c", 99)):
        run_rescore(RescoreConfig(
            runs=(tar_path,), pack=pack_dir, writers_file=ROOT / "writers.json",
            critic_model="nemotron-3.5-lightning-30b-a3b", seed=seed, out=tmp_path / name, mock=True,
        ))

    man_a = json.loads((tmp_path / "run-a" / "sample-manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((tmp_path / "run-b" / "sample-manifest.json").read_text(encoding="utf-8"))
    man_c = json.loads((tmp_path / "run-c" / "sample-manifest.json").read_text(encoding="utf-8"))

    assert man_a["queue"] == man_b["queue"]
    assert man_a["items"] == man_b["items"]
    assert man_a["queue"] != man_c["queue"]
    # Within-arm sample also differs across seeds when pool > 2.
    sample_a = {(x["arm"], x["slot"]) for x in man_a["items"]}
    sample_c = {(x["arm"], x["slot"]) for x in man_c["items"]}
    assert sample_a != sample_c or [x["slot"] for x in man_a["queue"]] != [x["slot"] for x in man_c["queue"]]


def test_family_skip_reports_same_family_arm_as_unsampled(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    pilot_dir = tmp_path / "pilot" / "pilot"
    # Bolt is nemotron, same family as the second critic Lightning (nemotron).
    _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "nemotron-3.5-bolt-74b-a9b",
        writer_model="nemotron-3.5-bolt-74b-a9b", writer_family="nemotron", refuse_count=2, monkeypatch=monkeypatch,
    )
    _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "qwen3.8-27b",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    tar_path = _pack_tar(tmp_path / "pilot-out.tar", tmp_path / "pilot")

    out_dir = tmp_path / "out"
    report = run_rescore(RescoreConfig(
        runs=(tar_path,),
        pack=pack_dir,
        writers_file=ROOT / "writers.json",
        critic_model="nemotron-3.5-lightning-30b-a3b",
        seed=3,
        out=out_dir,
        mock=True,
    ))
    manifest = json.loads((out_dir / "sample-manifest.json").read_text(encoding="utf-8"))

    assert "nemotron-3.5-bolt-74b-a9b" in report["unsampled"]
    bolt_rep = report["arms"]["nemotron-3.5-bolt-74b-a9b"]
    assert bolt_rep["unsampled"] is True and bolt_rep["sampled"] is False and bolt_rep["n"] == 0
    assert "nemotron" in bolt_rep["skip_reason"].lower()
    assert manifest["arms"]["nemotron-3.5-bolt-74b-a9b"]["unsampled"] is True
    assert all(i["arm"] != "nemotron-3.5-bolt-74b-a9b" for i in manifest["items"])
    assert report["arms"]["qwen3.8-27b"]["sampled"] is True and report["arms"]["qwen3.8-27b"]["n"] == 4
    assert "unsampled" in report["table"]


def test_flag_threshold_one_disagreement_is_not_flagged_and_two_are(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    pilot_dir = tmp_path / "pilot" / "pilot"
    w0_a, _, ref_a = _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "qwen3.8-27b",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    w0_b, _, ref_b = _make_arm_waves(
        screen_plan, pack_dir, pilot_dir / "gemma-4-31b-it",
        writer_model="gemma-4-31b-it", writer_family="gemma", refuse_count=2, monkeypatch=monkeypatch,
    )
    tar_path = _pack_tar(tmp_path / "pilot-out.tar", tmp_path / "pilot")

    # Arm A (qwen3.8-27b): second critic refuses only 1 of the 2 Muse-refused items -> 1 disagreement of 4 (not flagged).
    # Arm B (gemma-4-31b-it): second critic refuses neither of the 2 Muse-refused items -> 2 disagreements of 4 (flagged).
    answers: dict = {}
    for p in (w0_a, w0_b):
        h, _ = waves.read(p)
        answers.update(waves.load_answers(h))
    one_of_a = {sorted(ref_a)[0]}
    second_critic = MockCritic(answers=answers, disagree=one_of_a)

    report = run_rescore(
        RescoreConfig(
            runs=(tar_path,), pack=pack_dir, writers_file=ROOT / "writers.json",
            critic_model="nemotron-3.5-lightning-30b-a3b", seed=5, out=tmp_path / "flag-out", mock=True,
        ),
        critic=second_critic,
    )
    rep_a = report["arms"]["qwen3.8-27b"]
    rep_b = report["arms"]["gemma-4-31b-it"]

    assert rep_a["n"] == 4 and rep_a["verdict_agreements"] == 3 and rep_a["disagreements"] == 1
    assert rep_a["group_split"] == "accepted 2/2, refused 1/2"
    assert rep_a["flagged"] is False

    assert rep_b["n"] == 4 and rep_b["verdict_agreements"] == 2 and rep_b["disagreements"] == 2
    assert rep_b["group_split"] == "accepted 2/2, refused 0/2"
    assert rep_b["flagged"] is True
    assert report["flagged"] == ["gemma-4-31b-it"]


def test_mechanical_gate_assert_stops_on_a_tampered_candidate(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    w0_p, w1_p, _ = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "arm",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    # Tamper with every awaiting_critique candidate in w0 so any drawn item fails Runner.mechanical, and reseal w0.
    h0, d0 = waves.read(w0_p)
    for d in d0:
        if d.get("candidate"):
            d["candidate"]["text"] = "not valid json after tampering"
    waves.write(w0_p, h0, d0)

    out_dir = tmp_path / "tampered-out"
    with pytest.raises(RescoreError, match="mechanical check.*differed from w0"):
        run_rescore(RescoreConfig(
            drafts=(f"qwen3.8-27b={w0_p},{w1_p}",),
            pack=pack_dir,
            writers_file=ROOT / "writers.json",
            critic_model="nemotron-3.5-lightning-30b-a3b",
            seed=1,
            out=out_dir,
            mock=True,
        ))
    # sample-manifest.json was written before scoring started, while rescore.json was not written.
    assert (out_dir / "sample-manifest.json").exists()
    assert not (out_dir / "rescore.json").exists()


def test_a_group_short_of_two_is_drawn_as_it_stands_and_noted_in_the_manifest(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    w0_p, w1_p, _ = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "arm-short",
        writer_model="granite-4.2-30b", writer_family="granite", refuse_count=1, monkeypatch=monkeypatch,
    )
    out_dir = tmp_path / "short-out"
    report = run_rescore(RescoreConfig(
        drafts=(f"granite-4.2-30b={w0_p},{w1_p}",),
        pack=pack_dir,
        writers_file=ROOT / "writers.json",
        critic_model="nemotron-3.5-lightning-30b-a3b",
        seed=0,
        out=out_dir,
        mock=True,
    ))
    manifest = json.loads((out_dir / "sample-manifest.json").read_text(encoding="utf-8"))
    arm_man = manifest["arms"]["granite-4.2-30b"]
    assert arm_man["available"] == {"accepted": 7, "refused": 1}
    assert arm_man["drawn"] == {"accepted": 2, "refused": 1}
    assert arm_man["short"] is True
    assert "refused" in arm_man["short_groups"]
    assert "1 of 2" in arm_man["note"]
    assert report["arms"]["granite-4.2-30b"]["n"] == 3


def test_sealed_drafts_file_with_no_tar_is_rescored_and_bad_seal_is_refused(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    w0_p, w1_p, _ = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "arm",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    # Copy only the sealed drafts files and their .sha256 seals to a directory with no tar and no run dir files.
    standalone = tmp_path / "standalone"
    standalone.mkdir()
    s_w0 = standalone / "qwen-w0.jsonl"
    s_w1 = standalone / "qwen-w1.jsonl"
    for src, dst in ((w0_p, s_w0), (w1_p, s_w1)):
        dst.write_bytes(src.read_bytes())
        waves.seal_path(dst).write_text(waves.seal_path(src).read_text(encoding="utf-8"), encoding="utf-8")

    out_ok = tmp_path / "drafts-ok"
    assert main([
        "rescore",
        "--drafts", f"qwen3.8-27b={s_w0},{s_w1}",
        "--pack", str(pack_dir),
        "--writers", str(ROOT / "writers.json"),
        "--critic-model", "nemotron-3.5-lightning-30b-a3b",
        "--seed", "4",
        "--out", str(out_ok),
        "--mock",
    ]) == 0
    report = json.loads((out_ok / "rescore-report.json").read_text(encoding="utf-8"))
    assert report["arms"]["qwen3.8-27b"]["n"] == 4

    # Corrupt a byte of s_w1 without updating its .sha256 seal: refused.
    s_w1.write_bytes(s_w1.read_bytes() + b"\n")
    out_bad = tmp_path / "drafts-bad"
    assert main([
        "rescore",
        "--drafts", f"qwen3.8-27b={s_w0},{s_w1}",
        "--pack", str(pack_dir),
        "--writers", str(ROOT / "writers.json"),
        "--critic-model", "nemotron-3.5-lightning-30b-a3b",
        "--seed", "4",
        "--out", str(out_bad),
        "--mock",
    ]) == 2


def test_old_format_w0_header_fixture_is_sampled_blinded_and_scored_like_any_other(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    w0_p, w1_p, ref_set = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "granite",
        writer_model="granite-4.2-30b", writer_family="granite", refuse_count=2, monkeypatch=monkeypatch,
    )
    # Replace w0's header with the real window 12b old-format Granite header (card pins in batch_pins, none in writer_pins),
    # keeping mock_answers for MockCritic, and strip card pins from w1's header as the pre-fix critique did.
    fixture_h0 = json.loads(OLD_GRANITE_W0_HEADER.read_text(encoding="utf-8"))
    assert waves.CARD_KEYS <= set(fixture_h0["batch_pins"])
    assert not (waves.CARD_KEYS & set(fixture_h0["writer_pins"]))

    h0, d0 = waves.read(w0_p)
    old_h0 = {**fixture_h0, "mock_answers": h0.get("mock_answers")}
    waves.write(w0_p, old_h0, d0)

    h1, d1 = waves.read(w1_p)
    h1["batch_pins"] = {k: v for k, v in h1["batch_pins"].items() if k not in waves.CARD_KEYS}
    h1["writer_pins"] = {k: v for k, v in fixture_h0["writer_pins"].items() if k not in waves.CARD_KEYS}
    waves.write(w1_p, h1, d1)

    captured: list[list[dict]] = []
    base_critic = MockCritic(answers=waves.load_answers(old_h0), disagree=ref_set)

    def spy_critic(messages: list[dict], **kwargs):
        captured.append(messages)
        return base_critic(messages, **kwargs)

    out_dir = tmp_path / "old-header-out"
    report = run_rescore(
        RescoreConfig(
            drafts=(f"granite-4.2-30b={w0_p},{w1_p}",),
            pack=pack_dir,
            writers_file=ROOT / "writers.json",
            critic_model="nemotron-3.5-lightning-30b-a3b",
            seed=9,
            out=out_dir,
            mock=True,
        ),
        critic=spy_critic,
    )
    assert report["arms"]["granite-4.2-30b"]["n"] == 4
    assert report["arms"]["granite-4.2-30b"]["verdict_agreements"] == 4
    blob = json.dumps(captured)
    assert "granite-4.2-30b" not in blob and "ibm-granite" not in blob
    assert "release-colab-external-images_20261014-060047_RC00" not in blob


def test_a_sample_that_would_fail_the_batch_mix_rule_still_rescores_to_the_end(
    tmp_path: Path, domain_table: Path, pack_dir: Path, monkeypatch
):
    from role_set.gates import load_gates
    from role_set.runner import Runner

    # Build a plan of 4 single-turn D slots all keyed `supported` with escalate="no": a 4-item sample that fails
    # aspire-si-gates' batch_check (cannot_tell and supported shares, ESCALATE, multi-turn D).
    plan = tmp_path / "unbalanced-plan.json"
    assert main(["plan", "--pilot", "--seed", "0", "--domains", str(domain_table), "--out", str(plan)]) == 0
    data = json.loads(plan.read_text(encoding="utf-8"))
    d_slots = [s for s in data["slots"] if s["kind"] == "D"][:4]
    for i, s in enumerate(d_slots):
        s["id"] = f"T02-D-0{i + 1}"
        s["verdict"] = "supported"
        s["escalate"] = "no"
        s["two_turn"] = None
    data["slots"] = d_slots
    plan.write_text(json.dumps(data, indent=1), encoding="utf-8")

    w0_p, w1_p, _ = _make_arm_waves(
        plan, pack_dir, tmp_path / "unbalanced-arm",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    # Prove that the 4 candidates really fail gates.check_batch on their own.
    gates = load_gates()
    _, d0 = waves.read(w0_p)
    rows = []
    for d in d0:
        st, c = waves.state_of(d), waves.completion_of(d)
        runner = Runner(gates=gates, pack={}, trait_names={"T02": "go to the source"}, writer=None, critic=None, out_dir=tmp_path / "chk")
        ref, cand, _ = runner.mechanical(st, c)
        assert ref is None and cand is not None
        rows.append(cand.row)
    with pytest.raises(gates.error_type()):
        gates.check_batch(rows, models={})

    # rescore must not assert batch_check and must finish cleanly.
    out_dir = tmp_path / "batch-mix-out"
    report = run_rescore(RescoreConfig(
        drafts=(f"qwen3.8-27b={w0_p},{w1_p}",),
        pack=pack_dir,
        writers_file=ROOT / "writers.json",
        critic_model="nemotron-3.5-lightning-30b-a3b",
        seed=0,
        out=out_dir,
        mock=True,
    ))
    assert report["arms"]["qwen3.8-27b"]["n"] == 4
    assert (out_dir / "rescore-report.json").exists()


def test_real_critic_path_runs_at_card_values_and_screen_seed_rule(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch
):
    w0_p, w1_p, _ = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "real-arm",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=2, monkeypatch=monkeypatch,
    )
    h0, _ = waves.read(w0_p)
    listed = json.loads((ROOT / "writers.json").read_text(encoding="utf-8"))
    critic_name = "nemotron-3.5-lightning-30b-a3b"
    critic_entry = next(e for e in listed["critics"] if e["served_as"] == critic_name)
    stub = StubLlama("qwen3.8-27b", critic_name, critic=MockCritic(answers=waves.load_answers(h0)))
    try:
        out_dir = tmp_path / "real-out"
        rc = main([
            "rescore",
            "--drafts", f"qwen3.8-27b={w0_p},{w1_p}",
            "--pack", str(pack_dir),
            "--writers", str(ROOT / "writers.json"),
            "--cards", str(CARDS),
            "--critic-endpoint", stub.url,
            "--critic-model", critic_name,
            "--critic-gguf-sha256", critic_entry["gguf_sha256"],
            "--seed", "2",
            "--out", str(out_dir),
        ])
        assert rc == 0
        cards_data = json.loads(CARDS.read_text(encoding="utf-8"))
        card = next(c for c in cards_data["candidates"] if c["candidate"] == critic_name)
        assert stub.requests
        for req in stub.requests:
            assert all(req.get(k) == v for k, v in card["sampling"].items())
        # Trait vote calls vary the seed across 0, 1, 2 (the screen's critic seed rule).
        seeds = {req.get("seed") for req in stub.requests}
        assert {0, 1, 2} <= seeds
    finally:
        stub.close()


def test_rescore_input_validation_and_cli_refusals(
    screen_plan: Path, pack_dir: Path, tmp_path: Path, monkeypatch, capsys
):
    w0_p, w1_p, _ = _make_arm_waves(
        screen_plan, pack_dir, tmp_path / "val-arm",
        writer_model="qwen3.8-27b", writer_family="qwen", refuse_count=0, monkeypatch=monkeypatch,
    )
    # Missing --runs and --drafts is refused via CLI (returns exit code 2).
    assert main(["rescore", "--pack", str(pack_dir), "--seed", "0", "--out", str(tmp_path / "o1"), "--mock"]) == 2
    assert "REFUSED" in capsys.readouterr().err

    # Malformed --drafts specs are refused.
    with pytest.raises(RescoreError, match="--drafts must be"):
        run_rescore(RescoreConfig(drafts=("no-equals",), pack=pack_dir, seed=0, out=tmp_path / "o2", mock=True))
    with pytest.raises(RescoreError, match="--drafts must be"):
        run_rescore(RescoreConfig(drafts=("arm=only_one_path",), pack=pack_dir, seed=0, out=tmp_path / "o3", mock=True))

    # Duplicate arm across inputs is refused.
    with pytest.raises(RescoreError, match="given more than once"):
        run_rescore(RescoreConfig(
            drafts=(f"qwen3.8-27b={w0_p},{w1_p}", f"qwen3.8-27b={w0_p},{w1_p}"),
            pack=pack_dir, seed=0, out=tmp_path / "o4", mock=True,
        ))

    # Missing tar or empty tar is refused.
    with pytest.raises(RescoreError, match="cannot read --runs tar"):
        run_rescore(RescoreConfig(runs=(tmp_path / "missing.tar",), pack=pack_dir, seed=0, out=tmp_path / "o5", mock=True))
    empty_dir = tmp_path / "empty_pilot"
    empty_dir.mkdir()
    empty_tar = _pack_tar(tmp_path / "empty.tar", empty_dir)
    with pytest.raises(RescoreError, match="no arm with w0 and w1 drafts"):
        run_rescore(RescoreConfig(runs=(empty_tar,), pack=pack_dir, seed=0, out=tmp_path / "o5b", mock=True))

    # Non-mock without --cards is refused.
    with pytest.raises(RescoreError, match="pass --cards"):
        run_rescore(RescoreConfig(
            drafts=(f"qwen3.8-27b={w0_p},{w1_p}",),
            pack=pack_dir, writers_file=ROOT / "writers.json", critic_model="nemotron-3.5-lightning-30b-a3b",
            seed=0, out=tmp_path / "o6", mock=False,
        ))

    # Mismatched slot lists between w0 and w1 are refused.
    h1, d1 = waves.read(w1_p)
    short_w1 = tmp_path / "short-w1.jsonl"
    waves.write(short_w1, h1, d1[:-1])
    with pytest.raises(RescoreError, match="w0 and w1 slots do not match"):
        run_rescore(RescoreConfig(
            drafts=(f"qwen3.8-27b={w0_p},{short_w1}",),
            pack=pack_dir, seed=0, out=tmp_path / "o7", mock=True,
        ))


