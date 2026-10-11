import json
import types
from pathlib import Path

import pytest

from role_set.assemble import Material
from role_set.gates import Gates
from role_set.mock import MockCritic, MockWriter
from role_set.plan import plan_lesson
from role_set.runner import SUGGESTIONS, RunHalted, Runner, truth_checks

SIDES = ("quotes the source", "stops at the source")
PACK = {"traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n", "kinds.md": "kinds\n", "amendments/1.md": "## Amendment 1\n\nq\n"}
TRAITS = {"T02": "goes to the source", "T03": "precise"}


def _gates(refuse_ids=(), against=None, role_words=()):
    m = types.SimpleNamespace()
    m.DataError = ValueError

    def check_item(row):
        if row["id"] in refuse_ids:
            raise ValueError(f"refused {row['id']}")

    m.check_item = check_item
    m.check_against = against or (lambda row, accepted: [])
    m.check_batch = lambda rows: None
    m.role_words_in_prompt = lambda row: list(role_words) if row["kind"] in "CD" else []
    m.critic_mismatch = lambda key, critic, material: not ({" ".join(x.split()) for x in key} & {" ".join(x.split()) for x in critic})
    m.answer_format = types.SimpleNamespace(parse_answer=lambda reply: {
        "verdict": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("VERDICT:")), None),
        "deciding": next((l.split(":", 1)[1].strip() for l in reply.splitlines() if l.startswith("DECIDING:")), None),
    })
    return Gates(module=m, version="1.1.0", sha256="g" * 64)


def _runner(tmp_path: Path, **kw):
    answers = {}
    writer = MockWriter(answers=answers, fail_first=kw.pop("fail_first", None))
    critic = MockCritic(answers=answers, disagree=kw.pop("disagree", None), trait="T02")
    excerpts = {"changelog": [Material("## [1.0.0] - 2026-01-01\n- Added the thing.\n", {"repo": "x/y", "commit": "abc", "path": "C.md", "lines": "L1", "licence": "MIT"})]}
    r = Runner(gates=kw.pop("gates", _gates()), pack=PACK, trait_names=TRAITS, writer=writer, critic=critic, out_dir=tmp_path / "out", excerpts=excerpts, pins={"writer_revision": "deadbeef"}, **kw)
    return r, writer, critic


def test_a_whole_lesson_runs_end_to_end_with_the_mock(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    slots = plan_lesson("T02", SIDES, seed=0)
    outcomes = r.run(slots, {"T02": SIDES})
    assert all(o.accepted for o in outcomes), [(o.slot_id, o.refusals) for o in outcomes if not o.accepted]
    rows = [json.loads(l) for l in (tmp_path / "out" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and {x["kind"] for x in rows} == set("ABCDE")
    receipts = [json.loads(l) for l in (tmp_path / "out" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert receipts[0]["gates"]["version"] == "1.1.0" and receipts[0]["writer_revision"] == "deadbeef"
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["accepted"] == 40 and status["dropped"] == 0 and status["critic_agreement"] == 1.0
    assert status["actions"][-1]["action"] == "end" and "shut the pod down" in status["actions"][-1]["reminder"]
    assert (tmp_path / "out" / "warmup-T02.json").exists()
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert len(report["outcomes"]) == 40
    real = [x for x in rows if x["key_basis"] == "source"]
    assert real and real[0]["sources"][0]["repo"] == "x/y"
    assert report["batch_check"] == "clean" and r.batch_check == "clean"


def test_the_finished_set_goes_through_check_batch_and_a_refusal_is_reported(tmp_path: Path):
    gates = _gates()
    seen = []

    def refuse(rows):
        seen.append(len(rows))
        raise ValueError("T02: 0 of 12 verdict-bearing items have 2-4 DECIDING lines")

    gates.module.check_batch = refuse
    r, _, _ = _runner(tmp_path, gates=gates)
    r.run(plan_lesson("T02", SIDES, seed=0), {"T02": SIDES})
    report = json.loads((tmp_path / "out" / "batch-report.json").read_text(encoding="utf-8"))
    assert seen == [40] and report["batch_check"].startswith("REFUSED: T02: 0 of 12")
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert any(a["action"] == "batch_check" and a["result"].startswith("REFUSED") for a in status["actions"])


def test_refusal_becomes_a_suggestion_and_the_retry_is_accepted(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    d = next(s for s in slots if s.kind == "D" and not s.two_turn)
    r, writer, _ = _runner(tmp_path, fail_first={f"{d.lesson}-{d.kind}"})
    assert r.warmup("T02", SIDES)
    out = r.run_slot(d)
    assert out.accepted and out.attempts == 2
    assert out.refusals[0]["code"] == "PARSE" and "JSON object" in out.refusals[0]["suggestion"]
    assert any("Refused:" in s for s in writer.seen)


def test_gate_refusals_critic_mismatch_and_role_words(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    d = next(s for s in slots if s.kind == "D" and not s.two_turn)
    r, _, critic = _runner(tmp_path, gates=_gates(refuse_ids={d.id}))
    out = r.run_slot(d)
    assert not out.accepted and out.attempts == 3 and {x["code"] for x in out.refusals} == {"GATE"}
    assert "validator refused" in out.refusals[0]["suggestion"]

    r2, _, critic2 = _runner(tmp_path / "b")
    critic2.disagree_all = True
    d2 = next(s for s in slots if s.kind == "D" and s.verdict == "supported" and not s.two_turn)
    out2 = r2.run_slot(d2)
    assert not out2.accepted and out2.refusals[0]["code"] == "VERDICT_MISMATCH" and "second reader" in out2.refusals[0]["suggestion"]

    r3, _, _ = _runner(tmp_path / "c", gates=_gates(role_words=("verifier",)))
    out3 = r3.run_slot(d)
    assert out3.refusals[0]["code"] == "ROLE_WORDS" and "amendment 3" in out3.refusals[0]["suggestion"]

    r4, _, _ = _runner(tmp_path / "d", gates=_gates(against=lambda row, acc: [("FRAMING_REUSED", "the framing is already used")]))
    out4 = r4.run_slot(d)
    assert out4.refusals[0]["code"] == "AGAINST" and "FRAMING_REUSED" in out4.refusals[0]["detail"]


def test_andon_halts_on_the_drop_share(tmp_path: Path):
    slots = plan_lesson("T02", SIDES, seed=0)
    r, _, _ = _runner(tmp_path, gates=_gates(refuse_ids={s.id for s in slots if s.kind == "A"}))
    with pytest.raises(RunHalted, match="andon"):
        r.run(slots, {"T02": SIDES})
    status = json.loads((tmp_path / "out" / "status.json").read_text(encoding="utf-8"))
    assert status["halted"].startswith("andon") and status["dropped"] >= 1
    assert (tmp_path / "out" / "batch-report.json").exists()


def test_warmup_failure_halts_before_any_slot(tmp_path: Path):
    r, writer, _ = _runner(tmp_path)
    writer.__class__.__call__  # noqa: B018 - the writer is replaced below
    r.writer = lambda messages, seed=0: types.SimpleNamespace(text="not json", prompt_tokens=1, completion_tokens=1, model="m", seconds=0)
    with pytest.raises(RunHalted, match="warm-up"):
        r.run(plan_lesson("T02", SIDES, seed=0), {"T02": SIDES})
    assert r.status.attempted == 0


def test_concurrency_path_and_truth_checks(tmp_path: Path):
    r, _, _ = _runner(tmp_path, concurrency=3)
    outcomes = r.run(plan_lesson("T02", SIDES, seed=2), {"T02": SIDES})
    assert sum(o.accepted for o in outcomes) == 40, [(o.slot_id, o.refusals) for o in outcomes if not o.accepted]
    assert r.status.concurrency == 3

    row = {"material": "Desk hours\nThe desk is open on Sundays from 10:00 to 14:00.\n", "turns": [{"user": "Statement: x", "thinking": 'It says "the desk is shut on Sundays" here.', "reply": "r"}]}
    assert truth_checks(row, None) == ("TRUTH_QUOTE", "the desk is shut on Sundays")
    row["turns"][0]["thinking"] = "The count is 37, not 14:00."
    assert truth_checks(row, None) == ("TRUTH_NUMBER", "37")
    row["turns"][0]["thinking"] = "1. The line says 10:00 to 14:00. 2. Done."
    assert truth_checks(row, None) is None
    assert truth_checks({"material": "", "turns": []}, None) is None


def test_suggestion_table_is_pinned():
    assert set(SUGGESTIONS) == {"ASSEMBLY", "TRUTH_QUOTE", "TRUTH_NUMBER", "GATE", "ROLE_WORDS", "VERDICT_MISMATCH", "EVIDENCE_MISMATCH",
                                "FLAW_MISMATCH", "ANSWER_MISMATCH", "TRAIT_MISMATCH", "VOID", "AGAINST", "PARSE"}
    assert all("{detail}" in v for v in SUGGESTIONS.values())
