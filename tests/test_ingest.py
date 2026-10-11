import json
from pathlib import Path

import pytest

from role_set.ingest import IngestError, check_receipts_dir, check_row, ingest, parse_questions, shingles
from role_set.store import counts, open_store, void_round, voided_rounds

QUESTIONS = [f"Question number {i} asks you about how you check a statement against a document carefully." for i in range(12)]
Q_SHINGLES = set().union(*(shingles(q) for q in QUESTIONS))


def quiz_row(**over) -> dict:
    row = {
        "schema": "stage1-receipt-v1", "source": "train_quiz", "round": 2, "seed": 0, "arm": "ROLE", "run_id": "r2-s0-ROLE",
        "adapter_sha256": "ab12", "gates_sha256": "0e81", "stuck_rule_sha256": "716dc251", "written_at": "2026-10-12T14:03:22Z",
        "question_id": "T05-D-03-v2", "base_item_id": "T05-D-03", "variant": 2, "lesson": "T05", "kind": "D", "tier": 4,
        "attempt": 2, "ledger_box": 1, "ledger_state": {"box": 1, "misses": 1, "next_variant": "v3", "due_round": 3},
        "verdict_key": "cannot_tell", "verdict_given": "unsupported", "deciding_given": "It is closed on Sundays.", "escalate_given": "no",
        "correct_pinned": False, "correct_v2": False, "found_evidence": False,
        "wall_s": 21.7, "thinking_tokens": 812, "answer_tokens": 38, "total_tokens": 850, "truncated": False,
        "target_thinking_tokens": 240, "tier_p95": 572, "stuck": True, "stuck_reasons": ["long_not_progressing"],
        "reread_rate": 9.85, "hedge_rate": 4.93, "max_in_window": {"100": 2, "200": 3, "400": 3}, "bursts": [[0.21, 0.37]],
        "burst_count": 1, "gap_unit": "tokens (no streaming timestamps)", "trace": "the thinking text", "answer": "VERDICT: unsupported",
    }
    row.update(over)
    return row


def diag_row(**over) -> dict:
    row = {
        "schema": "stage1-receipt-v1", "source": "diagnostics", "round": 2, "seed": 0, "arm": "ROLE", "run_id": "r2-s0-ROLE",
        "adapter_sha256": None, "gates_sha256": "0e81", "stuck_rule_sha256": "716dc251", "written_at": "2026-10-12T16:40:00Z",
        "steps": 120, "quiz": {"asked": 96, "correct": 71, "stuck": 9}, "stuck_by_cell": [{"lesson": "T05", "kind": "D", "n": 12, "stuck": 3}],
        "proposals": [{"text": "weight T05-D in the next plan", "status": "registered"}],
    }
    row.update(over)
    return row


@pytest.fixture
def receipts(tmp_path: Path) -> Path:
    d = tmp_path / "outputs" / "stage1-train" / "2" / "ROLE" / "0" / "receipts"
    d.mkdir(parents=True)
    return d


def _write(d: Path, name: str, rows: list) -> None:
    (d / name).write_text("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n", encoding="utf-8")


def test_ingest_accepts_clean_rows_and_is_idempotent(receipts: Path, tmp_path: Path):
    _write(receipts, "quiz.jsonl", [quiz_row(), quiz_row(question_id="T05-D-04-v1", base_item_id="T05-D-04", source="rehearsal", attempt=None, trace=None)])
    _write(receipts, "diag.jsonl", [diag_row()])
    db = tmp_path / "store" / "role.db"
    r = ingest(receipts, db, questions=QUESTIONS)
    assert (r.files, r.accepted, r.duplicates, r.refusals) == (2, 3, 0, [])
    again = ingest(receipts, db, questions=QUESTIONS)
    assert (again.accepted, again.duplicates) == (0, 3)
    conn = open_store(db)
    assert counts(conn) == {"quiz_rows": 2, "diagnostics_rows": 1, "voided_rounds": [], "ingests": 2}
    stored = conn.execute("SELECT trace_present, stuck, payload FROM receipts_quiz ORDER BY id").fetchall()
    assert [row["trace_present"] for row in stored] == [1, 0] and stored[0]["stuck"] == 1
    assert json.loads(stored[0]["payload"])["ledger_state"]["box"] == 1
    void_round(conn, 2, "bad adapter")
    assert voided_rounds(conn) == {2}
    assert counts(conn)["quiz_rows"] == 0


def test_ingest_refuses_held_out_rows_with_reasons(receipts: Path, tmp_path: Path):
    rows = [
        quiz_row(question_id="sx-0041"),
        quiz_row(base_item_id="pl-07"),
        quiz_row(question_id="rv-03-v1"),
        quiz_row(q=3, arm="BASE"),
        {"schema": "stage1-receipt-v1", "source": "interview", "round": 2, "seed": 0, "arm": "ROLE", "sitting": "baseline", "run_id": "x", "written_at": "t"},
        quiz_row(arm="BASE"),
        quiz_row(source="dev_between_stages"),
        quiz_row(seed=1),
        quiz_row(schema="v0"),
        quiz_row(round="2"),
        quiz_row(run_id=None),
        quiz_row(lesson=""),
        quiz_row(answer="Here is the text: " + QUESTIONS[3]),
        "not json",
    ]
    _write(receipts, "mixed.jsonl", rows + [quiz_row()])
    r = ingest(receipts, tmp_path / "role.db", questions=QUESTIONS)
    codes = [x.code for x in r.refusals]
    assert codes == ["SEALED", "DEV", "DEV", "INTERVIEW", "INTERVIEW", "ARM", "DEV", "PATH", "SCHEMA", "SHAPE", "SHAPE", "SHAPE", "INTERVIEW", "JSON"]
    assert r.accepted == 1
    conn = open_store(tmp_path / "role.db")
    rec = conn.execute("SELECT accepted, refused, refusals FROM ingests").fetchone()
    assert (rec["accepted"], rec["refused"]) == (1, 14) and json.loads(rec["refusals"])[0]["code"] == "SEALED"


def test_dev_between_stages_needs_the_flag_and_an_amendment(receipts: Path, tmp_path: Path):
    _write(receipts, "dev.jsonl", [quiz_row(source="dev_between_stages")])
    with pytest.raises(IngestError, match="--amendment"):
        ingest(receipts, tmp_path / "role.db", questions=QUESTIONS, allow_dev=True)
    r = ingest(receipts, tmp_path / "role.db", questions=QUESTIONS, allow_dev=True, amendment="2026-11-01-dev-steer")
    assert r.accepted == 1
    conn = open_store(tmp_path / "role.db")
    assert conn.execute("SELECT amendment FROM ingests").fetchone()[0] == "2026-11-01-dev-steer"


def test_receipts_dir_shape_and_deny_list(tmp_path: Path):
    good = tmp_path / "outputs" / "stage1-train" / "3" / "NO-ROLE" / "1" / "receipts"
    good.mkdir(parents=True)
    assert check_receipts_dir(good) == (3, "NO-ROLE", 1)
    with pytest.raises(IngestError, match="held-out"):
        check_receipts_dir(tmp_path / "outputs" / "stage1-baseline" / "interview" / "x" / "y" / "receipts")
    with pytest.raises(IngestError, match="stage1-train"):
        check_receipts_dir(tmp_path / "outputs" / "elsewhere" / "3" / "ROLE" / "1" / "receipts")
    with pytest.raises(IngestError, match="expected form"):
        check_receipts_dir(tmp_path / "outputs" / "stage1-train" / "three" / "ROLE" / "1" / "receipts")
    with pytest.raises(IngestError, match="does not exist"):
        check_receipts_dir(tmp_path / "outputs" / "stage1-train" / "4" / "ROLE" / "1" / "receipts")
    with pytest.raises(IngestError, match="no .jsonl"):
        ingest(good, tmp_path / "role.db", questions=QUESTIONS)


def test_check_row_rejects_non_objects_and_parse_questions_checks_shape_and_hash():
    assert check_row(["x"], path_round=1, path_arm="ROLE", path_seed=0, question_shingles=set(), allow_dev=False) == ("SHAPE", "row is not an object")
    src = "import x\nQUESTIONS = [\n" + "".join(f"    {json.dumps(q)},\n" for q in QUESTIONS) + "]\n\ndef f():\n    pass\n"
    assert parse_questions(src) == QUESTIONS
    with pytest.raises(IngestError, match="doesn't start with"):
        parse_questions(src, expected_sha_prefix="ffffff")
    with pytest.raises(IngestError, match="no QUESTIONS"):
        parse_questions("nothing = 1\n")
    with pytest.raises(IngestError, match="twelve"):
        parse_questions("QUESTIONS = [\n    'only one',\n]\n")


def test_diagnostics_with_an_absolute_path_are_refused(receipts: Path, tmp_path: Path):
    rows = [
        diag_row(state="stopped", stop_reason="disk full", backup={"status": "copied", "to": "D:/archive/stage1/2"}),
        diag_row(run_id="r2-b", backup={"status": "failed", "reason": "cannot write /home/runner/out"}),
        diag_row(run_id="r2-c", state="done", stop_reason=None, backup={"status": "copied", "to": "external-archive"}),
        quiz_row(trace="the excerpt runs /usr/local/bin/tool, and the line decides it"),
    ]
    _write(receipts, "diag.jsonl", rows)
    r = ingest(receipts, tmp_path / "role.db", questions=QUESTIONS)
    assert [x.code for x in r.refusals] == ["PATH", "PATH"]
    # A label or a relative destination enters, and a quiz trace may quote a path from its material.
    assert r.accepted == 2
