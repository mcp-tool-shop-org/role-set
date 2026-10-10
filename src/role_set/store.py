"""The shared store: one SQLite file, append-only, with the firewall at its only door.

The trainer never opens this database. It writes JSONL receipts; `ingest` is the
one write path (with `write`'s own rows later). Nothing here updates or deletes a
row. A round is voided by appending to `voids`, and every reader skips voided rounds.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SOURCES = ("train_quiz", "rehearsal", "diagnostics", "dev_between_stages")
ARMS = ("ROLE", "NO-ROLE")

_MIGRATIONS = {
    1: """
    CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS ingests (
        id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, receipts_dir TEXT NOT NULL,
        amendment TEXT, files INTEGER NOT NULL, accepted INTEGER NOT NULL, refused INTEGER NOT NULL,
        refusals TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS receipts_quiz (
        id INTEGER PRIMARY KEY, ingest_id INTEGER NOT NULL, row_sha TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL CHECK (source IN ('train_quiz','rehearsal','dev_between_stages')),
        round INTEGER NOT NULL, seed INTEGER NOT NULL, arm TEXT NOT NULL CHECK (arm IN ('ROLE','NO-ROLE')),
        run_id TEXT NOT NULL, question_id TEXT NOT NULL, base_item_id TEXT, lesson TEXT NOT NULL, kind TEXT NOT NULL,
        tier INTEGER, attempt INTEGER, correct_pinned INTEGER, correct_v2 INTEGER, found_evidence INTEGER,
        stuck INTEGER, wall_s REAL, thinking_tokens INTEGER, answer_tokens INTEGER,
        verdict_key TEXT, verdict_given TEXT, reread_rate REAL, hedge_rate REAL, burst_count INTEGER,
        trace_present INTEGER NOT NULL, payload TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS receipts_diagnostics (
        id INTEGER PRIMARY KEY, ingest_id INTEGER NOT NULL, row_sha TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL CHECK (source = 'diagnostics'),
        round INTEGER NOT NULL, seed INTEGER NOT NULL, arm TEXT NOT NULL CHECK (arm IN ('ROLE','NO-ROLE')),
        run_id TEXT NOT NULL, payload TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS voids (id INTEGER PRIMARY KEY, round INTEGER NOT NULL, reason TEXT NOT NULL, voided_at TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS quiz_cell ON receipts_quiz (lesson, kind, round, arm, seed);
    """,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_sha(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    current = 0
    try:
        got = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
        current = int(got or 0)
    except sqlite3.OperationalError:
        current = 0
    for version in range(current + 1, SCHEMA_VERSION + 1):
        conn.executescript(_MIGRATIONS[version])
        conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)", (version, now_iso()))
    conn.commit()
    return conn


def begin_ingest(conn: sqlite3.Connection, receipts_dir_name: str, amendment: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO ingests (started_at, receipts_dir, amendment, files, accepted, refused, refusals) VALUES (?, ?, ?, 0, 0, 0, '[]')",
        (now_iso(), receipts_dir_name, amendment),
    )
    return int(cur.lastrowid)


def finish_ingest(conn: sqlite3.Connection, ingest_id: int, *, files: int, accepted: int, refused: int, refusals: list[dict]) -> None:
    # The ingest record is the one row that is completed after insertion; it is the ingest's own receipt.
    conn.execute(
        "UPDATE ingests SET files = ?, accepted = ?, refused = ?, refusals = ? WHERE id = ? AND files = 0",
        (files, accepted, refused, json.dumps(refusals), ingest_id),
    )
    conn.commit()


def insert_quiz(conn: sqlite3.Connection, ingest_id: int, row: dict) -> bool:
    """Returns False when the row was already in the store (same bytes)."""

    sha = row_sha(row)
    if conn.execute("SELECT 1 FROM receipts_quiz WHERE row_sha = ?", (sha,)).fetchone():
        return False
    conn.execute(
        """INSERT INTO receipts_quiz (ingest_id, row_sha, source, round, seed, arm, run_id, question_id, base_item_id,
           lesson, kind, tier, attempt, correct_pinned, correct_v2, found_evidence, stuck, wall_s, thinking_tokens,
           answer_tokens, verdict_key, verdict_given, reread_rate, hedge_rate, burst_count, trace_present, payload)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            ingest_id, sha, row["source"], row["round"], row["seed"], row["arm"], row["run_id"], row["question_id"],
            row.get("base_item_id"), row["lesson"], row["kind"], row.get("tier"), row.get("attempt"),
            _flag(row.get("correct_pinned")), _flag(row.get("correct_v2")), _flag(row.get("found_evidence")),
            _flag(row.get("stuck")), row.get("wall_s"), row.get("thinking_tokens"), row.get("answer_tokens"),
            row.get("verdict_key"), row.get("verdict_given"), row.get("reread_rate"), row.get("hedge_rate"),
            row.get("burst_count"), 1 if row.get("trace") else 0, json.dumps(row, sort_keys=True),
        ),
    )
    return True


def insert_diagnostics(conn: sqlite3.Connection, ingest_id: int, row: dict) -> bool:
    sha = row_sha(row)
    if conn.execute("SELECT 1 FROM receipts_diagnostics WHERE row_sha = ?", (sha,)).fetchone():
        return False
    conn.execute(
        "INSERT INTO receipts_diagnostics (ingest_id, row_sha, source, round, seed, arm, run_id, payload) VALUES (?,?,?,?,?,?,?,?)",
        (ingest_id, sha, row["source"], row["round"], row["seed"], row["arm"], row["run_id"], json.dumps(row, sort_keys=True)),
    )
    return True


def void_round(conn: sqlite3.Connection, round_: int, reason: str) -> None:
    conn.execute("INSERT INTO voids (round, reason, voided_at) VALUES (?, ?, ?)", (round_, reason.strip(), now_iso()))
    conn.commit()


def voided_rounds(conn: sqlite3.Connection) -> set[int]:
    return {int(r[0]) for r in conn.execute("SELECT DISTINCT round FROM voids")}


def counts(conn: sqlite3.Connection) -> dict:
    skip = voided_rounds(conn)
    out = {"quiz_rows": 0, "diagnostics_rows": 0, "voided_rounds": sorted(skip), "ingests": 0}
    for (r, n) in conn.execute("SELECT round, COUNT(*) FROM receipts_quiz GROUP BY round"):
        if int(r) not in skip:
            out["quiz_rows"] += int(n)
    for (r, n) in conn.execute("SELECT round, COUNT(*) FROM receipts_diagnostics GROUP BY round"):
        if int(r) not in skip:
            out["diagnostics_rows"] += int(n)
    out["ingests"] = int(conn.execute("SELECT COUNT(*) FROM ingests").fetchone()[0])
    return out


def _flag(v) -> int | None:
    if v is None:
        return None
    return 1 if bool(v) else 0
