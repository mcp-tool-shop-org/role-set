"""ingest: the trainer's JSONL receipts into the store, with the firewall at the door.

Refused, row by row, with a reason: a source outside the closed list; an arm
outside ROLE and NO-ROLE; any `q` or `sitting` field (the pre-interview's shape);
any id with a sealed (sx-) or DEV (pl-, rv-) prefix; any text sharing an 8-word
span with the twelve interview questions; a row whose round, arm or seed don't
match the directory it came from. Refused before any row is read: a receipts
directory that isn't `.../stage1-train/<round>/<arm>/<seed>/receipts`, or any
path that names a held-out set.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from role_set.store import ARMS, SOURCES, begin_ingest, finish_ingest, insert_diagnostics, insert_quiz, open_store

SCHEMA = "stage1-receipt-v1"
HELD_OUT_PREFIXES = {"sx-": "SEALED", "pl-": "DEV", "rv-": "DEV"}
# An absolute path: a drive letter, a UNC share, or a POSIX root (including Git Bash's /e/). Diagnostics rows carry
# backup destinations and stop reasons; the store feeds round reports, and paths stay out of anything shared.
ABS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|^\\\\[^\\]+\\|(?<![\w.:/])/(?:[a-z]|home|Users|mnt|root|tmp|var|opt|srv)/")
DENY_MARKERS = ("sealed", "interview", "pilot-30", "stage1-eval", "taskset")
INTERVIEW_FILE = "docs/runs/stage1-eval/interview.py"
SPAN = 8


class IngestError(RuntimeError):
    """The receipts directory or the questions could not be used. Nothing is written."""


@dataclass
class Refusal:
    file: str
    line: int
    code: str
    reason: str

    def as_dict(self) -> dict:
        return {"file": self.file, "line": self.line, "code": self.code, "reason": self.reason}


@dataclass
class IngestResult:
    ingest_id: int
    files: int
    accepted: int
    duplicates: int
    refusals: list[Refusal] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ingest_id": self.ingest_id,
            "files": self.files,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "refused": len(self.refusals),
            "refusals": [r.as_dict() for r in self.refusals],
        }


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def shingles(text: str, n: int = SPAN) -> set[tuple[str, ...]]:
    w = _norm_words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def interview_questions(repo: Path, commit: str, *, expected_sha_prefix: str | None = None) -> list[str]:
    """The twelve questions, parsed as a literal from the public eval file at a pinned commit, held in memory only."""

    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{INTERVIEW_FILE}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise IngestError(f"REFUSED: cannot read the interview questions at {commit[:12]}")
    return parse_questions(proc.stdout, expected_sha_prefix=expected_sha_prefix)


def parse_questions(source: str, *, expected_sha_prefix: str | None = None) -> list[str]:
    m = re.search(r"^QUESTIONS\s*=\s*(\[.*?^\])", source, re.M | re.S)
    if not m:
        raise IngestError("REFUSED: no QUESTIONS list found in the interview file")
    questions = ast.literal_eval(m.group(1))
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions) or len(questions) != 12:
        raise IngestError("REFUSED: QUESTIONS is not a list of twelve strings")
    sha = hashlib.sha256(json.dumps(questions).encode()).hexdigest()
    if expected_sha_prefix and not sha.startswith(expected_sha_prefix):
        raise IngestError(f"REFUSED: the questions hash {sha[:12]} doesn't start with {expected_sha_prefix}")
    return questions


def check_receipts_dir(receipts: Path) -> tuple[int, str, int]:
    """`.../stage1-train/<round>/<arm>/<seed>/receipts`, with no held-out marker anywhere in the path."""

    parts = [p for p in receipts.resolve().parts]
    lowered = "/".join(parts).lower()
    for marker in DENY_MARKERS:
        if marker in lowered:
            raise IngestError(f"REFUSED: the path names a held-out set ({marker})")
    if len(parts) < 5 or parts[-1] != "receipts" or parts[-5] != "stage1-train":
        raise IngestError("REFUSED: receipts must sit at .../stage1-train/<round>/<arm>/<seed>/receipts")
    round_s, arm, seed_s = parts[-4], parts[-3], parts[-2]
    if not round_s.isdigit() or not seed_s.isdigit() or arm not in ARMS:
        raise IngestError("REFUSED: the path's round, arm or seed is not of the expected form")
    if not receipts.is_dir():
        raise IngestError("REFUSED: the receipts directory does not exist")
    return int(round_s), arm, int(seed_s)


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def check_row(row: dict, *, path_round: int, path_arm: str, path_seed: int, question_shingles: set, allow_dev: bool) -> tuple[str, str] | None:
    """A refusal (code, reason), or None when the row may enter."""

    if not isinstance(row, dict):
        return ("SHAPE", "row is not an object")
    if row.get("schema") != SCHEMA:
        return ("SCHEMA", f"schema {row.get('schema')!r} is not {SCHEMA}")
    if "q" in row or "sitting" in row:
        return ("INTERVIEW", "row carries the pre-interview's fields (q or sitting)")
    src = row.get("source")
    if src not in SOURCES:
        return ("SOURCE", f"source {src!r} is outside the closed list")
    if src == "dev_between_stages" and not allow_dev:
        return ("DEV", "dev_between_stages rows need --dev-between-stages with a dated amendment")
    if row.get("arm") not in ARMS:
        return ("ARM", f"arm {row.get('arm')!r} is not ROLE or NO-ROLE")
    for key in ("round", "seed"):
        if not isinstance(row.get(key), int) or isinstance(row.get(key), bool):
            return ("SHAPE", f"{key} must be an integer")
    if (row["round"], row["arm"], row["seed"]) != (path_round, path_arm, path_seed):
        return ("PATH", "the row's round, arm or seed don't match its directory")
    if not isinstance(row.get("run_id"), str) or not isinstance(row.get("written_at"), str):
        return ("SHAPE", "run_id and written_at are required strings")
    for key, val in row.items():
        if key.endswith("_id") and isinstance(val, str):
            for prefix, code in HELD_OUT_PREFIXES.items():
                if val.startswith(prefix):
                    return (code, f"{key} {val!r} is a held-out id")
    if src == "diagnostics":
        for text in _strings(row):
            if ABS_PATH.search(text):
                return ("PATH", "a diagnostics field holds an absolute path; record it relative to the run's output root or as a label")
    if src != "diagnostics":
        for key in ("question_id", "lesson", "kind"):
            if not isinstance(row.get(key), str) or not row[key]:
                return ("SHAPE", f"{key} is required for {src} rows")
    if question_shingles:
        for text in _strings(row):
            if len(text) >= 40 and shingles(text) & question_shingles:
                return ("INTERVIEW", "a text field shares an 8-word span with the interview questions")
    return None


def ingest(receipts: Path, store_path: Path, *, questions: list[str], allow_dev: bool = False, amendment: str | None = None) -> IngestResult:
    path_round, path_arm, path_seed = check_receipts_dir(receipts)
    if allow_dev and not amendment:
        raise IngestError("REFUSED: --dev-between-stages needs --amendment <dated id>")
    q_shingles: set = set()
    for q in questions:
        q_shingles |= shingles(q)
    files = sorted(receipts.glob("*.jsonl"))
    if not files:
        raise IngestError("REFUSED: no .jsonl receipts in the directory")
    conn = open_store(store_path)
    ingest_id = begin_ingest(conn, receipts.name, amendment)
    accepted = duplicates = 0
    refusals: list[Refusal] = []
    for f in files:
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                refusals.append(Refusal(f.name, n, "JSON", str(e)[:120]))
                continue
            bad = check_row(row, path_round=path_round, path_arm=path_arm, path_seed=path_seed, question_shingles=q_shingles, allow_dev=allow_dev)
            if bad:
                refusals.append(Refusal(f.name, n, bad[0], bad[1]))
                continue
            fresh = insert_diagnostics(conn, ingest_id, row) if row["source"] == "diagnostics" else insert_quiz(conn, ingest_id, row)
            if fresh:
                accepted += 1
            else:
                duplicates += 1
    finish_ingest(conn, ingest_id, files=len(files), accepted=accepted, refused=len(refusals), refusals=[r.as_dict() for r in refusals])
    conn.close()
    return IngestResult(ingest_id=ingest_id, files=len(files), accepted=accepted, duplicates=duplicates, refusals=refusals)
