"""Exemplars: approved items from the merged batches, served to the writer by kind, from a different trait.

They come from aspire-si's data files at a pinned commit, read with `git show`
like the pack. Only merged, reviewed files count; the writer sees their form,
never their ids, and the batch checks count exemplar spans so a skeleton can't
cross traits through them.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

DATA_DIR = "docs/runs/stage1-training-data"
FORBIDDEN = ("sealed", "interview", "taskset", "pilot-30")


class ExemplarError(RuntimeError):
    """The exemplars couldn't be read as pinned. The runner then runs without them."""


def _git_show(repo: Path, commit: str, path: str) -> str:
    for marker in FORBIDDEN:
        if marker in path.lower():
            raise ExemplarError(f"REFUSED: {path} is held out")
    proc = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise ExemplarError(f"REFUSED: cannot read {path} at {commit[:12]}")
    return proc.stdout


def load_exemplars(repo: Path, commit: str, files: list[str], exclude_ids: frozenset[str] = frozenset()) -> dict[str, list[dict]]:
    """{lesson: [rows]} from the named data files (e.g. pilot-t04.jsonl, batch2-t01.jsonl) at one commit, without the
    excluded ids: reviewed items R&D found teaching a sibling trait's step, which would teach the writer the same blur."""

    by_lesson: dict[str, list[dict]] = {}
    for name in files:
        text = _git_show(repo, commit, f"{DATA_DIR}/{name}")
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("id") in exclude_ids:
                    continue
                by_lesson.setdefault(row["lesson"], []).append(row)
    return by_lesson


def load_exclusions(path: Path) -> tuple[frozenset[str], str]:
    """The exemplar exclusion list (roles/<role>/exemplar-exclude.json): its ids, and the file's sha256 for the pins."""

    import hashlib

    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return frozenset(e["id"] for e in data["excluded"]), hashlib.sha256(raw).hexdigest()


def pick(by_lesson: dict[str, list[dict]], *, kind: str, exclude_lesson: str, n: int = 3, seed: int = 0) -> list[dict]:
    """n exemplars of one kind from traits other than the slot's, ids stripped, rotated by seed."""

    pool = [r for lesson, rows in sorted(by_lesson.items()) if lesson != exclude_lesson for r in rows if r.get("kind") == kind]
    rng = random.Random(f"{kind}:{exclude_lesson}:{seed}")
    rng.shuffle(pool)
    out = []
    for r in pool[:n]:
        shown = {k: v for k, v in r.items() if k in ("kind", "turns", "key_points", "answer", "quiz", "variant")}
        out.append(shown)
    return out


def spans_of(exemplars: list[dict], n: int = 8) -> set[tuple[str, ...]]:
    """8-word spans of the exemplars' assistant text, for the batch checks."""

    import re

    out: set[tuple[str, ...]] = set()
    for e in exemplars:
        for t in e.get("turns", []):
            words = re.sub(r"[^a-z0-9 ]+", " ", (t.get("thinking", "") + " " + t.get("reply", "")).lower()).split()
            out |= {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}
    return out
