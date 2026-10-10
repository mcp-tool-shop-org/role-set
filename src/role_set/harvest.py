"""Real sources: excerpts from permissively licensed repos at pinned commits, fetched by code.

The writer never invents a citation. `harvest` reads an allow-list of repos, each
pinned to a commit, checks the LICENSE's first line, refuses anything on the
exclusion list, cuts candidate excerpts (changelog entries with their version
heading, README paragraphs and tables, short functions) and records
`repo@commit:path#lines` for each. The pool is a local file the runner draws
from; the clone paths never enter it.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_LICENCES = ("MIT License", "Apache License", "BSD")
# The handoff's exclusion list (§4 rule 13): never a source for training material.
EXCLUDED_REPOS = {
    "portlight", "ai-eyes-mcp", "backpropagate", "ai-rpg-engine", "vocal-synth-engine", "prompt-craft", "offrig",
    "aspire-si", "stillpoint", "research-os", "runforge", "loadout-os",
    "role-os", "rnd",
}
PRIVATE_LINE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|[A-Za-z]:\\|/Users/|/home/|\\\\Users\\\\")
CODE_SUFFIXES = (".py", ".ts", ".js", ".rs", ".mjs")


class HarvestError(RuntimeError):
    """A repo can't be used as a source. Nothing from it enters the pool."""


@dataclass
class Excerpt:
    text: str
    content_kind: str
    source: dict

    def as_dict(self) -> dict:
        return {"text": self.text, "content_kind": self.content_kind, "source": self.source}


@dataclass
class HarvestReport:
    repos: int = 0
    excerpts: int = 0
    by_kind: dict = field(default_factory=dict)
    refused: list = field(default_factory=list)
    dropped_private: int = 0

    def as_dict(self) -> dict:
        return {"repos": self.repos, "excerpts": self.excerpts, "by_kind": dict(sorted(self.by_kind.items())), "refused": self.refused, "dropped_private": self.dropped_private}


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise HarvestError(f"git {args[0]} failed in {repo.name}: {proc.stderr.strip()[:160]}")
    return proc.stdout


def licence_line(repo: Path, commit: str) -> str:
    try:
        text = _git(repo, "show", f"{commit}:LICENSE")
    except HarvestError:
        raise HarvestError(f"{repo.name}: no LICENSE file at {commit[:12]}") from None
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not any(first.startswith(a) or a in first for a in ALLOWED_LICENCES):
        raise HarvestError(f"{repo.name}: licence line {first!r} is not on the allowed list")
    return first


def _lines_ref(start: int, end: int) -> str:
    return f"L{start}" if start == end else f"L{start}-L{end}"


def changelog_excerpts(text: str, *, max_lines: int = 12) -> list[tuple[str, int, int]]:
    """Each version section: its heading plus up to max_lines following lines, stopping at the next version heading."""

    lines = text.splitlines()
    heads = [i for i, line in enumerate(lines) if re.match(r"^## +\[?v?\d+\.\d+", line)]
    out = []
    for k, i in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else len(lines)
        block = lines[i:min(stop, i + max_lines)]
        while block and not block[-1].strip():
            block.pop()
        if len(block) >= 3:
            out.append(("\n".join(block) + "\n", i + 1, i + len(block)))
    return out


def readme_excerpts(text: str) -> list[tuple[str, str, int, int]]:
    """Tables (consecutive | lines, 3 or more) and prose paragraphs of 2 to 6 lines."""

    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            if j - i >= 3:
                out.append(("label", "\n".join(lines[i:j]) + "\n", i + 1, j))
            i = j
            continue
        if lines[i].strip() and not lines[i].startswith(("#", "```", "<", "-", "*", "|", "[", "!")):
            j = i
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "```", "|")):
                j += 1
            if 2 <= j - i <= 6:
                out.append(("design_note", "\n".join(lines[i:j]) + "\n", i + 1, j))
            i = j
            continue
        i += 1
    return out


def code_excerpts(text: str) -> list[tuple[str, int, int]]:
    """Top-level functions of 4 to 12 lines, in Python, TypeScript, JavaScript or Rust."""

    lines = text.splitlines()
    out = []
    starts = [i for i, line in enumerate(lines) if re.match(r"^(def |export function |function |pub fn |fn |export const \w+ = \()", line)]
    for i in starts:
        j = i + 1
        while j < len(lines) and (lines[j].startswith((" ", "\t", "}")) or not lines[j].strip()):
            j += 1
            if lines[j - 1].startswith("}"):
                break
        block = lines[i:j]
        while block and not block[-1].strip():
            block.pop()
        if 4 <= len(block) <= 12:
            out.append(("\n".join(block) + "\n", i + 1, i + len(block)))
    return out


def harvest_repo(repo: Path, commit: str, *, name: str | None = None, max_per_kind: int = 12) -> tuple[list[Excerpt], int]:
    name = name or repo.name
    if name in EXCLUDED_REPOS:
        raise HarvestError(f"{name} is on the exclusion list and never enters the pool")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise HarvestError(f"{name}: pin a commit sha, not a branch")
    licence = licence_line(repo, commit)
    files = _git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
    excerpts: list[Excerpt] = []
    dropped = 0
    counts: dict[str, int] = {}

    def add(kind: str, text: str, path: str, start: int, end: int) -> None:
        nonlocal dropped
        if counts.get(kind, 0) >= max_per_kind:
            return
        if PRIVATE_LINE.search(text):
            dropped += 1
            return
        if "MIT License" in text:
            dropped += 1
            return
        counts[kind] = counts.get(kind, 0) + 1
        excerpts.append(Excerpt(text, kind, {"repo": name, "commit": commit, "path": path, "lines": _lines_ref(start, end), "licence": licence}))

    for path in files:
        base = path.rsplit("/", 1)[-1].lower()
        if base == "changelog.md":
            for text, s, e in changelog_excerpts(_git(repo, "show", f"{commit}:{path}")):
                add("changelog", text, path, s, e)
        elif base == "readme.md" and "/" not in path:
            for kind, text, s, e in readme_excerpts(_git(repo, "show", f"{commit}:{path}")):
                add(kind, text, path, s, e)
        elif path.endswith(CODE_SUFFIXES) and ("/src/" in path or path.startswith("src/")) and "test" not in path.lower():
            if counts.get("code", 0) >= max_per_kind:
                continue
            for text, s, e in code_excerpts(_git(repo, "show", f"{commit}:{path}")):
                add("code", text, path, s, e)
    return excerpts, dropped


def harvest(allow_list: list[dict], clones_root: Path, out: Path) -> HarvestReport:
    """allow_list entries: {"name", "commit"}; the clone is clones_root/name. Writes out as JSONL."""

    report = HarvestReport()
    pool: list[Excerpt] = []
    for entry in allow_list:
        name, commit = entry["name"], entry["commit"]
        try:
            got, dropped = harvest_repo(clones_root / name, commit, name=name)
        except HarvestError as e:
            report.refused.append(str(e))
            continue
        report.repos += 1
        report.dropped_private += dropped
        pool.extend(got)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for x in pool:
            f.write(json.dumps(x.as_dict(), ensure_ascii=False) + "\n")
            report.by_kind[x.content_kind] = report.by_kind.get(x.content_kind, 0) + 1
    report.excerpts = len(pool)
    return report


def load_pool(path: Path) -> dict[str, list]:
    """The pool grouped by content kind, as the runner takes it."""

    from role_set.assemble import Material

    groups: dict[str, list] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        groups.setdefault(d["content_kind"], []).append(Material(d["text"], d["source"]))
    return groups
