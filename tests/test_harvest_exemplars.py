import json
import subprocess
from pathlib import Path

import pytest

from role_set.exemplars import ExemplarError, load_exemplars, pick, spans_of
from role_set.harvest import (
    EXCLUDED_REPOS,
    PRIVATE_LINE,
    HarvestError,
    changelog_excerpts,
    code_excerpts,
    harvest,
    harvest_repo,
    load_pool,
    readme_excerpts,
)

CHANGELOG = """# Changelog

## [1.2.0] - 2026-02-01

### Added
- The `--dry-run` flag no longer writes the lock file.
- A `--quiet` flag.

## [1.1.0] - 2026-01-01

### Fixed
- Nothing of note.
"""
README = """# tool

A small tool that does one thing well. It reads a file and prints a summary.
It never writes anywhere but stdout.

| Flag | Meaning |
|---|---|
| `--json` | machine output |
| `--quiet` | no banner |

Contact: someone@example.com for questions.
This line and the next are a paragraph.
"""
CODE = """import os


def color_enabled(env):
    if env.get("FORCE_COLOR") and env["FORCE_COLOR"] != "0":
        return True
    if env.get("NO_COLOR"):
        return False
    return False


def tiny():
    return 1
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clones(tmp_path: Path):
    root = tmp_path / "clones"
    shas = {}
    for name, licence in (("good-tool", "MIT License\n\nCopyright (c) 2026 someone\n"), ("gpl-tool", "GNU GENERAL PUBLIC LICENSE\n")):
        repo = root / name
        (repo / "src").mkdir(parents=True)
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        _git(repo, "config", "core.longpaths", "true")
        (repo / "LICENSE").write_text(licence, encoding="utf-8")
        (repo / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
        (repo / "README.md").write_text(README, encoding="utf-8")
        (repo / "src" / "color.py").write_text(CODE, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "fixture")
        shas[name] = _git(repo, "rev-parse", "HEAD")
    return root, shas


def test_cutters():
    cl = changelog_excerpts(CHANGELOG)
    assert len(cl) == 2 and cl[0][0].startswith("## [1.2.0]") and "--dry-run" in cl[0][0] and cl[0][1] == 3
    rd = readme_excerpts(README)
    kinds = [k for k, *_ in rd]
    assert "label" in kinds and "design_note" in kinds
    table = next(t for k, t, *_ in rd if k == "label")
    assert table.count("\n") == 4
    code = code_excerpts(CODE)
    assert len(code) == 1 and code[0][0].startswith("def color_enabled") and code[0][1] == 4


def test_harvest_repo_checks_licence_exclusions_and_private_lines(clones):
    root, shas = clones
    got, dropped = harvest_repo(root / "good-tool", shas["good-tool"])
    kinds = {x.content_kind for x in got}
    assert kinds == {"changelog", "label", "design_note", "code"}
    assert all(x.source["repo"] == "good-tool" and x.source["commit"] == shas["good-tool"] and x.source["licence"] == "MIT License" for x in got)
    assert dropped == 1 and all(PRIVATE_LINE.search(x.text) is None for x in got)
    with pytest.raises(HarvestError, match="not on the allowed list"):
        harvest_repo(root / "gpl-tool", shas["gpl-tool"])
    with pytest.raises(HarvestError, match="exclusion list"):
        harvest_repo(root / "good-tool", shas["good-tool"], name="offrig")
    with pytest.raises(HarvestError, match="commit sha"):
        harvest_repo(root / "good-tool", "main")
    with pytest.raises(HarvestError, match="no LICENSE"):
        harvest_repo(root / "good-tool", "0" * 40)
    assert "aspire-si" in EXCLUDED_REPOS and "rnd" in EXCLUDED_REPOS


def test_harvest_writes_a_pool_the_runner_can_load(clones, tmp_path: Path):
    root, shas = clones
    out = tmp_path / "pool.jsonl"
    report = harvest([{"name": "good-tool", "commit": shas["good-tool"]}, {"name": "gpl-tool", "commit": shas["gpl-tool"]}, {"name": "missing", "commit": "a" * 40}], root, out)
    d = report.as_dict()
    assert d["repos"] == 1 and d["excerpts"] >= 4 and len(d["refused"]) == 2 and d["by_kind"]["changelog"] == 2
    pool = load_pool(out)
    assert set(pool) == {"changelog", "label", "design_note", "code"} and pool["changelog"][0].source["path"] == "CHANGELOG.md"
    assert not pool["code"][0].invented


def test_exemplars_are_read_pinned_and_picked_from_other_traits(tmp_path: Path):
    repo = tmp_path / "aspire"
    (repo / "docs/runs/stage1-training-data").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.longpaths", "true")
    rows = [{"id": f"T0{t}-{k}-01", "lesson": f"T0{t}", "kind": k, "turns": [{"user": "u", "thinking": f"think {t} {k} " * 5, "reply": f"reply {t} {k}"}]} for t in (4, 1) for k in "ABCDE"]
    (repo / "docs/runs/stage1-training-data/pilot-t04.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["lesson"] == "T04") + "\n", encoding="utf-8")
    (repo / "docs/runs/stage1-training-data/batch2-t01.jsonl").write_text("\n".join(json.dumps(r) for r in rows if r["lesson"] == "T01") + "\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    sha = _git(repo, "rev-parse", "HEAD")
    by_lesson = load_exemplars(repo, sha, ["pilot-t04.jsonl", "batch2-t01.jsonl"])
    assert set(by_lesson) == {"T04", "T01"}
    picked = pick(by_lesson, kind="D", exclude_lesson="T04", n=3)
    assert len(picked) == 1 and "id" not in picked[0] and picked[0]["kind"] == "D"
    assert spans_of(picked)
    with pytest.raises(ExemplarError, match="held out"):
        load_exemplars(repo, sha, ["../stage1-sealed-FROZEN/x.jsonl"])
    with pytest.raises(ExemplarError, match="cannot read"):
        load_exemplars(repo, sha, ["batch9-t99.jsonl"])
