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
    record_excerpts,
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
RECORD_TOML = """[project]
name = "good-tool"
version = "0.1.0"
description = "A small tool"
authors = [{name = "Dev", email = "dev@example.com"}]
requires-python = ">=3.12"
"""
RECORD_JSON = """{
  "name": "good-tool",
  "version": "0.1.0",
  "packageManager": "npm@10.2.0",
  "private": true,
  "type": "module"
}
"""
RECORD_YAML = """version: 2
updates:
  - package-ecosystem: npm
    directory: /
    schedule:
      interval: weekly
"""
RECORD_CSV = """code,title,tier,active
311,Food Manufacturing,3,true
312,Beverage and Tobacco,3,true
313,Textile Mills,4,false
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
        (repo / "pyproject.toml").write_text(RECORD_TOML, encoding="utf-8")
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

    # record_excerpts on TOML, JSON, YAML and CSV, dropping email / pkg@ver PRIVATE_LINE lines and skipping lockfiles / test paths.
    toml_rec = record_excerpts(RECORD_TOML, "pyproject.toml")
    assert len(toml_rec) == 1 and "dev@example.com" not in toml_rec[0][0]
    assert toml_rec[0][1] == 1 and toml_rec[0][2] == 6 and 4 <= len(toml_rec[0][0].splitlines()) <= 12

    json_rec = record_excerpts(RECORD_JSON, "package.json")
    assert len(json_rec) == 1 and "npm@10.2.0" not in json_rec[0][0]
    assert '"name": "good-tool"' in json_rec[0][0]

    yaml_rec = record_excerpts(RECORD_YAML, ".github/dependabot.yml")
    assert len(yaml_rec) == 1 and "package-ecosystem: npm" in yaml_rec[0][0]

    csv_rec = record_excerpts(RECORD_CSV, "data/sectors.csv")
    assert len(csv_rec) == 1 and "Food Manufacturing" in csv_rec[0][0]

    for skipped_path in ("uv.lock", "package-lock.json", "Cargo.lock", "pnpm-lock.yaml", "yarn.lock",
                         "node_modules/pkg/package.json", "tests/fixtures/data.json", "src/mock/config.toml"):
        assert record_excerpts(RECORD_JSON, skipped_path) == []


def test_harvest_repo_checks_licence_exclusions_and_private_lines(clones):
    root, shas = clones
    got, dropped = harvest_repo(root / "good-tool", shas["good-tool"])
    kinds = {x.content_kind for x in got}
    assert kinds == {"changelog", "label", "design_note", "code", "record"}
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
    assert d["repos"] == 1 and d["excerpts"] >= 5 and len(d["refused"]) == 2 and d["by_kind"]["changelog"] == 2 and d["by_kind"]["record"] == 1
    assert (tmp_path / "pool.jsonl.sha256").is_file()
    pool = load_pool(out)
    assert set(pool) == {"changelog", "label", "design_note", "code", "record"} and pool["changelog"][0].source["path"] == "CHANGELOG.md"
    assert not pool["code"][0].invented and not pool["record"][0].invented


def test_committed_pool_lines_sources_and_seal():
    import hashlib

    repo_root = Path(__file__).resolve().parents[1]
    pool_path = repo_root / "roles" / "verifier" / "pool.jsonl"
    seal_path = repo_root / "roles" / "verifier" / "pool.jsonl.sha256"
    sources_path = repo_root / "roles" / "verifier" / "sources.json"
    assert pool_path.is_file() and seal_path.is_file()
    got_sha = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    want_sha = seal_path.read_text(encoding="utf-8").split()[0].strip().lower()
    assert got_sha == want_sha

    allowed = {
        r["name"]: (r["commit"], r["licence"])
        for r in json.loads(sources_path.read_text(encoding="utf-8"))["repos"]
    }
    raw_lines = [ln for ln in pool_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(raw_lines) >= 180
    for idx, line in enumerate(raw_lines, 1):
        d = json.loads(line)
        text = d["text"]
        kind = d["content_kind"]
        src = d["source"]
        assert PRIVATE_LINE.search(text) is None, (idx, src)
        assert src["repo"] in allowed, (idx, src)
        assert src["commit"] == allowed[src["repo"]][0], (idx, src)
        assert src["licence"] == allowed[src["repo"]][1] == "MIT License", (idx, src)
        assert ":" not in src["path"] and not src["path"].startswith("/"), (idx, src)
        n_lines = len(text.splitlines())
        if kind in ("record", "code"):
            assert 4 <= n_lines <= 12 and 4 <= len(text.split("\n")) <= 13, (idx, kind, n_lines, src)
        elif kind == "changelog":
            assert 3 <= n_lines <= 12, (idx, kind, n_lines, src)
        elif kind == "design_note":
            assert 2 <= n_lines <= 6, (idx, kind, n_lines, src)
        elif kind == "label":
            assert n_lines >= 3, (idx, kind, n_lines, src)
        else:
            raise AssertionError(f"unexpected content_kind {kind!r} at line {idx}")


def test_committed_pool_covers_real_source_kinds_and_plans():
    from role_set.cli import VERIFIER_SIDES
    from role_set.plan import REAL_SOURCE_KINDS, plan_lesson

    repo_root = Path(__file__).resolve().parents[1]
    pool = load_pool(repo_root / "roles" / "verifier" / "pool.jsonl")
    for kind in REAL_SOURCE_KINDS:
        assert len(pool.get(kind) or ()) >= 6, (kind, len(pool.get(kind) or ()))

    # Every verifier lesson (and buffered T10) has enough excerpts per real_source kind in the committed pool.
    for lesson, sides in VERIFIER_SIDES.items():
        for buffers in ((False, True) if lesson == "T10" else (False,)):
            slots = plan_lesson(lesson, sides, seed=0, buffers=buffers)
            need: dict[str, int] = {}
            for s in slots:
                if s.real_source:
                    need[s.content_kind] = need.get(s.content_kind, 0) + 1
            for kind, req in need.items():
                assert len(pool.get(kind) or ()) >= req, (lesson, buffers, kind, req)


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
