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

    # record_excerpts on TOML, JSON, YAML, CSV and docs/**/*.md tables:
    # - PRIVATE_LINE ends the current run like a blank line, keeping blocks contiguous
    # - npm@10.2.0 survives while dev@example.com / someone@example.com drops the line
    toml_rec = record_excerpts(RECORD_TOML, "pyproject.toml")
    assert len(toml_rec) == 1 and "dev@example.com" not in toml_rec[0][0]
    assert toml_rec[0][1] == 1 and toml_rec[0][2] == 4
    assert toml_rec[0][0].splitlines() == RECORD_TOML.splitlines()[:4]

    json_rec = record_excerpts(RECORD_JSON, "package.json")
    assert len(json_rec) == 1 and "npm@10.2.0" in json_rec[0][0]
    assert json_rec[0][1] == 1 and json_rec[0][2] == 7
    assert len(json_rec[0][0].splitlines()) == 7
    assert PRIVATE_LINE.search("npm@10.2.0") is None
    assert PRIVATE_LINE.search("someone@example.com") is not None

    yaml_rec = record_excerpts(RECORD_YAML, "config/updates.yml")
    assert len(yaml_rec) == 1 and "package-ecosystem: npm" in yaml_rec[0][0]

    csv_rec = record_excerpts(RECORD_CSV, "data/sectors.csv")
    assert len(csv_rec) == 1 and "Food Manufacturing" in csv_rec[0][0]

    md_table = (
        "# Reference\n\n"
        "| Code | Name | Active |\n"
        "|---|---|---|\n"
        "| A1 | Alpha | true |\n"
        "| B2 | Beta | false |\n"
        "| C3 | dev@example.com | true |\n"
        "| D4 | Delta | true |\n"
    )
    md_rec = record_excerpts(md_table, "docs/guide/tables.md")
    assert len(md_rec) == 1
    assert md_rec[0][1] == 3 and md_rec[0][2] == 6
    assert len(md_rec[0][0].splitlines()) == 4 and "dev@example.com" not in md_rec[0][0]
    assert record_excerpts(md_table, "README.md") == []

    # At most 2 excerpts per file, even when 3+ 12-line blocks exist.
    long_toml = "\n".join(f'key_{i} = "val_{i}"' for i in range(1, 37)) + "\n"
    long_rec = record_excerpts(long_toml, "config/many.toml")
    assert len(long_rec) == 2
    assert (long_rec[0][1], long_rec[0][2]) == (1, 12)
    assert (long_rec[1][1], long_rec[1][2]) == (13, 24)

    for skipped_path in (
        "uv.lock",
        "package-lock.json",
        "Cargo.lock",
        "pnpm-lock.yaml",
        "yarn.lock",
        ".github/dependabot.yml",
        "sub/.github/workflows/ci.yml",
        "atlas/index.json",
        "pkg/atlas/manifest.json",
        ".vscode/settings.json",
        "pkg/.vscode/launch.json",
        ".devcontainer/devcontainer.json",
        "sub/.devcontainer/devcontainer.json",
        "node_modules/pkg/package.json",
        "sub/node_modules/pkg/package.json",
        "tsconfig.json",
        "tsconfig.base.json",
        "packages/web/tsconfig.build.json",
        "biome.json",
        "sub/biome.json",
        "renovate.json",
        ".eslintrc",
        ".eslintrc.json",
        "sub/.eslintrc.yml",
        "vitest.config.ts",
        "vitest.config.json",
        "jest.config.js",
        "jest.config.json",
        "tests/fixtures/data.json",
        "src/mock/config.toml",
    ):
        assert record_excerpts(RECORD_JSON, skipped_path) == [], skipped_path


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


def test_harvest_record_tiers_md_tables_cap_and_dedup(tmp_path: Path):
    import hashlib

    from role_set.harvest import _record_tier

    assert _record_tier("data/sectors.csv") == 1
    assert _record_tier("docs/spec/table.md") == 1
    assert _record_tier("schemas/tool-registry.json") == 1
    assert _record_tier("config/catalog.yaml") == 1
    assert _record_tier("pyproject.toml") == 2
    assert _record_tier("Cargo.toml") == 2
    assert _record_tier("site/package.json") == 2
    assert _record_tier("config/settings.toml") == 3
    assert _record_tier("schemas/bridge.schema.json") == 3

    root = tmp_path / "clones"
    shas = {}
    shared_pkg = '{\n  "name": "shared-site",\n  "version": "1.0.0",\n  "private": true,\n  "type": "module"\n}\n'
    for repo_name in ("repo-a", "repo-b"):
        repo = root / repo_name
        (repo / "docs").mkdir(parents=True)
        (repo / "config").mkdir(parents=True)
        (repo / "site").mkdir(parents=True)
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "t@example.com")
        _git(repo, "config", "user.name", "t")
        _git(repo, "config", "core.longpaths", "true")
        (repo / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026 x\n", encoding="utf-8")
        # Tier 3 file (sorted early by ls-tree, but must be visited after tier 1 and tier 2)
        (repo / "config" / "aaa.toml").write_text(
            f'repo = "{repo_name}"\nmode = "strict"\nretries = 3\ntimeout = 30\n',
            encoding="utf-8",
        )
        # Tier 2 file with identical text across both repos (second repo's copy must be dropped by harvest)
        (repo / "site" / "package.json").write_text(shared_pkg, encoding="utf-8")
        # Tier 1 file: Markdown table in docs/**/*.md with 3 tables (capped at 2 per file)
        tables = []
        for t_idx in range(3):
            tables.append(
                f"| Col | {repo_name}_{t_idx} |\n|---|---|\n| r1 | v{t_idx}a |\n| r2 | v{t_idx}b |\n"
            )
        (repo / "docs" / "zzz.md").write_text("# Spec\n\n" + "\n".join(tables), encoding="utf-8")
        # Tier 1 file: CSV
        (repo / "sectors.csv").write_text(
            f"id,name,repo\n1,alpha,{repo_name}\n2,beta,{repo_name}\n3,gamma,{repo_name}\n",
            encoding="utf-8",
        )
        # Skipped paths inside the repo
        (repo / "tsconfig.json").write_text(RECORD_JSON, encoding="utf-8")
        (repo / ".github").mkdir()
        (repo / ".github" / "ci.yml").write_text(RECORD_YAML, encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "fixture")
        shas[repo_name] = _git(repo, "rev-parse", "HEAD")

    got_a, _ = harvest_repo(root / "repo-a", shas["repo-a"], name="repo-a")
    recs_a = [x for x in got_a if x.content_kind == "record"]
    # Tier order honoured: all tier 1 files before tier 2 before tier 3
    tiers_a = [_record_tier(x.source["path"]) for x in recs_a]
    assert tiers_a == sorted(tiers_a) and tiers_a[0] == 1 and tiers_a[-1] == 3
    # Within tier 1, ordered by sha256(f"{repo_name}:{path}")
    t1_paths = list(dict.fromkeys(x.source["path"] for x in recs_a if _record_tier(x.source["path"]) == 1))
    assert t1_paths == sorted(t1_paths, key=lambda p: hashlib.sha256(f"repo-a:{p}".encode()).hexdigest())
    # Markdown table in docs/zzz.md became record, capped at 2 excerpts per file
    md_recs = [x for x in recs_a if x.source["path"] == "docs/zzz.md"]
    assert len(md_recs) == 2

    # Across the pool, exact duplicate text (site/package.json in repo-a and repo-b) is kept once and dropped on the second repo.
    out = tmp_path / "dedup-pool.jsonl"
    report = harvest(
        [{"name": "repo-a", "commit": shas["repo-a"]}, {"name": "repo-b", "commit": shas["repo-b"]}],
        root,
        out,
    )
    rows = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    texts = [r["text"] for r in rows]
    assert len(texts) == len(set(texts)) == report.excerpts
    pkg_rows = [r for r in rows if r["source"]["path"] == "site/package.json"]
    assert len(pkg_rows) == 1 and pkg_rows[0]["source"]["repo"] == "repo-a"


def test_committed_pool_lines_sources_and_seal():
    import hashlib

    from role_set.harvest import _record_tier

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
    assert len(raw_lines) == 218
    seen_texts: set[str] = set()
    rec_by_file: dict[tuple[str, str], int] = {}
    rec_tiers: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for idx, line in enumerate(raw_lines, 1):
        d = json.loads(line)
        text = d["text"]
        kind = d["content_kind"]
        src = d["source"]
        assert text not in seen_texts, (idx, src)
        seen_texts.add(text)
        assert PRIVATE_LINE.search(text) is None, (idx, src)
        assert src["repo"] in allowed, (idx, src)
        assert src["commit"] == allowed[src["repo"]][0], (idx, src)
        assert src["licence"] == allowed[src["repo"]][1] == "MIT License", (idx, src)
        assert ":" not in src["path"] and not src["path"].startswith("/"), (idx, src)
        n_lines = len(text.splitlines())
        if kind in ("record", "code"):
            assert 4 <= n_lines <= 12 and 4 <= len(text.split("\n")) <= 13, (idx, kind, n_lines, src)
            if kind == "record":
                s_ln, e_ln = [int(x[1:]) for x in src["lines"].split("-")]
                assert e_ln - s_ln + 1 == n_lines, (idx, src, n_lines)
                key = (src["repo"], src["path"])
                rec_by_file[key] = rec_by_file.get(key, 0) + 1
                assert rec_by_file[key] <= 2, (idx, key)
                rec_tiers[_record_tier(src["path"])] += 1
        elif kind == "changelog":
            assert 3 <= n_lines <= 12, (idx, kind, n_lines, src)
        elif kind == "design_note":
            assert 2 <= n_lines <= 6, (idx, kind, n_lines, src)
        elif kind == "label":
            assert n_lines >= 3, (idx, kind, n_lines, src)
        else:
            raise AssertionError(f"unexpected content_kind {kind!r} at line {idx}")
    assert sum(rec_tiers.values()) == 47 and rec_tiers[1] >= 12 and rec_tiers[2] >= 12


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
