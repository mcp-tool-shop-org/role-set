import json
import subprocess
from pathlib import Path

import pytest

from role_set.pack import (
    VERIFIER_SOURCES,
    PackError,
    build_files,
    git_show,
    load_manifest,
    split_traits,
    sync,
    verify,
)

TRAITS = "# As a verifier, these are your traits\n\nintro\n\n" + "".join(
    (f"## {i}. As a verifier, you are trait {i}\n\nbody {i}\n\n" if i <= 12 else "")
    for i in range(1, 13)
) + "## What a verifier doesn't do\n\n" + "".join(
    f"### {i}. As a verifier, you don't do thing {i}\n\nbody {i}\n\n" for i in range(13, 19)
)
QUALITIES = "# q\n\n## The method\n\nm\n\n## Thinking patterns\n\ntrees and grids\n\n## Lessons\n\nl\n"
SPEC = (
    "# spec\n\n## 1. What\n\nw\n\n## 2. What to deliver\n\nfive kinds\n\n## 3. Rules\n\n"
    "1. **Instant agreement.** cut it.\n7. **Stay clear of the test sets.** Don't draw from:\n   - repos\n"
    "8. **The pre-interview is off-limits.**\n\n## 4. How to work\n\nh\n"
)
HANDOFF = (
    "# handoff\n\n## 4. The rules\n\nr\n\n## 5. How to work\n\nx\n\n"
    "## Amendment 1 (date): quoting\n\na1\n\n## Amendment 2 (date): habits\n\na2\n\n## Amendment 3: neutral\n\na3\n"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "aspire"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "core.longpaths", "true")  # Windows: fixture paths under a deep temp dir
    texts = {
        "lesson1": "# Lesson 1\n\nthe role\n", "lesson2": "# Lesson 2\n\nhow it thinks\n", "lesson3": "# Lesson 3\n\nexpected\n",
        "traits": TRAITS, "qualities": QUALITIES, "spec": SPEC, "handoff": HANDOFF, "schema": "# schema\n\nfields\n",
    }
    for key, rel in VERIFIER_SOURCES.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texts[key], encoding="utf-8")
    (repo / "docs/runs/2026-10-10-stage1-pre-interview-draft.md").write_text("never\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_sync_builds_the_pack_and_a_stable_hash(source_repo, tmp_path: Path):
    repo, sha = source_repo
    out = tmp_path / "pack"
    m1 = sync(repo, sha, out)
    assert (out / "traits" / "T18.md").read_text(encoding="utf-8").startswith("### 18. As a verifier")
    assert (out / "amendments" / "2.md").read_text(encoding="utf-8").startswith("## Amendment 2")
    assert "five kinds" in (out / "kinds.md").read_text(encoding="utf-8")
    assert "Stay clear" in (out / "exclusions.md").read_text(encoding="utf-8")
    assert "pre-interview" not in "".join(p.name for p in out.rglob("*"))
    manifest = json.loads((out / "pack.json").read_text(encoding="utf-8"))
    assert manifest["commit"] == sha and len(manifest["files"]) == 18 + 8
    assert sync(repo, sha, tmp_path / "pack2").pack_sha256 == m1.pack_sha256
    assert verify(out).pack_sha256 == m1.pack_sha256
    assert load_manifest(out).role == "verifier"


def test_verify_refuses_a_drifted_pack(source_repo, tmp_path: Path):
    repo, sha = source_repo
    out = tmp_path / "pack"
    sync(repo, sha, out)
    (out / "rules.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(PackError, match="drifted.*rules.md"):
        verify(out)
    with pytest.raises(PackError, match="pack sync first"):
        load_manifest(tmp_path / "nowhere")


def test_sync_refuses_branches_roles_missing_files_and_held_out_paths(source_repo, tmp_path: Path):
    repo, sha = source_repo
    with pytest.raises(PackError, match="commit sha"):
        sync(repo, "main", tmp_path / "p")
    with pytest.raises(PackError, match="no sources"):
        sync(repo, sha, tmp_path / "p", role="poet")
    with pytest.raises(PackError, match="cannot read"):
        git_show(repo, sha, "docs/runs/missing.md")
    for bad in ("docs/runs/2026-10-10-stage1-pre-interview-draft.md", "outputs/stage1-sealed-FROZEN/x", "docs/runs/stage1-taskset/a.jsonl"):
        with pytest.raises(PackError, match="held-out"):
            git_show(repo, sha, bad)
    assert not (tmp_path / "p").exists()


def test_split_and_build_refuse_malformed_sources():
    with pytest.raises(PackError, match="T01..T18"):
        split_traits("## 1. As a verifier, you are x\n\nbody\n")
    good = {"lesson1": "a", "lesson2": "b", "lesson3": "c", "traits": TRAITS, "qualities": QUALITIES, "spec": SPEC, "handoff": HANDOFF, "schema": "s"}
    assert "role.md" in build_files(good)
    with pytest.raises(PackError, match="not found"):
        build_files({**good, "qualities": "# nothing here\n"})
    with pytest.raises(PackError, match="no amendments"):
        build_files({**good, "handoff": "## 4. The rules\n\nr\n\n## 5. x\n\ny\n"})
    with pytest.raises(PackError, match="came out empty"):
        build_files({**good, "schema": "   \n"})
