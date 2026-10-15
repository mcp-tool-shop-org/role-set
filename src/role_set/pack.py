"""The role pack: the text the humans wrote.

The verifier pack is fetched from aspire-si at a pinned commit. role-set adds nothing
to it. `sync` reads each source file at one commit with `git show`, splits the trait
lessons into one file per trait, and writes a manifest with a sha256 per file and one
hash for the whole pack, which every item records. The pre-interview is never read.

The advocate pack is the exception. A person committed it in the repo. `sync` refuses
that role, and `verify` checks the four files against the manifest already beside them.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

RUNS = "docs/runs"
FORBIDDEN = ("pre-interview", "sealed", "taskset")

# What the verifier pack is made of, and where each part comes from.
VERIFIER_SOURCES = {
    "lesson1": f"{RUNS}/2026-10-10-stage1-lesson-1-the-verifier-role-draft.md",
    "lesson2": f"{RUNS}/2026-10-10-stage1-lesson-2-how-a-verifier-thinks-draft.md",
    "lesson3": f"{RUNS}/2026-10-10-stage1-lesson-3-what-is-expected-of-a-verifier-draft.md",
    "traits": f"{RUNS}/2026-10-10-stage1-verifier-trait-lessons-draft.md",
    "qualities": f"{RUNS}/2026-10-10-stage1-verifier-qualities-lessons-draft.md",
    "spec": f"{RUNS}/2026-10-10-stage1-training-data-kimi-handoff.md",
    "handoff": f"{RUNS}/2026-10-10-stage1-training-data-grok-handoff.md",
    "schema": f"{RUNS}/stage1-training-data-schema.md",
}

TRAIT_HEADING = re.compile(r"^#{2,3} (\d{1,2})\. As a verifier, you (.+)$", re.M)
SECTION = re.compile(r"^## ", re.M)


class PackError(RuntimeError):
    """The pack could not be built as asked. Nothing partial is written."""


@dataclass(frozen=True)
class PackManifest:
    role: str
    commit: str
    files: dict[str, str]
    pack_sha256: str

    def as_dict(self) -> dict:
        return {"role": self.role, "commit": self.commit, "files": dict(sorted(self.files.items())), "pack_sha256": self.pack_sha256}


def git_show(repo: Path, commit: str, path: str) -> str:
    """One file at one commit. The repo path never enters the pack."""

    for marker in FORBIDDEN:
        if marker in path.lower():
            raise PackError(f"REFUSED: {path} is a held-out file and never enters the pack")
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise PackError(f"REFUSED: cannot read {path} at {commit[:12]}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def section(text: str, start: str, end: str | None = None) -> str:
    """The text from the heading that starts with `start` up to the next heading that starts with `end`."""

    i = text.find(start)
    if i < 0:
        raise PackError(f"REFUSED: section {start!r} not found")
    j = text.find(end, i + len(start)) if end else -1
    return text[i:] if j < 0 else text[i:j]


def split_traits(text: str) -> dict[str, str]:
    """T01..T18, one lesson each, split on their headings. Eighteen are required."""

    heads = list(TRAIT_HEADING.finditer(text))
    out: dict[str, str] = {}
    for k, m in enumerate(heads):
        n = int(m.group(1))
        stop = heads[k + 1].start() if k + 1 < len(heads) else len(text)
        body = text[m.start():stop].rstrip() + "\n"
        out[f"T{n:02d}"] = body
    if sorted(out) != [f"T{i:02d}" for i in range(1, 19)]:
        raise PackError(f"REFUSED: expected traits T01..T18, found {sorted(out)}")
    return out


def amendments(handoff: str) -> dict[str, str]:
    heads = [m.start() for m in re.finditer(r"^## Amendment (\d+)", handoff, re.M)]
    if not heads:
        raise PackError("REFUSED: no amendments found in the handoff")
    out = {}
    for k, start in enumerate(heads):
        stop = heads[k + 1] if k + 1 < len(heads) else len(handoff)
        num = re.match(r"^## Amendment (\d+)", handoff[start:], re.M).group(1)  # type: ignore[union-attr]
        out[f"amendments/{num}.md"] = handoff[start:stop].rstrip() + "\n"
    return out


def build_files(sources: dict[str, str]) -> dict[str, str]:
    """From the raw source texts to the pack's files."""

    files: dict[str, str] = {}
    files["role.md"] = "\n\n".join(sources[k].rstrip() for k in ("lesson1", "lesson2", "lesson3")) + "\n"
    for trait, body in split_traits(sources["traits"]).items():
        files[f"traits/{trait}.md"] = body
    files["patterns.md"] = section(sources["qualities"], "## Thinking patterns", "## ")
    spec = sources["spec"]
    files["kinds.md"] = section(spec, "## 2. What to deliver", "## 3.")
    files["rules.md"] = section(spec, "## 3. Rules", "## 4.") + "\n" + section(sources["handoff"], "## 4. The rules", "## 5.")
    files["exclusions.md"] = section(spec, "7. **Stay clear of the test sets.**", "8. **")
    files.update(amendments(sources["handoff"]))
    files["schema.md"] = sources["schema"]
    for name, text in files.items():
        if "pre-interview" in name.lower():  # pragma: no cover - defensive; names are fixed above
            raise PackError("REFUSED: the pre-interview never enters the pack")
        if not text.strip():
            raise PackError(f"REFUSED: {name} came out empty")
    return files


def pack_hash(files: dict[str, str]) -> tuple[dict[str, str], str]:
    shas = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in files.items()}
    whole = hashlib.sha256()
    for name in sorted(shas):
        whole.update(f"{name}\n{shas[name]}\n".encode("utf-8"))
    return shas, whole.hexdigest()


def sync(repo: Path, commit: str, out: Path, *, role: str = "verifier") -> PackManifest:
    """Fetch, split, write, and record. Writes nothing until every file is built."""

    if role == "advocate":
        raise PackError("REFUSED: no sources known for role 'advocate'; the advocate pack is in-repo")
    if role != "verifier":
        raise PackError(f"REFUSED: no sources known for role {role!r}")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise PackError("REFUSED: pin the pack to a commit sha, not a branch")
    raw = {key: git_show(repo, commit, path) for key, path in VERIFIER_SOURCES.items()}
    files = build_files(raw)
    shas, whole = pack_hash(files)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    manifest = PackManifest(role=role, commit=commit, files=shas, pack_sha256=whole)
    (out / "pack.json").write_text(json.dumps(manifest.as_dict(), indent=1) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_manifest(pack_dir: Path) -> PackManifest:
    try:
        d = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    except OSError as e:
        raise PackError(f"REFUSED: no pack.json in {pack_dir.name}; run pack sync first") from e
    return PackManifest(role=d["role"], commit=d["commit"], files=d["files"], pack_sha256=d["pack_sha256"])


def verify(pack_dir: Path) -> PackManifest:
    """The files on disk must still hash to the manifest; a drifted pack is refused."""

    manifest = load_manifest(pack_dir)
    files = {name: (pack_dir / name).read_text(encoding="utf-8") for name in manifest.files}
    shas, whole = pack_hash(files)
    if whole != manifest.pack_sha256:
        changed = sorted(n for n in shas if shas[n] != manifest.files.get(n))
        raise PackError(f"REFUSED: the pack has drifted from its manifest: {changed}")
    return manifest
