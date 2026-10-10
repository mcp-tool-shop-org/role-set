import json
from pathlib import Path

import pytest

from role_set.cli import main
from role_set.pack import PackManifest, pack_hash
from role_set.write import WriteConfig, WriteRefused, build, load_plan, run, trait_names_from_pack

TRAIT_FILES = {
    "role.md": "# Lesson 1\n\nthe role\n",
    "traits/T02.md": "## 2. As a verifier, you go to the source\n\nThe source decides.\n",
    "traits/T03.md": "## 3. As a verifier, you are precise\n\nExact.\n",
    "traits/T14.md": "### 14. As a verifier, you don't fill gaps\n\nSilence stays silent.\n",
    "traits/T99.md": "no heading here\n",
    "kinds.md": "## 2. What to deliver\n\nfive kinds\n",
    "amendments/1.md": "## Amendment 1\n\nquote exactly\n",
}


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    d = tmp_path / "pack"
    for name, text in TRAIT_FILES.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    shas, whole = pack_hash(TRAIT_FILES)
    (d / "pack.json").write_text(json.dumps(PackManifest("verifier", "abc1234", shas, whole).as_dict()), encoding="utf-8")
    return d


@pytest.fixture
def plan_file(tmp_path: Path) -> Path:
    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(out)]) == 0
    return out


def test_mock_write_runs_a_whole_lesson_through_the_cli(pack_dir: Path, plan_file: Path, tmp_path: Path, capsys):
    rc = main(["write", "--plan", str(plan_file), "--pack", str(pack_dir), "--out", str(tmp_path / "o"), "--mock", "--concurrency", "2"])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and result["accepted"] == 40 and result["mock"] is True and result["out_dir"] == "mock"
    rows = [json.loads(l) for l in (tmp_path / "o" / "mock" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and rows[0]["trait"] == "goes to the source"
    receipt = json.loads((tmp_path / "o" / "mock" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert receipt["mock"] is True and receipt["pack_commit"] == "abc1234" and receipt["plan_sha256"]


def test_mock_write_without_the_critic(pack_dir: Path, plan_file: Path, tmp_path: Path):
    result = run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", mock=True, no_critic=True))
    assert result["accepted"] == 40 and result["halted"] is None


def test_plan_and_pack_helpers(pack_dir: Path, plan_file: Path):
    slots, sides, sha = load_plan(plan_file)
    assert len(slots) == 40 and sides["T02"] == ("quotes the source", "stops at the source") and len(sha) == 64
    from role_set.prompts import load_pack

    names = trait_names_from_pack(load_pack(pack_dir))
    assert names == {"T02": "goes to the source", "T03": "precise", "T14": "doesn't fill gaps", "T99": "T99"}


def test_real_write_refuses_each_missing_piece(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w

    writers = tmp_path / "writers.json"
    writers.write_text(json.dumps({"writers": [{"served_as": "qwen3-235b-instruct-2507", "repo": "Q/R", "revision": "9654", "licence": "Apache-2.0"}],
                                   "critics": [{"served_as": "gemma4:31b", "digest": "6316" + "0" * 60}]}), encoding="utf-8")
    base = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, critic_endpoint="http://127.0.0.1:11434", critic_model="gemma4:31b")

    class _Resp:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    opener = lambda url, timeout: _Resp({"models": [{"name": "gemma4:31b", "digest": "6316" + "0" * 60}]})  # noqa: E731

    def fake_gates():
        from role_set.mock import mock_gates

        return mock_gates()

    monkeypatch.setattr(w, "load_gates", fake_gates)
    with pytest.raises(WriteRefused, match="127.0.0.1"):
        build(WriteConfig(endpoint="http://pod.example:8000/v1", model="qwen3-235b-instruct-2507", **base))
    with pytest.raises(WriteRefused, match="licence-checked"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="other-model", **base))
    with pytest.raises(WriteRefused, match="cloud"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3:235b-cloud", **base))
    with pytest.raises(WriteRefused, match="--grant"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", **base))
    with pytest.raises(WriteRefused, match="--heartbeat"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", **base))
    hb = tmp_path / "hb"
    hb.write_text("x")
    with pytest.raises(WriteRefused, match="heartbeat is"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", heartbeat=hb, **base), now=lambda: hb.stat().st_mtime + 999)
    with pytest.raises(WriteRefused, match="local critic"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", heartbeat=hb, **{**base, "critic_model": "gemma4:31b-cloud"}), now=lambda: hb.stat().st_mtime + 1)
    with pytest.raises(WriteRefused, match="critic list"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", heartbeat=hb, **{**base, "critic_model": "mistral-small:24b"}), now=lambda: hb.stat().st_mtime + 1)
    with pytest.raises(WriteRefused, match="differs from the pinned"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", heartbeat=hb, **base), now=lambda: hb.stat().st_mtime + 1,
              digest_opener=lambda url, timeout: _Resp({"models": [{"name": "gemma4:31b", "digest": "ffff"}]}))
    slots, sides, pack, names, gates, writer, critic, pins, cleanup = build(
        WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", grant="G-1", heartbeat=hb, **base), now=lambda: hb.stat().st_mtime + 1, digest_opener=opener)
    assert pins["writer_revision"] == "9654" and pins["grant"] == "G-1" and pins["critic_model"] == "gemma4:31b" and pins["critic_digest"].startswith("6316") and callable(cleanup)
    no_critic = build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", no_critic=True, **base))
    assert no_critic[6] is None and no_critic[8] is None

    def missing():
        from role_set.gates import GatesUnavailable

        raise GatesUnavailable("REFUSED: pin it")

    monkeypatch.setattr(w, "load_gates", missing)
    with pytest.raises(WriteRefused, match="pin it"):
        build(WriteConfig(endpoint="http://127.0.0.1:11435/v1", model="qwen3-235b-instruct-2507", **base))


def test_write_halts_are_reported_and_cleanup_runs(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.runner import RunHalted

    cleaned = []

    class Halting:
        def __init__(self, **kw):
            from role_set.runner import Status

            self.status = Status(started_at=0.0)

        def run(self, slots, sides):
            raise RunHalted("andon: test")

    monkeypatch.setattr(w, "Runner", Halting)
    real_build = w.build

    def build_with_cleanup(cfg, now=None):
        parts = list(real_build(cfg))
        parts[8] = lambda: cleaned.append(True)
        return tuple(parts)

    monkeypatch.setattr(w, "build", build_with_cleanup)
    result = run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", mock=True))
    assert result["halted"] == "andon: test" and cleaned == [True]


def test_mock_write_draws_from_a_pool_and_refuses_bad_exemplars(pack_dir: Path, plan_file: Path, tmp_path: Path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({"text": "## [1.0.0] - 2026-01-01\n- Added the thing.\n", "content_kind": "changelog",
                                "source": {"repo": "x/y", "commit": "abc", "path": "CHANGELOG.md", "lines": "L1-L2", "licence": "MIT License"}}) + "\n", encoding="utf-8")
    result = run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", mock=True, pool=pool))
    rows = [json.loads(l) for l in (tmp_path / "o" / "mock" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["accepted"] == 40 and any(r["key_basis"] == "source" for r in rows)
    with pytest.raises(WriteRefused, match="cannot read|held out"):
        run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o2", mock=True, exemplar_repo=tmp_path, exemplar_commit="abc1234", exemplar_files=("nope.jsonl",)))
