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
def plan_file(tmp_path: Path, domain_table: Path) -> Path:
    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02", "--seed", "3", "--out", str(out), "--domains", str(domain_table)]) == 0
    return out


def test_mock_write_runs_a_whole_lesson_through_the_cli(pack_dir: Path, plan_file: Path, tmp_path: Path, capsys):
    rc = main(["write", "--plan", str(plan_file), "--pack", str(pack_dir), "--out", str(tmp_path / "o"), "--mock", "--concurrency", "2"])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0 and result["accepted"] == 40 and result["mock"] is True and result["out_dir"] == "mock"
    rows = [json.loads(l) for l in (tmp_path / "o" / "mock" / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and rows[0]["trait"] == "goes to the source"
    receipt = json.loads((tmp_path / "o" / "mock" / "receipts.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert receipt["mock"] is True and receipt["pack_commit"] == "abc1234" and receipt["plan_sha256"]
    # The pinned aspire-si-gates passed every item and the finished lesson: the real gates, not the stand-in.
    assert receipt["gates"]["version"] == "1.7.4" and result["batch_check"] == "clean"
    # The closed-form settings the run used, resolved by the pinned gates: their defaults, as no --closed-form was given.
    assert receipt["gates"]["closed_form"]["carry"] == "paraphrase"


def test_mock_write_without_the_critic(pack_dir: Path, plan_file: Path, tmp_path: Path):
    result = run(WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", mock=True, no_critic=True))
    assert result["accepted"] == 40 and result["halted"] is None and result["batch_check"] == "clean"


def test_plan_and_pack_helpers(pack_dir: Path, plan_file: Path):
    slots, sides, sha = load_plan(plan_file)
    assert len(slots) == 40 and sides["T02"] == ("quotes the source", "stops at the source") and len(sha) == 64
    from role_set.prompts import load_pack

    names = trait_names_from_pack(load_pack(pack_dir))
    assert names == {"T02": "goes to the source", "T03": "precise", "T14": "doesn't fill gaps", "T99": "T99"}


def _proven_setup(tmp_path: Path, *, card: str = "NVIDIA A100-SXM4-40GB") -> tuple[Path, Path]:
    from test_proven import proven_file

    machine = tmp_path / "machine.json"
    where = {"rig": "robot"} if card == "rig" or "5090" in card else {"colab_release_tag": "release-colab_20261015"}
    card = "NVIDIA A100-SXM4-40GB" if card == "rig" else card
    machine.write_text(json.dumps({"driver": "580.82", "compat": True, "runtime": "13.4", "card": card, **where}), encoding="utf-8")
    return proven_file(tmp_path), machine


def test_real_write_refuses_each_missing_piece(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.mock import mock_gates
    from test_proven import SHA_C, SHA_S, SHA_W

    writers, machine = _proven_setup(tmp_path)
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    good = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, endpoint="http://127.0.0.1:8080/v1", model="big-writer",
                writer_gguf_sha256=SHA_W, critic_endpoint="http://127.0.0.1:8081/v1", critic_model="other-critic", critic_gguf_sha256=SHA_C,
                server_sha256=SHA_S, machine=machine, partition="trial:j1-wave1")

    def refused(match: str, **change):
        with pytest.raises(WriteRefused, match=match):
            build(WriteConfig(**{**good, **change}))

    refused("127.0.0.1", endpoint="http://pod.example:8000/v1")
    refused("licence-checked", model="other-model")
    refused("cloud", model="nemotron:cloud")
    refused("--partition", partition=None)
    refused("--machine", machine=None)
    refused("rebuild or re-pin", server_sha256="6" * 64)
    refused("differs from the pinned", writer_gguf_sha256="b" * 64)
    refused("local critic", critic_model="")
    refused("proven critic list", critic_model="gemma4:31b")
    refused("own family", critic_model="same-critic")
    refused("comparison only", critic_model="compare-critic")
    refused("measured sha256", critic_gguf_sha256=None)
    refused("127.0.0.1", critic_endpoint="http://10.0.0.2:8081/v1")

    slots, sides, pack, names, gates, writer, critic, pins, cleanup = build(WriteConfig(**good))
    assert pins["partition"] == "trial:j1-wave1" and pins["wave"] == "inline" and pins["drafts_sha256"] is None
    assert pins["writer_family"] == "nemotron" and pins["writer_gguf_sha256"] == SHA_W and pins["writer_revision"] == "r1"
    assert pins["critic_model"] == "other-critic" and pins["critic_family"] == "muse" and pins["critic_gguf_sha256"] == SHA_C
    assert pins["llama_server_commit"] == "b6000" and pins["llama_server_sha256"] == SHA_S
    assert pins["machine"]["runtime"] == "13.4" and len(pins["writers_list_sha256"]) == 64 and "grant" not in pins
    assert cleanup is None and callable(writer) and callable(critic)
    no_critic = build(WriteConfig(**{**good, "no_critic": True, "critic_model": None}))
    assert no_critic[6] is None and "critic_model" not in no_critic[7]

    calls = []
    monkeypatch.setattr(w, "chat", lambda endpoint, model, messages, temperature=0.7, seed=None: calls.append((endpoint, model, temperature, seed)) or "ok")
    assert writer([{"role": "user", "content": "x"}], seed=2) == "ok" and critic([{"role": "user", "content": "y"}]) == "ok"
    assert critic([], temperature=0.6, seed=3) == "ok"
    assert calls == [("http://127.0.0.1:8080/v1", "big-writer", w.WRITER_TEMPERATURE, 2), ("http://127.0.0.1:8081/v1", "other-critic", 0.0, 0),
                     ("http://127.0.0.1:8081/v1", "other-critic", 0.6, 3)]

    def missing(closed_form=None):
        from role_set.gates import GatesUnavailable

        raise GatesUnavailable("REFUSED: pin it")

    monkeypatch.setattr(w, "load_gates", missing)
    refused("pin it")


def test_a_step_on_the_5090_needs_the_grant_and_a_fresh_heartbeat(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.mock import mock_gates
    from test_proven import SHA_S, SHA_W

    writers, machine = _proven_setup(tmp_path, card="NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    base = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, endpoint="http://127.0.0.1:8080/v1", model="big-writer",
                writer_gguf_sha256=SHA_W, server_sha256=SHA_S, machine=machine, partition="trial:j1-wave1", no_critic=True)
    with pytest.raises(WriteRefused, match="--grant"):
        build(WriteConfig(**base))
    with pytest.raises(WriteRefused, match="--heartbeat"):
        build(WriteConfig(grant="G-1", **base))
    hb = tmp_path / "hb"
    hb.write_text("x")
    with pytest.raises(WriteRefused, match="heartbeat is"):
        build(WriteConfig(grant="G-1", heartbeat=hb, **base), now=lambda: hb.stat().st_mtime + 999)
    pins = build(WriteConfig(grant="G-1", heartbeat=hb, **base), now=lambda: hb.stat().st_mtime + 1)[7]
    assert pins["grant"] == "G-1" and pins["writer_registry_id"] == "openmdw-1.1"
    # A rig line needs the grant whatever its card string says (fail closed).
    (tmp_path / "rig").mkdir()
    _, rig_a100 = _proven_setup(tmp_path / "rig", card="rig")
    with pytest.raises(WriteRefused, match="--grant"):
        build(WriteConfig(**{**base, "machine": rig_a100}))


def test_write_halts_are_reported_and_cleanup_runs(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.runner import RunHalted

    cleaned = []

    class Halting:
        def __init__(self, **kw):
            from role_set.runner import Status

            self.status = Status(started_at=0.0)
            self.batch_check = None  # a halted run never reaches the batch check

        def run(self, slots, **kw):
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


def test_a_rehearsal_writer_writes_only_under_rehearsal(pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch):
    import role_set.write as w
    from role_set.mock import mock_gates
    from role_set.write import output_dir, require_rehearsal_dir
    from test_proven import SHA_S, SHA_W, entry, proven_file

    rehearsal = {"rehearsal": True, "writer_model": "small-writer"}
    assert output_dir(tmp_path, rehearsal, mock=False) == tmp_path / "rehearsal"
    assert output_dir(tmp_path, {"rehearsal": False}, mock=False) == tmp_path
    assert output_dir(tmp_path, {}, mock=True) == tmp_path / "mock"
    for elsewhere in (tmp_path, tmp_path / "batch4", tmp_path / "rehearsal" / "data"):
        with pytest.raises(WriteRefused, match="rehearsal-only"):
            require_rehearsal_dir(elsewhere, rehearsal)
    require_rehearsal_dir(tmp_path / "anything", {"rehearsal": False})

    # A proven, pinned model marked rehearsal-only is a rehearsal: its pins say so, and run() writes it under rehearsal/.
    writers = proven_file(tmp_path, writers=[entry("small-writer", "nemotron", SHA_W, rehearsal_only=True)])
    (tmp_path / "m").mkdir()
    _, machine = _proven_setup(tmp_path / "m")
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    cfg = WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, endpoint="http://127.0.0.1:8080/v1",
                      model="small-writer", writer_gguf_sha256=SHA_W, server_sha256=SHA_S, machine=machine, partition="trial:rehearsal", no_critic=True)
    parts = build(cfg)
    assert parts[7]["rehearsal"] is True
    seen = {}

    class Stub:
        def __init__(self, **kw):
            from role_set.runner import Status

            seen["out_dir"] = kw["out_dir"]
            self.status = Status(started_at=0.0)
            self.batch_check = None

        def run(self, slots, **kw):
            return []

    monkeypatch.setattr(w, "Runner", Stub)
    result = run(cfg)
    assert seen["out_dir"].name == "rehearsal" and result["rehearsal"] is True
