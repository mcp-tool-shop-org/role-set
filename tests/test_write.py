import json
from pathlib import Path

import pytest
from test_cards import cards_file

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
    assert receipt["gates"]["version"] == "1.7.6" and result["batch_check"] == "clean"
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
    good = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, cards=cards_file(tmp_path), endpoint="http://127.0.0.1:8080/v1", model="big-writer",
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
    monkeypatch.setattr(w, "chat", lambda endpoint, model, messages, seed=None, **kw: calls.append((endpoint, model, seed, kw)) or "ok")
    assert writer([{"role": "user", "content": "x"}], seed=2) == "ok" and critic([{"role": "user", "content": "y"}]) == "ok"
    assert critic([], temperature=0.6, seed=3) == "ok"
    # Writer and critic both run at their card: every sampling key and the template kwargs on every call, the same
    # budget, and a caller's temperature ignored (R&D, 2026-10-17).
    from test_cards import SAMPLING

    assert [(c[0], c[1], c[2]) for c in calls] == [("http://127.0.0.1:8080/v1", "big-writer", 2), ("http://127.0.0.1:8081/v1", "other-critic", 0),
                                                   ("http://127.0.0.1:8081/v1", "other-critic", 3)]
    for c in calls:
        assert c[3]["sampling"] == SAMPLING and c[3]["chat_template_kwargs"] == {"enable_thinking": True}
        assert c[3]["max_tokens"] == 16384 and "temperature" not in c[3]
    assert pins["card_sampling"] == SAMPLING and pins["critic_card_candidate"] == "other-critic" and len(pins["cards_sha256"]) == 64
    refused("--cards", cards=None)

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
    base = dict(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, cards=cards_file(tmp_path), endpoint="http://127.0.0.1:8080/v1", model="big-writer",
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
    committed_pool = Path(__file__).resolve().parents[1] / "roles" / "verifier" / "pool.jsonl"
    cfg = WriteConfig(plan=plan_file, pack=pack_dir, out=tmp_path / "o", writers_file=writers, cards=cards_file(tmp_path), endpoint="http://127.0.0.1:8080/v1",
                      model="small-writer", writer_gguf_sha256=SHA_W, server_sha256=SHA_S, machine=machine, partition="trial:rehearsal", no_critic=True, pool=committed_pool)
    parts = build(cfg)
    assert parts[7]["rehearsal"] is True and len(parts[7]["pool_sha256"]) == 64
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


def test_no_critic_never_runs_on_production(tmp_path: Path, monkeypatch):
    """Argon §9.4 #1: every counted row has its second reader."""

    from role_set.guards import GuardError

    cfg = WriteConfig(plan=tmp_path / "p.json", pack=tmp_path, out=tmp_path / "o", no_critic=True, partition="production")
    # The guard sits right after the partition is read; it refuses before any model is resolved.
    import role_set.write as w

    with pytest.raises((WriteRefused, GuardError), match="--no-critic never runs on partition production"):
        w._require_critic_policy(cfg, "production")
    w._require_critic_policy(WriteConfig(plan=tmp_path, pack=tmp_path, out=tmp_path, no_critic=True), "trial:x")


def test_real_write_requires_pool_and_sufficient_excerpts_before_warmup_and_pins_pool_sha256(
    pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch
):
    import hashlib

    import role_set.write as w
    from role_set.mock import mock_gates
    from test_proven import SHA_S, SHA_W, entry, proven_file

    writers = proven_file(tmp_path, writers=[entry("small-writer", "nemotron", SHA_W)])
    (tmp_path / "m").mkdir()
    _, machine = _proven_setup(tmp_path / "m")
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())
    writer_calls: list = []
    monkeypatch.setattr(w, "chat", lambda *a, **kw: writer_calls.append(a))

    base_kw = dict(
        plan=plan_file,
        pack=pack_dir,
        out=tmp_path / "o",
        writers_file=writers,
        cards=cards_file(tmp_path),
        endpoint="http://127.0.0.1:8080/v1",
        model="small-writer",
        writer_gguf_sha256=SHA_W,
        server_sha256=SHA_S,
        machine=machine,
        partition="trial:pool-check",
        no_critic=True,
    )

    # 1. Missing --pool on a real write with real_source slots refuses before warmup with zero writer calls.
    with pytest.raises(WriteRefused, match=r"REFUSED: --pool is required when the plan has real_source slots \(kind '.*': need \d+, have 0\)"):
        run(WriteConfig(**base_kw))
    assert writer_calls == []

    # 2. A short kind in the pool (fewer excerpts than the plan's real_source slots of that kind) refuses before warmup.
    short_pool = tmp_path / "short-pool.jsonl"
    short_pool.write_text(
        json.dumps({
            "text": "## [1.0.0] - 2026-01-01\n- Added a flag.\n- Fixed a path.\n- Documented output.\n",
            "content_kind": "changelog",
            "source": {"repo": "x/y", "commit": "abc1234", "path": "CHANGELOG.md", "lines": "L1-L4", "licence": "MIT License"},
        })
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WriteRefused, match=r"REFUSED: real_source pool for kind '.*' has \d+ excerpt\(s\), need \d+"):
        run(WriteConfig(**base_kw, pool=short_pool))
    assert writer_calls == []

    # 3. pool_sha256 lands in the pins when cfg.pool is loaded.
    committed_pool = Path(__file__).resolve().parents[1] / "roles" / "verifier" / "pool.jsonl"
    expected_sha = hashlib.sha256(committed_pool.read_bytes()).hexdigest()
    seen_pins: dict = {}

    class StubRunner:
        def __init__(self, **kw):
            from role_set.runner import Status

            seen_pins.update(kw["pins"])
            self.status = Status(started_at=0.0)
            self.batch_check = None

        def run(self, slots, **kw):
            return []

    monkeypatch.setattr(w, "Runner", StubRunner)
    run(WriteConfig(**base_kw, pool=committed_pool))
    assert seen_pins["pool_sha256"] == expected_sha
    assert writer_calls == []


def test_server_builds_pin_record_in_write_and_waves_for_default_and_g4(
    pack_dir: Path, plan_file: Path, tmp_path: Path, monkeypatch
):
    import role_set.write as w
    from role_set import waves
    from role_set.mock import mock_gates
    from test_proven import DEFAULT_SERVER_SHA, G4_SERVER_SHA, SHA_W

    committed_writers = Path(__file__).resolve().parents[1] / "writers.json"
    committed_pool = Path(__file__).resolve().parents[1] / "roles" / "verifier" / "pool.jsonl"
    raw_writers = json.loads(committed_writers.read_text(encoding="utf-8"))
    gemma = next(e for e in raw_writers["writers"] if e["served_as"] == "gemma-4-31b-it")
    gemma_sha = gemma["gguf_sha256"]
    cards_path = Path(__file__).parent / "fixtures" / "card-values.json"

    machine_a100 = tmp_path / "machine-a100.json"
    machine_a100.write_text(
        json.dumps({"driver": "580.82", "compat": False, "runtime": "13.4", "card": "NVIDIA A100-SXM4-40GB", "colab_release_tag": "release-colab_20261015"}),
        encoding="utf-8",
    )
    machine_g4 = tmp_path / "machine-g4.json"
    machine_g4.write_text(
        json.dumps({"driver": "580.82", "compat": False, "runtime": "13.4", "card": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "colab_release_tag": "release-colab_20261023"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(w, "load_gates", lambda closed_form=None: mock_gates())

    # 1. Real build() + wave run with default server sha (b4659b3f...) vs G4 server sha (39742d8a...)
    pins_default = build(
        WriteConfig(
            plan=plan_file,
            pack=pack_dir,
            out=tmp_path / "real-default",
            writers_file=committed_writers,
            cards=cards_path,
            endpoint="http://127.0.0.1:8080/v1",
            model="gemma-4-31b-it",
            writer_gguf_sha256=gemma_sha,
            server_sha256=DEFAULT_SERVER_SHA,
            machine=machine_a100,
            partition="trial:a100-default",
            no_critic=True,
            pool=committed_pool,
        )
    )[7]
    assert pins_default["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert pins_default["llama_server_sha256"] == DEFAULT_SERVER_SHA
    assert pins_default["llama_server_archs"] == "80;90;100 (SASS only, no PTX)"

    pins_g4 = build(
        WriteConfig(
            plan=plan_file,
            pack=pack_dir,
            out=tmp_path / "real-g4",
            writers_file=committed_writers,
            cards=cards_path,
            endpoint="http://127.0.0.1:8080/v1",
            model="gemma-4-31b-it",
            writer_gguf_sha256=gemma_sha,
            server_sha256=G4_SERVER_SHA,
            machine=machine_g4,
            partition="trial:g4-build",
            no_critic=True,
            pool=committed_pool,
        )
    )[7]
    assert pins_g4["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert pins_g4["llama_server_sha256"] == G4_SERVER_SHA
    assert pins_g4["llama_server_archs"] == "80;90;100;120 (sm_120a SASS)"

    # 2. Unlisted sha is refused on both real and mock runs
    with pytest.raises(WriteRefused, match="rebuild or re-pin"):
        build(
            WriteConfig(
                plan=plan_file,
                pack=pack_dir,
                out=tmp_path / "real-bad",
                writers_file=committed_writers,
                cards=cards_path,
                endpoint="http://127.0.0.1:8080/v1",
                model="gemma-4-31b-it",
                writer_gguf_sha256=gemma_sha,
                server_sha256=SHA_W,
                machine=machine_g4,
                partition="trial:g4-bad",
                no_critic=True,
                pool=committed_pool,
            )
        )
    assert (
        main(
            [
                "write",
                "--plan",
                str(plan_file),
                "--pack",
                str(pack_dir),
                "--writers",
                str(committed_writers),
                "--out",
                str(tmp_path / "mock-bad"),
                "--mock",
                "--wave",
                "write",
                "--partition",
                "trial:mock-bad",
                "--llama-server-sha256",
                SHA_W,
            ]
        )
        == 2
    )

    # 3. End-to-end `role-set write` in mock mode with `--llama-server-sha256` set to the G4 sha starts its wave and records the matched G4 entry in the wave header (and default when set to the default sha)
    rc_g4 = main(
        [
            "write",
            "--plan",
            str(plan_file),
            "--pack",
            str(pack_dir),
            "--writers",
            str(committed_writers),
            "--out",
            str(tmp_path / "mock-g4-w0"),
            "--mock",
            "--wave",
            "write",
            "--partition",
            "trial:mock-g4",
            "--llama-server-sha256",
            G4_SERVER_SHA,
        ]
    )
    assert rc_g4 == 0
    header_g4, drafts_g4 = waves.read(tmp_path / "mock-g4-w0" / "mock" / waves.DRAFTS_FILE)
    assert len(drafts_g4) == 40 and all(d["state"] == "awaiting_critique" for d in drafts_g4)
    assert header_g4["writer_host"]["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert header_g4["writer_host"]["llama_server_sha256"] == G4_SERVER_SHA
    assert header_g4["writer_host"]["llama_server_archs"] == "80;90;100;120 (sm_120a SASS)"
    assert header_g4["waves"][0]["pins"]["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert header_g4["waves"][0]["pins"]["llama_server_sha256"] == G4_SERVER_SHA
    assert header_g4["waves"][0]["pins"]["llama_server_archs"] == "80;90;100;120 (sm_120a SASS)"

    rc_def = main(
        [
            "write",
            "--plan",
            str(plan_file),
            "--pack",
            str(pack_dir),
            "--writers",
            str(committed_writers),
            "--out",
            str(tmp_path / "mock-def-w0"),
            "--mock",
            "--wave",
            "write",
            "--partition",
            "trial:mock-def",
            "--llama-server-sha256",
            DEFAULT_SERVER_SHA,
        ]
    )
    assert rc_def == 0
    header_def, drafts_def = waves.read(tmp_path / "mock-def-w0" / "mock" / waves.DRAFTS_FILE)
    assert len(drafts_def) == 40
    assert header_def["writer_host"]["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert header_def["writer_host"]["llama_server_sha256"] == DEFAULT_SERVER_SHA
    assert header_def["writer_host"]["llama_server_archs"] == "80;90;100 (SASS only, no PTX)"
    assert header_def["waves"][0]["pins"]["llama_server_commit"] == "620d2225cc436ab0b001e3200e590e3b286ef9ab"
    assert header_def["waves"][0]["pins"]["llama_server_sha256"] == DEFAULT_SERVER_SHA
    assert header_def["waves"][0]["pins"]["llama_server_archs"] == "80;90;100 (SASS only, no PTX)"

