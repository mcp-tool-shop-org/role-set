import json
from pathlib import Path

import pytest

from role_set.cli import main


def test_plan_writes_slots_and_figures(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T02,T03", "--seed", "4", "--out", str(out)]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert len(plan["slots"]) == 80 and len(plan["figures"]) == 2
    assert "wrote 80 slots" in capsys.readouterr().out


def test_plan_refuses_an_unknown_lesson(tmp_path: Path, capsys):
    assert main(["plan", "--lessons", "T99", "--out", str(tmp_path / "p.json")]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_plan_refuses_when_the_menu_is_too_small(tmp_path: Path, capsys):
    assert main(["plan", "--lessons", "T02,T03,T06", "--out", str(tmp_path / "p.json")]) == 2
    assert "menu" in capsys.readouterr().err


def test_cost_reads_a_plan(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    main(["plan", "--lessons", "T02", "--out", str(out)])
    assert main(["cost", "--plan", str(out), "--profile", "two-card"]) == 0
    figures = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert figures["slots"] == 40 and figures["dollars"] > 0
    assert main(["cost", "--plan", str(out), "--profile", "frontier", "--attempts", "0"]) == 2


def test_write_refuses_without_what_it_needs(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    main(["plan", "--lessons", "T02", "--out", str(out)])
    # no pack on disk: refused before anything else
    assert main(["write", "--plan", str(out), "--pack", str(tmp_path / "nopack"), "--out", str(tmp_path / "o"), "--mock"]) == 2
    assert "pack sync first" in capsys.readouterr().err


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_pack_sync_and_verify_through_the_cli(tmp_path: Path, capsys, monkeypatch):
    from role_set import cli
    from role_set.pack import PackError, PackManifest

    m = PackManifest(role="verifier", commit="abc1234", files={"a": "1"}, pack_sha256="deadbeef")
    monkeypatch.setattr(cli, "sync", lambda *a, **k: m)
    monkeypatch.setattr(cli, "verify", lambda *a, **k: m)
    assert main(["pack", "sync", "--repo", str(tmp_path), "--commit", "abc1234", "--out", str(tmp_path / "p")]) == 0
    assert "deadbeef" in capsys.readouterr().out
    assert main(["pack", "verify", "--pack", str(tmp_path / "p")]) == 0

    def boom(*a, **k):
        raise PackError("REFUSED: test")

    monkeypatch.setattr(cli, "sync", boom)
    monkeypatch.setattr(cli, "verify", boom)
    assert main(["pack", "sync", "--repo", str(tmp_path), "--commit", "main"]) == 2
    assert main(["pack", "verify"]) == 2


def test_ingest_and_store_through_the_cli(tmp_path: Path, capsys, monkeypatch):
    from role_set import cli
    from role_set.ingest import IngestError, IngestResult

    monkeypatch.setattr(cli, "interview_questions", lambda *a, **k: ["q"] * 12)
    monkeypatch.setattr(cli, "ingest", lambda *a, **k: IngestResult(ingest_id=1, files=1, accepted=3, duplicates=0))
    args = ["ingest", "--receipts", str(tmp_path), "--store", str(tmp_path / "s.db"), "--repo", str(tmp_path), "--commit", "abc1234"]
    assert main(args) == 0
    assert '"accepted": 3' in capsys.readouterr().out

    def boom(*a, **k):
        raise IngestError("REFUSED: test")

    monkeypatch.setattr(cli, "ingest", boom)
    assert main(args) == 2
    assert main(["store", "status", "--store", str(tmp_path / "s.db")]) == 0
    assert main(["store", "void", "--store", str(tmp_path / "s.db"), "--round", "1", "--reason", "test"]) == 0
    assert '"voided_rounds": [1]' in capsys.readouterr().out


def test_critic_control_through_the_cli(tmp_path: Path, capsys, monkeypatch):
    import types

    from role_set import cli
    from role_set import write as w
    from role_set.critic_control import ControlReport
    from test_proven import SHA_C, SHA_S, proven_file

    items = tmp_path / "items.jsonl"
    items.write_text("", encoding="utf-8")
    hb = tmp_path / "hb"
    hb.write_text("x")
    writers = proven_file(tmp_path)
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps({"driver": "580", "compat": False, "runtime": "13.4", "card": "NVIDIA GeForce RTX 5090", "rig": "robot"}), encoding="utf-8")
    from role_set.mock import mock_gates

    monkeypatch.setattr(cli, "load_gates", mock_gates)
    from test_cards import cards_file

    args = ["critic-control", "--items", str(items), "--writers", str(writers), "--cards", str(cards_file(tmp_path)), "--machine", str(machine), "--critic-endpoint", "http://127.0.0.1:8081/v1",
            "--critic-model", "other-critic", "--critic-gguf-sha256", SHA_C, "--llama-server-sha256", SHA_S, "--out", str(tmp_path / "c.json")]
    # The machine line names the 5090, so the grant and heartbeat are required.
    assert main(args) == 2 and "--grant" in capsys.readouterr().err
    assert main([*args, "--grant", "G-1"]) == 2 and "--heartbeat" in capsys.readouterr().err
    monkeypatch.setattr(w, "chat", lambda *a, **k: types.SimpleNamespace(text="VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no", prompt_tokens=1, completion_tokens=1, model="m", seconds=0))
    good = ControlReport(items=1, verdict_asks=1, agree=1)
    monkeypatch.setattr(cli, "run_control", lambda *a, **k: good)
    assert main([*args, "--grant", "G-1", "--heartbeat", str(hb)]) == 0
    out = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    assert out["grant"] == "G-1" and out["critic_pins"]["gguf_sha256"] == SHA_C and out["llama_server"]["sha256"] == SHA_S
    assert out["critic_card"]["card_candidate"] == "other-critic" and len(out["critic_card"]["cards_sha256"]) == 64
    # A same-family critic is fine for the control (no writer is involved); a cloud tag and a drifted GGUF are not.
    def changed(flag: str, value: str) -> list[str]:
        out = list(args)
        out[out.index(flag) + 1] = value
        return [*out, "--grant", "G-1", "--heartbeat", str(hb)]

    capsys.readouterr()
    assert main(changed("--critic-model", "nemotron:cloud")) == 2 and "cloud" in capsys.readouterr().err
    assert main(changed("--critic-gguf-sha256", "b" * 64)) == 2 and "differs from the pinned" in capsys.readouterr().err
    assert main(changed("--critic-model", "same-critic")) == 0


def test_harvest_through_the_cli(tmp_path: Path, capsys, monkeypatch):
    from role_set import cli
    from role_set.harvest import HarvestReport

    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"repos": [{"name": "x", "commit": "a" * 40}]}), encoding="utf-8")
    monkeypatch.setattr(cli, "harvest", lambda repos, clones, out: HarvestReport(repos=1, excerpts=5, by_kind={"code": 5}))
    assert main(["harvest", "--allow-list", str(allow), "--clones", str(tmp_path), "--out", str(tmp_path / "pool.jsonl")]) == 0
    assert '"excerpts": 5' in capsys.readouterr().out
    monkeypatch.setattr(cli, "harvest", lambda repos, clones, out: HarvestReport())
    assert main(["harvest", "--allow-list", str(allow), "--clones", str(tmp_path)]) == 2


def test_the_pilot_trait_t18_plans(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    assert main(["plan", "--lessons", "T18", "--seed", "7", "--out", str(out)]) == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    sides = {s["side"] for s in plan["slots"]}
    assert len(plan["slots"]) == 40 and sides == {"doesn't rewrite the statement", "still quotes the deciding line"}


def test_every_trait_with_sides_plans_on_its_own(tmp_path: Path):
    from role_set.cli import VERIFIER_SIDES

    for lesson in sorted(VERIFIER_SIDES):
        out = tmp_path / f"{lesson}.json"
        assert main(["plan", "--lessons", lesson, "--seed", "7", "--out", str(out)]) == 0, lesson
        assert len(json.loads(out.read_text(encoding="utf-8"))["slots"]) == 40


def test_plan_lesson_t10_buffers_writes_slots_and_sha256(tmp_path: Path, capsys):
    import hashlib

    from role_set import domains, frames, solvefirst
    from role_set.plan import Slot

    out = tmp_path / "t10-buffered.json"
    assert main(["plan", "--lesson", "T10", "--buffers", "--out", str(out)]) == 0
    sha_file = Path(f"{out}.sha256")
    assert out.exists() and sha_file.exists()
    expected_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert sha_file.read_text(encoding="utf-8").strip().split()[0] == expected_sha
    plan = json.loads(out.read_text(encoding="utf-8"))
    slots = [Slot(**s) for s in plan["slots"]]
    assert len(slots) == 60
    assert sum(1 for s in slots if s.buffer) == 10
    for s in slots:
        if domains.bears_domain(s):
            assert s.domain, s.id
        if frames.framed(s.kind, s.e_type):
            assert s.frame, s.id
        if s.kind == "B":
            assert s.pack and s.b_seed is not None and s.b_path, s.id
        if s.kind == "B" and s.b_path == "solve_first":
            assert s.b_skeleton, s.id


def test_plan_writes_lf_bytes_with_matching_sha256_and_no_cr(tmp_path: Path):
    import hashlib

    out = tmp_path / "plan-lf.json"
    assert main(["plan", "--lesson", "T10", "--buffers", "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw
    lf_bytes = raw.replace(b"\r\n", b"\n")
    assert raw == lf_bytes
    lf_sha = hashlib.sha256(lf_bytes).hexdigest()
    sidecar = Path(f"{out}.sha256")
    assert b"\r" not in sidecar.read_bytes()
    assert sidecar.read_text(encoding="utf-8").strip().split()[0] == lf_sha

