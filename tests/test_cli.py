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
    from role_set.critic_control import ControlReport

    items = tmp_path / "items.jsonl"
    items.write_text("", encoding="utf-8")
    hb = tmp_path / "hb"
    hb.write_text("x")
    from role_set.mock import mock_gates

    monkeypatch.setattr(cli, "load_gates", mock_gates)
    assert main(["critic-control", "--items", str(items), "--grant", "", "--heartbeat", str(hb), "--out", str(tmp_path / "c.json")]) == 2
    assert "--grant" in capsys.readouterr().err
    monkeypatch.setattr(cli, "unload", lambda *a, **k: None)
    monkeypatch.setattr(cli, "ollama_chat", lambda *a, **k: types.SimpleNamespace(text="VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no", prompt_tokens=1, completion_tokens=1, model="m", seconds=0))
    monkeypatch.setattr(cli, "require_ollama_digest", lambda *a, **k: "6316f062")
    good = ControlReport(items=1, verdict_asks=1, agree=1)
    monkeypatch.setattr(cli, "run_control", lambda *a, **k: good)
    rc = main(["critic-control", "--items", str(items), "--grant", "G-1", "--heartbeat", str(hb), "--out", str(tmp_path / "c.json")])
    assert rc == 0 and json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["grant"] == "G-1"
    assert main(["critic-control", "--items", str(items), "--grant", "G-1", "--heartbeat", str(hb), "--critic-model", "gemma4:cloud", "--out", str(tmp_path / "c.json")]) == 2
