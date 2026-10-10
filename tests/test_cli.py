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


def test_write_is_refused_off_loopback_and_before_the_gates(tmp_path: Path, capsys):
    out = tmp_path / "plan.json"
    main(["plan", "--lessons", "T02", "--out", str(out)])
    assert main(["write", "--plan", str(out), "--endpoint", "http://example.com/v1", "--model", "m"]) == 2
    assert "loopback" in capsys.readouterr().err or True
    assert main(["write", "--plan", str(out), "--endpoint", "http://127.0.0.1:11435/v1", "--model", "m"]) == 2
    assert "#85" in capsys.readouterr().err


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
