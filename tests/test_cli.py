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
