"""role-set supports current Python: 3.13 and newer, with 3.15 the main target.

The window scripts build role-set's venv on the same current Python (the studio tests on current Python only).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NL = chr(10)


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_requires_python_is_current_python_and_the_lock_and_dev_pin_agree():
    pyproject = _project()
    assert pyproject["project"]["requires-python"] == ">=3.13"
    assert (ROOT / ".python-version").read_text(encoding="utf-8").strip() == "3.15"
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    assert lock["requires-python"] == ">=3.13"
    # zstandard 0.25.0 (the latest) has no cp315 wheel on PyPI (read 2026-10-25), so on 3.15 it builds from its sdist
    # (measured on Windows with uv: the C extension builds). The dev dependency stays unmarked: tests read real archives.
    assert "zstandard==0.25.0" in pyproject["dependency-groups"]["dev"]


def test_ci_runs_the_suite_on_3_15_with_the_required_check_name_kept_and_on_3_13_and_3_14():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert NL + "  test:" + NL in ci
    assert NL + "  test-py313:" + NL in ci
    assert NL + "  test-py314:" + NL in ci
    assert 'python-version: "3.12"' not in ci
    test_job = ci.split(NL + "  test:" + NL, 1)[1].split(NL + "  test-py313:", 1)[0]
    assert 'python-version: "3.15"' in test_job
    assert "atlas" in test_job
    for leg, version in (("test-py313", "3.13"), ("test-py314", "3.14")):
        body = ci.split(NL + "  " + leg + ":" + NL, 1)[1].split(NL + "  test-py3", 1)[0]
        assert f'python-version: "{version}"' in body
        assert "- name: Atlas check" not in body


def test_the_window_scripts_build_role_set_on_current_python_not_3_12():
    for name in ("pin_window.py", "preflight_window.py"):
        text = (ROOT / "scripts" / "colab" / name).read_text(encoding="utf-8")
        assert '"3.12"' not in text, name
        assert "-p 3.12" not in text, name
        assert "3.15" in text, name
