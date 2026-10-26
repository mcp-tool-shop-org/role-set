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


# ---- the window's interpreter is an exact, released CPython, built by the pinned uv ------------------------------

import importlib.util  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "colab"))
import pin_window as pw  # noqa: E402

EXACT = re.compile(r"^\d+\.\d+\.\d+$")


def _preflight():
    spec = importlib.util.spec_from_file_location("preflight_window", ROOT / "scripts" / "colab" / "preflight_window.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_python_and_uv_pins_are_exact_released_patch_versions_and_agree_between_the_scripts():
    pre = _preflight()
    for value in (pw.ROLE_SET_PYTHON, pw.UV_VERSION, pre.ROLE_SET_PYTHON, pre.UV_VERSION):
        assert EXACT.match(value), f"{value!r} must be an exact patch release (a bare minor or a pre-release resolves to a pre-release)"
    assert (pw.ROLE_SET_PYTHON, pw.UV_VERSION) == (pre.ROLE_SET_PYTHON, pre.UV_VERSION) == ("3.15.1", "0.13.5")


def test_requires_python_allows_reads_the_clauses_role_set_has_used():
    assert pw.requires_python_allows(">=3.13", "3.15.1")
    assert not pw.requires_python_allows(">=3.12,<3.13", "3.15.1")
    assert pw.requires_python_allows(">=3.12,<3.13", "3.12.4")
    assert not pw.requires_python_allows(">=3.16", "3.15.1")
    assert pw.requires_python_allows("==3.15.*", "3.15.1") and not pw.requires_python_allows("==3.12.*", "3.15.1")
    with pytest.raises(RuntimeError, match="cannot read"):
        pw.requires_python_allows("~=3.13", "3.15.1")


def _fake_role_set(tmp_path, monkeypatch, requires, python="3.15.1", uv_on_path="uv 0.13.5 (abc 2026-10-20)"):
    src = tmp_path / "role-set"
    (src / ".git").mkdir(parents=True)
    (src / "pyproject.toml").write_text(f'[project]\nrequires-python = "{requires}"\n', encoding="utf-8")
    calls = []

    def fake_sh(cmd, **kwargs):
        calls.append(cmd)
        if cmd.endswith("--version"):
            return uv_on_path
        if "rev-parse" in cmd:
            return "c" * 40
        if "sys.version" in cmd:
            return python
        if "sha256sum" in cmd:
            return "d" * 64
        return ""

    monkeypatch.setattr(pw, "CONTENT", tmp_path)
    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setattr(pw.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    return calls


def test_a_window_refuses_a_role_set_commit_that_needs_an_older_python(tmp_path, monkeypatch):
    calls = _fake_role_set(tmp_path, monkeypatch, ">=3.12,<3.13")
    with pytest.raises(RuntimeError, match=r"requires Python >=3\.12,<3\.13.*cannot be reproduced"):
        pw.role_set_env("c" * 40)
    assert not any("uv venv" in c for c in calls)


def test_the_venv_is_exactly_the_pinned_python_and_the_run_record_names_it(tmp_path, monkeypatch):
    _fake_role_set(tmp_path, monkeypatch, ">=3.13")
    record = pw.role_set_env("c" * 40)
    assert record["python"] == "3.15.1" and record["python_sha256"] == "d" * 64 and record["uv"].startswith("uv 0.13.5")
    _fake_role_set(tmp_path / "again", monkeypatch, ">=3.13", python="3.15.0a1")
    with pytest.raises(RuntimeError, match="not 3.15.1"):
        pw.role_set_env("c" * 40)


def test_an_older_uv_on_the_vm_is_replaced_by_the_pinned_one(tmp_path, monkeypatch):
    calls = _fake_role_set(tmp_path, monkeypatch, ">=3.13", uv_on_path="uv 0.9.5 (old)")
    pw.role_set_env("c" * 40)
    assert any(f"pip install -q uv=={pw.UV_VERSION}" in c for c in calls)


def test_every_action_in_the_workflows_is_pinned_by_commit_sha_with_a_version_comment():
    found = 0
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for ref, rest in re.findall(r"uses: (\S+)(.*)", workflow.read_text(encoding="utf-8")):
            found += 1
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), f"{workflow.name}: {ref} is not pinned by commit sha"
            assert re.search(r"# v\d", rest), f"{workflow.name}: {ref} has no version comment"
    assert found
