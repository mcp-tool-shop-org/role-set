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
            return f"uv {pw.UV_VERSION} (pinned)" if any(f"uv=={pw.UV_VERSION}" in c for c in calls) else uv_on_path
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


def test_release_workflow_runs_only_on_a_published_release():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    on_block = text.split("on:", 1)[1].split("concurrency:", 1)[0]
    assert "types: [published]" in on_block
    assert "workflow_dispatch" not in on_block and "push:" not in on_block and "pull_request" not in on_block


def test_requires_python_allows_follows_pep_440_for_exact_versions_and_refuses_what_it_cannot_read():
    assert not pw.requires_python_allows("<=3.15", "3.15.1")  # 3.15.1 > 3.15
    assert pw.requires_python_allows("<=3.15.1", "3.15.1")
    assert not pw.requires_python_allows("==3.15", "3.15.1")  # not a prefix without .*
    assert pw.requires_python_allows("==3.15.1", "3.15.1") and pw.requires_python_allows("==3.15.*", "3.15.1")
    assert not pw.requires_python_allows("!=3.15.*", "3.15.1") and pw.requires_python_allows("!=3.14.*", "3.15.1")
    assert pw.requires_python_allows(">=3.15", "3.15.1") and pw.requires_python_allows("<3.16", "3.15.1")
    assert not pw.requires_python_allows("<3.15", "3.15.1")
    with pytest.raises(RuntimeError, match="REFUSED: cannot read the version"):
        pw.requires_python_allows(">=3.x", "3.15.1")
    with pytest.raises(RuntimeError, match="cannot read the requires-python clause"):
        pw.requires_python_allows(">=3.13.*", "3.15.1")


def test_a_pinned_uv_install_that_reports_another_version_is_refused(tmp_path, monkeypatch):
    def fake_sh(cmd, **kwargs):
        if cmd.endswith("--version"):
            return "uv 0.9.5 (old)"
        return ""

    monkeypatch.setattr(pw, "sh", fake_sh)
    monkeypatch.setattr(pw.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"uv 0\.9\.5 after the pinned install"):
        pw.pinned_uv()


def test_a_venv_with_a_python_but_no_role_set_is_rebuilt(tmp_path, monkeypatch):
    calls = _fake_role_set(tmp_path, monkeypatch, ">=3.13")
    (tmp_path / "rs-venv" / "bin").mkdir(parents=True)
    (tmp_path / "rs-venv" / "bin" / "python").write_text("", encoding="utf-8")  # a failed install left only python
    pw.role_set_env("c" * 40)
    assert any("uv venv" in c for c in calls) and any("pip install" in c and "-e" in c for c in calls)


def test_zstandard_in_stub_mode_on_a_host_without_a_pinned_wheel_uses_the_vm_tag(monkeypatch):
    monkeypatch.setattr(pw, "_host_py_tag", lambda: "cp315")
    monkeypatch.setattr(pw, "stub_mode", lambda: True)
    assert pw.zstandard_wheel_sha256() == pw.zstandard_wheel_sha256("cp313")
    monkeypatch.setattr(pw, "stub_mode", lambda: False)
    with pytest.raises(RuntimeError, match="no pinned zstandard"):
        pw.zstandard_wheel_sha256()  # outside stub mode a host with no pinned wheel is refused
    with pytest.raises(RuntimeError, match="no pinned zstandard"):
        pw.zstandard_wheel_sha256("cp315")


def _preflight_args(tmp_path):
    rs = tmp_path / "role-set"
    (rs / ".git").mkdir(parents=True, exist_ok=True)
    return ["--roleset", "c" * 40, "--rnd", "d" * 40, "--pack", str(tmp_path / "p.tar"), "--cards", str(tmp_path / "c.json"),
            "--models", "m", "--work", str(tmp_path)]


def test_preflight_refuses_the_wrong_uv_even_when_the_venv_already_exists(tmp_path, monkeypatch):
    pre = _preflight()
    (tmp_path / "rs-venv").mkdir()
    monkeypatch.setattr(pre, "run", lambda cmd, cwd=None: "uv 0.9.5 (old)" if cmd[:2] == ["uv", "--version"] else "")
    assert pre.main(_preflight_args(tmp_path)) == 1  # main() reports a failure as exit 1


def test_preflight_refuses_an_older_role_set_commit_and_a_venv_on_the_wrong_python(tmp_path, monkeypatch, capsys):
    pre = _preflight()
    (tmp_path / "role-set" / ".git").mkdir(parents=True, exist_ok=True)
    (tmp_path / "role-set" / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.12,<3.13"\n', encoding="utf-8")
    monkeypatch.setattr(pre, "run", lambda cmd, cwd=None: f"uv {pre.UV_VERSION}" if cmd[:2] == ["uv", "--version"] else "")
    assert pre.main(_preflight_args(tmp_path)) == 1
    assert "requires Python >=3.12,<3.13" in capsys.readouterr().out + capsys.readouterr().err
    (tmp_path / "role-set" / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.13"\n', encoding="utf-8")
    scripts = tmp_path / "rs-venv" / ("Scripts" if sys.platform == "win32" else "bin")
    scripts.mkdir(parents=True)
    (scripts / ("role-set.exe" if sys.platform == "win32" else "role-set")).write_text("", encoding="utf-8")

    def fake(cmd, cwd=None):
        if cmd[:2] == ["uv", "--version"]:
            return f"uv {pre.UV_VERSION}"
        if "sys.version" in " ".join(cmd):
            return "3.15.0a1"
        return ""

    monkeypatch.setattr(pre, "run", fake)
    assert pre.main(_preflight_args(tmp_path)) == 1


def test_ci_installs_exactly_the_pinned_uv_because_the_window_script_refuses_any_other():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    setups = ci.count("astral-sh/setup-uv@")
    assert setups and ci.count(f'version: "{pw.UV_VERSION}"') == setups
