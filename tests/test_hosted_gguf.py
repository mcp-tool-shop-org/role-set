"""Pre-built GGUFs in the studio's private repo (R&D, 2026-10-18): the token file, the pilot window's prefetch, and the
prebuild window's upload, without Colab or the Hub. Byte-match or don't host; the token appears in nothing kept."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))
import pin_window as pw  # noqa: E402

SENTINEL = "hf_SENTINEL_never_persist_9f3c2a71"
GRANITE = next(m for m in pw.MODELS if m["served_as"] == "granite-4.2-30b")
REV = "a" * 40


@pytest.fixture
def vm(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "CONTENT", tmp_path)
    monkeypatch.setattr(pw, "WORK", tmp_path / "pin")
    monkeypatch.setattr(pw, "HOSTED", {})
    monkeypatch.setattr(pw, "resources", lambda *a: None)
    monkeypatch.setattr(pw, "say", lambda *a: None)
    # The VM's disk, not this host's (WSL's /tmp is a 14 GB tmpfs): an A100 VM's 200 GB free at the start.
    monkeypatch.setattr(pw.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(200e9), total=int(250e9), used=0))
    (tmp_path / "pin").mkdir()
    (tmp_path / ".hf_token").write_text(SENTINEL + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "should-be-dropped")
    return tmp_path


def _hub(monkeypatch, **attrs):
    """A stand-in huggingface_hub module: the role-set env doesn't carry it; the VM does."""

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(**attrs))


def _fake_hub(monkeypatch, content: bytes, seen: list):
    def hf_hub_download(repo, filename, revision=None, token=None, local_dir=None, **kw):
        seen.append({"repo": repo, "filename": filename, "revision": revision, "token": token})
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return str(p)

    class Api:
        def __init__(self, token=None):
            seen.append({"api_token": token})

        def get_paths_info(self, repo, paths, revision=None):
            return [SimpleNamespace(path=p, size=len(content)) for p in paths]

    _hub(monkeypatch, HfApi=Api, hf_hub_download=hf_hub_download)


def test_the_token_file_is_read_once_deleted_and_the_hub_variables_dropped(vm):
    assert pw.take_token() == SENTINEL
    assert not (vm / ".hf_token").exists() and "HF_TOKEN" not in pw.os.environ
    assert pw.take_token() is None


def test_the_prefetch_takes_each_q8_by_its_pin_before_any_wave_and_keeps_no_token(vm, monkeypatch):
    import hashlib

    content = b"gguf bytes"
    spec = {**GRANITE, "expect_sha256": hashlib.sha256(content).hexdigest()}
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    seen, result = [], {}
    _fake_hub(monkeypatch, content, seen)
    pw.prefetch_hosted([spec["served_as"]], {spec["served_as"]: spec}, result)
    assert seen == [{"api_token": SENTINEL}, {"repo": "studio/gguf-private", "filename": "granite-4.2-30b/granite-4.2-30b-Q8_0.gguf",
                     "revision": REV, "token": SENTINEL}]
    assert result["hosted_disk"]["need_gb"] == 0.0 and result["hosted_disk"]["margin_gb"] == 20
    assert not (vm / ".hf_token").exists()
    assert result["hosted"]["models"] == {"granite-4.2-30b": spec["expect_sha256"]}
    assert SENTINEL not in json.dumps(result)
    m = pw.fetch_model(spec, {})  # the Pool's fetch finds it: no download, no conversion
    assert m["gguf_sha256"] == spec["expect_sha256"] and m["gguf_source"].startswith("pre-built, studio/gguf-private@aaaa")


def test_a_hosted_file_that_isnt_its_pin_stops_the_window_and_is_deleted(vm, monkeypatch):
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    _fake_hub(monkeypatch, b"other bytes", [])
    with pytest.raises(RuntimeError, match="isn't its pin"):
        pw.prefetch_hosted([GRANITE["served_as"]], {GRANITE["served_as"]: GRANITE}, {})
    assert not list((vm / "pin" / "hosted").rglob("*.gguf")) and not pw.HOSTED


def test_a_set_that_wont_fit_with_its_margin_stops_before_any_download(vm, monkeypatch):
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    seen = []
    _fake_hub(monkeypatch, b"x", seen)
    monkeypatch.setattr(pw.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(15e9), total=int(250e9), used=0))
    with pytest.raises(RuntimeError, match="20 GB margin"):
        pw.prefetch_hosted([GRANITE["served_as"]], {GRANITE["served_as"]: GRANITE}, {})
    assert not [s for s in seen if "filename" in s]  # nothing downloaded


def test_the_prefetch_needs_a_full_revision_and_a_token(vm, monkeypatch):
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", "main")
    with pytest.raises(RuntimeError, match="full commit sha"):
        pw.prefetch_hosted([], {}, {})
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    (vm / ".hf_token").unlink()
    with pytest.raises(RuntimeError, match="no token file"):
        pw.prefetch_hosted([], {}, {})


def test_no_repo_set_means_no_prefetch_and_the_token_file_is_left_alone(vm, monkeypatch):
    monkeypatch.delenv("PIN_GGUF_REPO", raising=False)
    pw.prefetch_hosted(["granite-4.2-30b"], {"granite-4.2-30b": GRANITE}, {})
    assert (vm / ".hf_token").exists() and not pw.HOSTED


@pytest.mark.parametrize("card, want", [("---\nlicense: apache-2.0\nlibrary_name: x\n---\n# M\n", "apache-2.0"),
                                         ("---\nlicense: 'other'\n---\n", "other"), ("# no front matter\n", None)])
def test_the_cards_licence_field(tmp_path, card, want):
    p = tmp_path / "c.md"
    p.write_text(card, encoding="utf-8")
    assert pw.card_licence(p) == want


class _Api:
    def __init__(self, private=True, files=("README.md",)):
        self.private, self.files, self.calls, self.uploaded = private, list(files), [], {}

    def __call__(self, token=None):  # HfApi(token=...) or HfApi()
        self.calls.append(("token", token))
        return self

    def repo_info(self, repo):
        return SimpleNamespace(private=self.private)

    def list_repo_files(self, repo, revision=None):
        return self.files

    def upload_folder(self, repo_id, folder_path, path_in_repo, commit_message):
        self.calls.append(("upload", repo_id, path_in_repo))
        self.uploaded = {p.name: p.read_bytes() for p in Path(folder_path).iterdir()}
        return SimpleNamespace(oid="c" * 40)


def _upload(vm, monkeypatch, api, card="---\nlicense: apache-2.0\n---\n"):
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")

    def hf_hub_download(repo, filename, revision=None, local_dir=None, **kw):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(card if filename == "README.md" else f"{filename} text", encoding="utf-8")
        return str(p)

    _hub(monkeypatch, HfApi=api, hf_hub_download=hf_hub_download)
    gguf = vm / "pin" / "granite-4.2-30b-Q8_0.gguf"
    gguf.write_bytes(b"q8")
    m = {"path": str(gguf), "revision": GRANITE["revision"], "gguf_sha256": GRANITE["expect_sha256"]}
    result = {}
    pw.upload_hosted(GRANITE, m, result)
    return result


def test_an_upload_carries_the_licence_card_and_a_readme_and_names_its_pin(vm, monkeypatch):
    api = _Api(files=["README.md", "LICENSE", "NOTICE", "config.json"])
    result = _upload(vm, monkeypatch, api)
    assert result["uploaded"]["files"] == ["LICENSE", "NOTICE", "README.md", "UPSTREAM-CARD.md", "granite-4.2-30b-Q8_0.gguf"]
    readme = api.uploaded["README.md"].decode()
    assert GRANITE["revision"] in readme and pw.LLAMA_CPP_COMMIT in readme and GRANITE["expect_sha256"] in readme
    assert ("token", SENTINEL) in api.calls and not (vm / ".hf_token").exists()
    assert SENTINEL not in json.dumps(result) and not (vm / "pin" / "upload" / "granite-4.2-30b").exists()


def test_a_card_only_apache_repo_gets_the_vendored_licence_text_and_anything_else_refuses(vm, monkeypatch):
    vendored = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "licences" / "apache-2.0.txt"
    api = _Api(files=["README.md"])  # Gemma 4 and Granite 4.2: no LICENSE file, Apache-2.0 in the card
    assert "LICENSE" in _upload(vm, monkeypatch, api)["uploaded"]["files"]
    assert api.uploaded["LICENSE"] == vendored.read_bytes() and b"Apache License" in api.uploaded["LICENSE"]
    (vm / ".hf_token").write_text(SENTINEL, encoding="utf-8")
    with pytest.raises(RuntimeError, match="no known text"):
        _upload(vm, monkeypatch, _Api(files=["README.md"]), card="---\nlicense: other\n---\n")


def test_an_upload_to_a_public_repo_is_refused(vm, monkeypatch):
    api = _Api(private=False)
    with pytest.raises(RuntimeError, match="is public"):
        _upload(vm, monkeypatch, api)
    assert not any(c[0] == "upload" for c in api.calls)


def _prebuild(vm, monkeypatch, sha):
    monkeypatch.setenv("PIN_ROLE", "prebuild")
    monkeypatch.setenv("PIN_ONLY", "granite-4.2-30b")
    monkeypatch.delenv("PIN_STUB", raising=False)
    monkeypatch.setattr(pw, "sh", lambda *a, **k: "")
    monkeypatch.setattr(pw, "machine_line", lambda cpu_ok=False: {"card": "CPU", "cpu": True})
    gguf = vm / "pin" / "granite-4.2-30b-Q8_0.gguf"
    gguf.write_bytes(b"q8")
    monkeypatch.setattr(pw, "fetch_model", lambda spec, built, keep_snapshot=False: {
        "served_as": spec["served_as"], "gguf_sha256": sha, "path": str(gguf), "revision": spec["revision"]})
    uploads = []
    monkeypatch.setattr(pw, "upload_hosted", lambda spec, m, result: uploads.append(m["gguf_sha256"]))
    pw._main()
    return json.loads((vm / "pin" / "pin-result.json").read_text(encoding="utf-8")), uploads


def test_byte_match_or_dont_host(vm, monkeypatch):
    res, uploads = _prebuild(vm, monkeypatch, "0" * 64)
    assert uploads == [] and res["ok"] is False and res["model_errors"] == 1  # a mismatch is never hosted
    res, uploads = _prebuild(vm, monkeypatch, GRANITE["expect_sha256"])
    assert uploads == [GRANITE["expect_sha256"]] and res["ok"] is True


def test_a_prebuild_converts_exactly_one_model(vm, monkeypatch):
    monkeypatch.setenv("PIN_ROLE", "prebuild")
    monkeypatch.setenv("PIN_ONLY", "granite-4.2-30b,gemma-4-31b-it")
    monkeypatch.setattr(pw, "sh", lambda *a, **k: "")
    monkeypatch.setattr(pw, "machine_line", lambda cpu_ok=False: {"card": "CPU"})
    pw._main()
    res = json.loads((vm / "pin" / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is False and "exactly one" in res["error"]


def test_the_token_file_lies_outside_everything_a_window_tars_or_sends_back():
    # pilot-out.tar is `tar -C WORK pilot`; the downloads are WORK files. The token file is CONTENT/.hf_token.
    src = (Path(__file__).resolve().parents[1] / "scripts" / "colab" / "pin_window.py").read_text(encoding="utf-8")
    assert 'tar -C {WORK} -cf {WORK}/pilot-out.tar pilot"' in src
    tok = pw.token_file()
    assert tok.parent == pw.CONTENT and pw.WORK not in tok.parents and (pw.WORK / "pilot") not in tok.parents


def test_a_vendored_licence_text_that_drifted_from_its_pin_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "CONTENT", tmp_path)
    monkeypatch.setitem(pw.LICENCE_FALLBACK, "apache-2.0", ("apache-2.0.txt", "0" * 64))
    with pytest.raises(RuntimeError, match="pinned sha256"):
        pw.licence_text("apache-2.0")
    monkeypatch.setitem(pw.LICENCE_FALLBACK, "apache-2.0", ("missing.txt", "0" * 64))
    with pytest.raises(RuntimeError, match="isn't here"):
        pw.licence_text("apache-2.0")
