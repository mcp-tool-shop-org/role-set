"""Pre-built GGUFs in the studio's private repo (R&D, 2026-10-18): the token file, the pilot window's prefetch, and the
prebuild window's upload, without Colab or the Hub. Byte-match or don't host; the token appears in nothing kept."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))
import pin_window as pw  # noqa: E402

SENTINEL = "hf_" + "SENTINELneverPERSIST9f3c2a71abcdEFGHijk"  # split: the source never holds the shape
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
    (tmp_path / "shm").mkdir()
    monkeypatch.setattr(pw, "TOKEN_PATH", tmp_path / "shm" / "hf.token")
    monkeypatch.setattr(pw, "_TOKEN", None)
    pw.TOKEN_PATH.write_text(SENTINEL + "\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "should-be-dropped")
    pw.read_token_file(needed=True)  # as _main does first thing
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
    assert not pw.token_file().exists() and "HF_TOKEN" not in pw.os.environ
    assert pw.take_token() == SENTINEL
    assert pw.take_token() is None and pw._TOKEN is None  # handed to one step, then forgotten


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
    assert not pw.token_file().exists()
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
    pw.take_token()  # no token left
    with pytest.raises(RuntimeError, match="no token file"):
        pw.prefetch_hosted([], {}, {})


@pytest.mark.parametrize("card, want", [("---\nlicense: apache-2.0\nlibrary_name: x\n---\n# M\n", "apache-2.0"),
                                         ("---\nlicense: 'other'\n---\n", "other"), ("# no front matter\n", None)])
def test_the_cards_licence_field(tmp_path, card, want):
    p = tmp_path / "c.md"
    p.write_text(card, encoding="utf-8")
    assert pw.card_licence(p) == want


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
    # pilot-out.tar is `tar -C WORK pilot`; the downloads are WORK files. The token file is in RAM-backed /dev/shm.
    src = (Path(__file__).resolve().parents[1] / "scripts" / "colab" / "pin_window.py").read_text(encoding="utf-8")
    assert 'tar -C {WORK} -cf {WORK}/pilot-out.tar pilot"' in src
    tok = pw.token_file()
    assert tok == Path("/dev/shm/hf.token") and pw.CONTENT not in tok.parents and pw.WORK not in tok.parents


def test_a_token_no_step_will_read_is_deleted_at_the_start(vm, monkeypatch):
    pw.TOKEN_PATH.write_text(SENTINEL, encoding="utf-8")  # a fresh file for this job
    monkeypatch.setattr(pw, "_TOKEN", None)
    monkeypatch.setenv("PIN_ROLE", "smoke")
    monkeypatch.delenv("PIN_GGUF_REPO", raising=False)

    def stop(cpu_ok=False):
        assert not pw.token_file().exists()  # already gone before anything else runs
        raise RuntimeError("stop here")

    monkeypatch.setattr(pw, "machine_line", stop)
    pw._main()
    assert not pw.token_file().exists() and pw._TOKEN is None  # read and dropped, not kept


def test_a_pilot_window_without_a_repo_deletes_a_token_it_was_sent(vm, monkeypatch):
    monkeypatch.delenv("PIN_GGUF_REPO", raising=False)
    pw.prefetch_hosted(["granite-4.2-30b"], {"granite-4.2-30b": GRANITE}, {})
    assert pw._TOKEN is None and not pw.HOSTED


def test_a_window_that_needs_the_token_refuses_without_dev_shm(vm, monkeypatch):
    monkeypatch.setattr(pw, "TOKEN_PATH", vm / "no-shm" / "hf.token")
    monkeypatch.setenv("PIN_ROLE", "prebuild")
    monkeypatch.setattr(pw, "machine_line", lambda cpu_ok=False: (_ for _ in ()).throw(AssertionError("ran on")))
    assert pw._main() == 1
    res = json.loads((vm / "pin" / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is False and "no /dev/shm" in res["error"]


def test_a_hosted_file_the_repo_doesnt_list_stops_before_any_download(vm, monkeypatch):
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    seen = []

    class Api:
        def __init__(self, token=None):
            pass

        def get_paths_info(self, repo, paths, revision=None):
            return []

    _hub(monkeypatch, HfApi=Api, hf_hub_download=lambda *a, **k: seen.append(a))
    with pytest.raises(RuntimeError, match="doesn't hold granite-4.2-30b/granite-4.2-30b-Q8_0.gguf"):
        pw.prefetch_hosted([GRANITE["served_as"]], {GRANITE["served_as"]: GRANITE}, {})
    assert not seen


def test_a_prebuild_of_a_model_without_a_pin_refuses_before_any_download(vm, monkeypatch):
    monkeypatch.setitem(GRANITE, "expect_sha256", None)
    monkeypatch.setenv("PIN_ROLE", "prebuild")
    monkeypatch.setenv("PIN_ONLY", "granite-4.2-30b")
    monkeypatch.setattr(pw, "sh", lambda *a, **k: "")
    monkeypatch.setattr(pw, "machine_line", lambda cpu_ok=False: {"card": "CPU"})
    fetched = []
    monkeypatch.setattr(pw, "fetch_model", lambda *a, **k: fetched.append(a))
    pw._main()
    res = json.loads((vm / "pin" / "pin-result.json").read_text(encoding="utf-8"))
    assert res["ok"] is False and "has no expect_sha256" in res["error"] and not fetched


# ---- the upload in a fresh interpreter (R&D, 2026-10-18, window 13i): the real child process, a stand-in hub

FAKE_HUB = r"""
import hashlib, json, os
from pathlib import Path
from types import SimpleNamespace
__version__ = "0.36.2-standin"
CFG = json.loads(os.environ.get("FAKE_HUB_CFG", "{}"))
LOG = Path(os.environ["FAKE_HUB_LOG"])
if CFG.get("die_on_import"):
    raise SystemExit(3)


def _log(rec):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


class HfApi:
    def __init__(self, token=None):
        _log({"call": "init", "token_sha": hashlib.sha256(token.encode()).hexdigest() if token else None})

    def repo_info(self, repo):
        return SimpleNamespace(private=CFG.get("private", True), card_data={"license": CFG.get("repo_licence", "apache-2.0")})

    def list_repo_files(self, repo, revision=None):
        return CFG.get("files", ["README.md"])

    def upload_file(self, *a, **k):
        _log({"call": "upload_file"})

    def upload_folder(self, repo_id, folder_path, path_in_repo, commit_message):
        if CFG.get("fail_upload"):
            raise RuntimeError(CFG["fail_upload"])
        files = {q.name: q.read_bytes().decode("latin-1") for q in Path(folder_path).iterdir()}
        _log({"call": "upload", "repo": repo_id, "path_in_repo": path_in_repo, "files": sorted(files),
              "readme": files.get("README.md"), "license": files.get("LICENSE")})
        return SimpleNamespace(oid="c" * 40)


def hf_hub_download(repo, filename, revision=None, local_dir=None, **kw):
    q = Path(local_dir) / filename
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(CFG.get("card", "---\nlicense: apache-2.0\n---\n") if filename == "README.md" else filename + " text",
                 encoding="utf-8")
    return str(q)
"""


def _child_upload(vm, monkeypatch, tmp_path, spec=GRANITE, **cfg):
    """upload_hosted with its real child process, which imports the stand-in hub. Returns (result, hub log, what the
    child was started with)."""

    import hashlib
    import subprocess

    hub = tmp_path / "standin"
    (hub / "huggingface_hub").mkdir(parents=True, exist_ok=True)
    (hub / "huggingface_hub" / "__init__.py").write_text(FAKE_HUB, encoding="utf-8")
    log = tmp_path / "hub.log"
    log.unlink(missing_ok=True)
    monkeypatch.setattr(pw, "UPLOAD_CHILD_ENV", {"PYTHONPATH": str(hub), "FAKE_HUB_LOG": str(log),
                                                  "FAKE_HUB_CFG": json.dumps(cfg)})
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    started = []
    real_run = subprocess.run

    def run(args, **kw):
        started.append({"args": args, "env": kw.get("env"), "input": kw.get("input")})
        return real_run(args, **kw)

    monkeypatch.setattr(pw.subprocess, "run", run)
    gguf = vm / "pin" / f"{spec['served_as']}-Q8_0.gguf"
    gguf.write_bytes(b"q8")
    m = {"path": str(gguf), "revision": spec["revision"], "gguf_sha256": spec["expect_sha256"] or "0" * 64}
    result, err = {}, None
    try:
        pw.upload_hosted(spec, m, result)
    except RuntimeError as e:
        err = str(e)
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    assert started, "no child was started"
    s = started[-1]
    # The token reached the child on stdin only: never its argv or its environment.
    assert SENTINEL in s["input"] and SENTINEL not in json.dumps(s["args"]) and SENTINEL not in json.dumps(s["env"])
    assert "HF_TOKEN" not in s["env"] and "HUGGING_FACE_HUB_TOKEN" not in s["env"]
    inits = [r for r in rows if r["call"] == "init" and r["token_sha"]]
    if inits:
        assert inits[0]["token_sha"] == hashlib.sha256(SENTINEL.encode()).hexdigest()
    assert pw._TOKEN is None  # this process kept none
    assert SENTINEL not in json.dumps(result) and SENTINEL not in (err or "")
    return result, rows, err


def test_the_child_uploads_with_the_licence_card_readme_and_its_hub_version(vm, monkeypatch, tmp_path):
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, files=["README.md", "LICENSE", "NOTICE", "config.json"])
    assert err is None, err
    up = result["uploaded"]
    assert up["files"] == ["LICENSE", "NOTICE", "README.md", "UPSTREAM-CARD.md", "granite-4.2-30b-Q8_0.gguf"]
    assert up["hub_version"] == "0.36.2-standin" and up["commit"] == "c" * 40 and up["licence"] == "apache-2.0"
    readme = [r for r in rows if r["call"] == "upload"][0]["readme"]
    assert GRANITE["revision"] in readme and pw.LLAMA_CPP_COMMIT in readme and GRANITE["expect_sha256"] in readme
    assert "Licence: apache-2.0" in readme
    assert not any(r["call"] == "upload_file" for r in rows)  # Mike's root card is never written
    assert not (vm / "pin" / "upload" / "granite-4.2-30b").exists()


def test_a_card_only_apache_repo_gets_the_vendored_text_checked_against_its_pin(vm, monkeypatch, tmp_path):
    vendored = Path(__file__).resolve().parents[1] / "scripts" / "colab" / "licences" / "apache-2.0.txt"
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, files=["README.md"])
    assert err is None, err
    assert [r for r in rows if r["call"] == "upload"][0]["license"] == vendored.read_bytes().decode("latin-1")
    pw.TOKEN_PATH.write_text(SENTINEL, encoding="utf-8")
    pw.read_token_file(needed=True)
    monkeypatch.setitem(pw.LICENCE_FALLBACK, "apache-2.0", ("apache-2.0.txt", "0" * 64))
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, files=["README.md"])
    assert "pinned sha256" in err and not [r for r in rows if r["call"] == "upload"]


@pytest.mark.parametrize("cfg, why", [
    ({"private": False}, "is public"),
    ({"repo_licence": None}, "names no licence"),
    ({"files": ["README.md", "LICENSE"], "card": "---\nlicense: other\nlicense_name: openmdw-1.1\n---\n"},
     "card licence is 'other'"),
])
def test_the_child_refuses_before_any_upload(vm, monkeypatch, tmp_path, cfg, why):
    # A public repo; a repo card with no licence; Lightning's OpenMDW card in the Apache repo.
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, **cfg)
    assert why in err and "REFUSED" in err and "uploaded" not in result
    assert not [r for r in rows if r["call"] in ("upload", "upload_file")]


def test_a_failing_child_fails_the_window_with_its_scrubbed_reason(vm, monkeypatch, tmp_path):
    leak = "hf_" + "LEAKEDinAnERRORmessage0123456789abcdefgh"
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, fail_upload=f"server said no to {leak}")
    assert "exited 1" in err and "[token]" in err and leak not in err


def test_a_child_that_dies_without_a_result_fails_the_window(vm, monkeypatch, tmp_path):
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, die_on_import=True)
    assert "exited 1" in err or "exited 3" in err
    assert "uploaded" not in result


def test_an_env_model_takes_the_same_child_path(vm, monkeypatch, tmp_path):
    muse = next(x for x in pw.MODELS if x["served_as"] == "muse-glimmer-30b")
    assert muse.get("env")  # Muse and Gemma convert in their own venv; their upload still goes through the child
    result, rows, err = _child_upload(vm, monkeypatch, tmp_path, spec=muse, files=["README.md", "LICENSE"])
    assert err is None, err
    assert result["uploaded"]["path"] == "muse-glimmer-30b/muse-glimmer-30b-Q8_0.gguf"


# ---- no implicit token, and no residue before any wave (R&D, from Argon's W-A reply, 2026-10-18) -----------------------

def test_the_job_disables_the_hubs_implicit_token_before_any_import():
    assert pw.os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    src = (Path(pw.__file__)).read_text(encoding="utf-8")
    assert src.index('os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"') < src.index("from huggingface_hub")


def test_a_hub_an_earlier_exec_imported_gets_the_constant_too(monkeypatch):
    const = SimpleNamespace(HF_HUB_DISABLE_IMPLICIT_TOKEN=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", const)
    pw.no_implicit_token()
    assert const.HF_HUB_DISABLE_IMPLICIT_TOKEN is True


def test_every_hub_call_names_its_token():
    """Every HfApi(), hf_hub_download() and snapshot_download() in the job and in the upload child passes token=: the
    private repo's (token) or, for a public repo, token=False. None is left to an implicit lookup."""

    import ast

    trees = [ast.parse(Path(pw.__file__).read_text(encoding="utf-8")), ast.parse(pw.UPLOAD_CHILD)]
    calls = [n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) in ("HfApi", "hf_hub_download", "snapshot_download")]
    assert len(calls) >= 8, len(calls)  # prefetch 2, upload child 3, fetch_model 3
    for c in calls:
        tok = [k.value for k in c.keywords if k.arg == "token"]
        assert tok, f"line {c.lineno}: {ast.unparse(c)[:80]} has no token="
        v = tok[0]
        assert (isinstance(v, ast.Name) and v.id == "token") or (isinstance(v, ast.Constant) and v.value is False), \
            f"line {c.lineno}: token={ast.unparse(v)}"


def test_a_maker_gguf_fetch_asks_the_hub_with_token_false(vm, monkeypatch):
    seen = []

    class Api:
        def __init__(self, token=None):
            seen.append({"api_token": token})

        def list_repo_files(self, repo, revision=None):
            return ["model-Q8_0.gguf"]

    def hf_hub_download(repo, filename, revision=None, local_dir=None, token=None, **kw):
        seen.append({"download_token": token})
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"gguf")
        return str(p)

    _hub(monkeypatch, HfApi=Api, hf_hub_download=hf_hub_download, snapshot_download=None)
    spec = {"served_as": "maker-x", "how": "maker_gguf", "repo": "maker/x-GGUF", "revision": REV, "file": "model-Q8_0.gguf"}
    pw.fetch_model(spec, {})
    assert seen == [{"api_token": False}, {"download_token": False}]


def test_a_clean_prefetch_records_token_clean_all_true(vm, monkeypatch):
    import hashlib

    content = b"gguf bytes"
    spec = {**GRANITE, "expect_sha256": hashlib.sha256(content).hexdigest()}
    monkeypatch.setenv("PIN_GGUF_REPO", "studio/gguf-private")
    monkeypatch.setenv("PIN_GGUF_REVISION", REV)
    _fake_hub(monkeypatch, content, [])
    result = {}
    pw.prefetch_hosted([spec["served_as"]], {spec["served_as"]: spec}, result)
    assert result["token_clean"] == {"file_gone": True, "value_dropped": True, "env_clear": True,
                                     "no_cached_token": True, "no_colab_secret": True}


def test_a_window_without_a_repo_checks_the_residues_too(vm):
    result = {}
    pw.prefetch_hosted([], {}, result)
    assert all(result["token_clean"].values()) and pw._TOKEN is None


@pytest.mark.parametrize("residue, key", [("env", "env_clear"), ("cache", "no_cached_token"), ("colab", "no_colab_secret"),
                                          ("file", "file_gone")])
def test_any_token_residue_stops_the_window_before_pool(vm, monkeypatch, tmp_path, residue, key):
    if residue == "env":
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "x")
    elif residue == "cache":
        home = tmp_path / "hfh"
        home.mkdir()
        (home / "token").write_text("x", encoding="utf-8")
        monkeypatch.setenv("HF_HOME", str(home))
    elif residue == "colab":
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils._auth", SimpleNamespace(_GOOGLE_COLAB_SECRET="x"))
    else:
        pw.TOKEN_PATH.write_text("x", encoding="utf-8")
    result = {}
    with pytest.raises(RuntimeError, match=rf"token residue is left before any wave \(.*{key}"):
        pw.prefetch_hosted([], {}, result)
    assert result["token_clean"][key] is False


def test_a_hub_without_the_colab_secret_attribute_reads_clean(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._auth", SimpleNamespace())
    assert pw.token_clean()["no_colab_secret"] is True
