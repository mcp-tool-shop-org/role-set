import time
from pathlib import Path

import pytest

from role_set.guards import (
    ollama_digest,
    require_ollama_digest,
    GuardError,
    heartbeat_age_s,
    require_grant,
    require_heartbeat,
    require_local_endpoint,
    require_model_allowed,
)


def test_only_loopback_http_with_a_port_is_accepted():
    assert require_local_endpoint(" http://127.0.0.1:11435/v1 ") == "http://127.0.0.1:11435/v1"
    with pytest.raises(GuardError, match="host"):
        require_local_endpoint("http://localhost:11435/v1")
    with pytest.raises(GuardError, match="host"):
        require_local_endpoint("http://10.0.0.5:11435/v1")
    with pytest.raises(GuardError, match="plain http"):
        require_local_endpoint("https://127.0.0.1:11435/v1")
    with pytest.raises(GuardError, match="port"):
        require_local_endpoint("http://127.0.0.1/v1")
    with pytest.raises(GuardError):
        require_local_endpoint(None)  # type: ignore[arg-type]


def test_writer_must_be_allowed_and_never_cloud():
    allowed = ["Qwen/Qwen3-235B-A22B-Instruct-2507"]
    assert require_model_allowed(allowed[0], allowed) == allowed[0]
    with pytest.raises(GuardError, match="cloud"):
        require_model_allowed("qwen3:235b-cloud", allowed)
    with pytest.raises(GuardError, match="licence-checked"):
        require_model_allowed("some/other-model", allowed)
    with pytest.raises(GuardError, match="required"):
        require_model_allowed("   ", allowed)


def test_grant_is_required_only_for_the_card():
    assert require_grant(None, uses_card=False) is None
    assert require_grant(" window-3 ", uses_card=False) == "window-3"
    assert require_grant(" window-3 ", uses_card=True) == "window-3"
    with pytest.raises(GuardError, match="--grant"):
        require_grant("", uses_card=True)


def test_heartbeat_uses_mtime_and_never_opens_the_file(tmp_path: Path):
    hb = tmp_path / "heartbeat"
    assert heartbeat_age_s(hb, time.time()) == float("inf")
    with pytest.raises(GuardError, match="missing"):
        require_heartbeat(hb, time.time())
    hb.write_text("x")
    now = hb.stat().st_mtime
    assert require_heartbeat(hb, now + 5) == pytest.approx(5)
    with pytest.raises(GuardError, match="old"):
        require_heartbeat(hb, now + 61)


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        import json

        return json.dumps(self.payload).encode("utf-8")


def test_ollama_digest_is_checked_against_the_pin_and_cloud_tags_are_refused():
    tags = {"models": [{"name": "gemma4:31b", "digest": "6316f062" + "0" * 56}, {"name": "gemma4:31b-cloud", "digest": "x"}]}
    opener = lambda url, timeout: _Resp(tags)  # noqa: E731
    assert ollama_digest("http://127.0.0.1:11434", "gemma4:31b", opener=opener).startswith("6316f062")
    assert ollama_digest("http://127.0.0.1:11434", "missing:1b", opener=opener) is None
    assert require_ollama_digest("http://127.0.0.1:11434", "gemma4:31b", "6316f062" + "0" * 56, opener=opener).startswith("6316f062")
    with pytest.raises(GuardError, match="differs from the pinned"):
        require_ollama_digest("http://127.0.0.1:11434", "gemma4:31b", "deadbeef" + "0" * 56, opener=opener)
    with pytest.raises(GuardError, match="not present"):
        require_ollama_digest("http://127.0.0.1:11434", "missing:1b", "x", opener=opener)
    with pytest.raises(GuardError, match="cloud"):
        require_ollama_digest("http://127.0.0.1:11434", "gemma4:31b-cloud", "x", opener=opener)
    with pytest.raises(GuardError, match="host"):
        ollama_digest("http://ollama.example:11434", "gemma4:31b", opener=opener)
