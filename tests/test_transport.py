import urllib.error

import pytest

from role_set.guards import GuardError
from role_set.transport import BACKOFF_S, TransportError, chat, ollama_chat, post_with_retries, unload


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:1/x", code, "err", {}, None)  # type: ignore[arg-type]


def test_retries_5xx_and_resets_with_fixed_backoff_then_gives_up():
    calls, waits, logged = [], [], []
    errors = [_http_error(503), urllib.error.URLError("reset"), TimeoutError(), _http_error(502)]

    def post(url, body, timeout_s):
        calls.append(url)
        raise errors[len(calls) - 1]

    with pytest.raises(TransportError, match="after 3 retries"):
        post_with_retries("http://127.0.0.1:11435/v1/x", {}, post=post, sleep=waits.append, log=logged.append)
    assert len(calls) == 4 and waits == list(BACKOFF_S)
    assert [e["attempt"] for e in logged] == [1, 2, 3] and logged[0]["error"] == "HTTPError"


def test_4xx_is_not_retried_and_other_hosts_are_refused():
    def post(url, body, timeout_s):
        raise _http_error(400)

    with pytest.raises(TransportError, match="HTTP 400"):
        post_with_retries("http://127.0.0.1:11435/v1/x", {}, post=post, sleep=lambda s: None)
    with pytest.raises(GuardError):
        post_with_retries("http://10.0.0.2:11435/v1/x", {}, post=post)


def test_retry_succeeds_on_a_later_attempt():
    n = {"calls": 0}

    def post(url, body, timeout_s):
        n["calls"] += 1
        if n["calls"] < 3:
            raise ConnectionError("boom")
        return {"ok": True}

    assert post_with_retries("http://127.0.0.1:11435/v1/x", {}, post=post, sleep=lambda s: None) == {"ok": True}


def test_chat_builds_an_openai_request_and_reads_usage():
    seen = {}

    def post(url, body, timeout_s):
        seen.update(url=url, body=body)
        return {"model": "qwen3-235b-instruct-2507", "choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    t = iter([1.0, 3.5])
    c = chat("http://127.0.0.1:11435/v1/", "qwen3-235b-instruct-2507", [{"role": "user", "content": "x"}], seed=7, post=post, clock=lambda: next(t))
    assert seen["url"] == "http://127.0.0.1:11435/v1/chat/completions"
    assert seen["body"]["seed"] == 7 and seen["body"]["temperature"] == 0.7
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.seconds) == ("hello", 10, 2, 2.5)
    with pytest.raises(TransportError, match="no message content"):
        chat("http://127.0.0.1:11435/v1", "m", [], post=lambda u, b, timeout_s: {"choices": []})


def test_ollama_chat_and_unload():
    seen = {}

    def post(url, body, timeout_s):
        seen.update(url=url, body=body)
        return {"model": "gemma4:31b", "message": {"content": "VERDICT: supported"}, "prompt_eval_count": 50, "eval_count": 5}

    c = ollama_chat("http://127.0.0.1:11434", "gemma4:31b", [{"role": "user", "content": "x"}], post=post)
    assert seen["url"] == "http://127.0.0.1:11434/api/chat" and seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["keep_alive"] == "30m" and seen["body"]["think"] is False
    assert c.text == "VERDICT: supported" and c.prompt_tokens == 50
    with pytest.raises(TransportError, match="no message content"):
        ollama_chat("http://127.0.0.1:11434", "m", [], post=lambda u, b, timeout_s: {"message": None})

    unloaded = {}
    unload("http://127.0.0.1:11434", "gemma4:31b", post=lambda u, b, timeout_s: unloaded.update(b))
    assert unloaded["keep_alive"] == 0

    def failing(u, b, timeout_s):
        raise TransportError("down")

    unload("http://127.0.0.1:11434", "gemma4:31b", post=failing)  # never raises at the end of a window
