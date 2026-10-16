import urllib.error

import pytest

from role_set.guards import GuardError
from role_set.transport import BACKOFF_S, TransportError, chat, post_with_retries


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
        return {"model": "qwen3-235b-instruct-2507", "choices": [{"message": {"content": "hello"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    t = iter([1.0, 3.5])
    c = chat("http://127.0.0.1:11435/v1/", "qwen3-235b-instruct-2507", [{"role": "user", "content": "x"}], seed=7, post=post, clock=lambda: next(t))
    assert seen["url"] == "http://127.0.0.1:11435/v1/chat/completions"
    assert seen["body"]["seed"] == 7 and seen["body"]["temperature"] == 0.7
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.seconds) == ("hello", 10, 2, 2.5)
    assert c.finish_reason == "length"
    with pytest.raises(TransportError, match="no message content"):
        chat("http://127.0.0.1:11435/v1", "m", [], post=lambda u, b, timeout_s: {"choices": []})


def test_only_the_reply_content_is_read_never_the_reasoning():
    # llama-server returns a thinking model's reasoning in reasoning_content; the gates score content only (R&D on #70).
    out = {"model": "m", "choices": [{"message": {"content": "VERDICT: supported", "reasoning_content": "VERDICT: unsupported, I think"},
                                      "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 9}}
    c = chat("http://127.0.0.1:8081/v1", "m", [{"role": "user", "content": "x"}], post=lambda u, b, timeout_s: out)
    assert c.text == "VERDICT: supported"
    # A reply that put everything in its reasoning has no content: it reaches the gates empty, never as the reasoning.
    out["choices"][0]["message"] = {"content": None, "reasoning_content": "VERDICT: supported"}
    assert chat("http://127.0.0.1:8081/v1", "m", [], post=lambda u, b, timeout_s: out).text == ""
