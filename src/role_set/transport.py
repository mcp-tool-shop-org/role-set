"""The only way out of this machine: an OpenAI-compatible chat endpoint on the loopback.

The writer sits behind offrig's SSH tunnel on 127.0.0.1 (the lane's port), and the
critic behind the local Ollama on 127.0.0.1:11434. Both are reached through this
module, which refuses any other host before a byte is sent. Transport errors are
retried three times with 2 s, 8 s and 32 s waits, as the design's table says;
anything else is raised to the runner, which stops.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from role_set.guards import GuardError, require_local_endpoint

BACKOFF_S = (2.0, 8.0, 32.0)
DEFAULT_TIMEOUT_S = 600.0


class TransportError(RuntimeError):
    """The endpoint failed after the retries. The runner records it and stops the slot."""


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    model: str
    seconds: float


def _post(url: str, body: dict, timeout_s: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - loopback only, checked above
        return json.loads(resp.read().decode("utf-8"))


def post_with_retries(url: str, body: dict, *, timeout_s: float = DEFAULT_TIMEOUT_S, sleep=time.sleep, post=_post, log=None) -> dict:
    """POST, retrying a 5xx, a reset or a timeout three times with fixed backoff. A 4xx is not retried."""

    require_local_endpoint(url)
    last: Exception | None = None
    for attempt in range(len(BACKOFF_S) + 1):
        try:
            return post(url, body, timeout_s)
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise TransportError(f"endpoint refused the request: HTTP {e.code}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = e
        if attempt < len(BACKOFF_S):
            wait = BACKOFF_S[attempt]
            if log:
                log({"action": "retry_transport", "attempt": attempt + 1, "wait_s": wait, "error": type(last).__name__})
            sleep(wait)
    raise TransportError(f"endpoint failed after {len(BACKOFF_S)} retries: {type(last).__name__}: {last}")


def chat(
    endpoint: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    seed: int | None = None,
    max_tokens: int = 3500,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    post=post_with_retries,
    clock=time.monotonic,
) -> Completion:
    """One chat completion against an OpenAI-compatible /v1 endpoint (SGLang on the pod)."""

    url = require_local_endpoint(endpoint).rstrip("/") + "/chat/completions"
    body: dict = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    t0 = clock()
    out = post(url, body, timeout_s=timeout_s)
    try:
        text = out["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise TransportError("endpoint returned no message content") from e
    usage = out.get("usage") or {}
    return Completion(
        text=text or "",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        model=str(out.get("model") or model),
        seconds=clock() - t0,
    )


def ollama_chat(
    endpoint: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.0,
    seed: int = 0,
    num_ctx: int = 16384,
    keep_alive: str | int = "30m",
    think: bool = False,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    post=post_with_retries,
    clock=time.monotonic,
) -> Completion:
    """One chat against the local Ollama's /api/chat (the critic). A cloud tag is refused by the model guard upstream."""

    url = require_local_endpoint(endpoint).rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": num_ctx},
    }
    t0 = clock()
    out = post(url, body, timeout_s=timeout_s)
    try:
        text = out["message"]["content"]
    except (KeyError, TypeError) as e:
        raise TransportError("ollama returned no message content") from e
    return Completion(
        text=text or "",
        prompt_tokens=out.get("prompt_eval_count"),
        completion_tokens=out.get("eval_count"),
        model=str(out.get("model") or model),
        seconds=clock() - t0,
    )


def unload(endpoint: str, model: str, *, post=post_with_retries) -> None:
    """`keep_alive: 0` at the end of the window, so the card is left empty."""

    url = require_local_endpoint(endpoint).rstrip("/") + "/api/chat"
    try:
        post(url, {"model": model, "messages": [], "stream": False, "keep_alive": 0}, timeout_s=60.0)
    except (TransportError, GuardError):
        pass
