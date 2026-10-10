"""Guards that stop a run before it sends a prompt or touches the card.

role-set never plans, launches or shuts down a pod. It takes an endpoint on the
loopback and a model id, and refuses everything else. The heartbeat check reads
the file's modified time and never opens it.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK = "127.0.0.1"
HEARTBEAT_MAX_AGE_S = 60
CLOUD_MARKERS = ("cloud",)


class GuardError(Exception):
    """A run is refused. Nothing is sent."""


def require_local_endpoint(url: str) -> str:
    """Only http://127.0.0.1:<port>/... is accepted: offrig's tunnel or the local Ollama."""

    parsed = urlparse(url.strip() if isinstance(url, str) else "")
    if parsed.scheme != "http":
        raise GuardError(f"REFUSED: endpoint must be plain http on the loopback, got {url!r}")
    if parsed.hostname != LOOPBACK:
        raise GuardError(f"REFUSED: endpoint host must be {LOOPBACK}, got {parsed.hostname!r}")
    if parsed.port is None:
        raise GuardError("REFUSED: endpoint needs an explicit port (the lane's tunnel port)")
    return url.strip()


def require_model_allowed(model: str, allowed: list[str]) -> str:
    """The writer must be on the licence-checked list, and never a cloud tag."""

    name = model.strip() if isinstance(model, str) else ""
    if not name:
        raise GuardError("REFUSED: a model id is required")
    lowered = name.lower()
    for marker in CLOUD_MARKERS:
        if marker in lowered:
            raise GuardError(f"REFUSED: cloud model tags are never used ({name})")
    if name not in allowed:
        raise GuardError(f"REFUSED: {name} is not on the licence-checked writer list")
    return name


def require_grant(grant: str | None, *, uses_card: bool) -> str | None:
    """The critic on the 5090 runs only inside a Publisher grant."""

    if not uses_card:
        return grant.strip() if isinstance(grant, str) and grant.strip() else None
    if not isinstance(grant, str) or not grant.strip():
        raise GuardError("REFUSED: a 5090 step requires --grant")
    return grant.strip()


def heartbeat_age_s(path: Path, now: float) -> float:
    """Age of the watchdog heartbeat from its modified time. The file is not opened."""

    try:
        modified = path.stat().st_mtime
    except OSError:
        return float("inf")
    return now - modified


def require_heartbeat(path: Path, now: float, *, max_age_s: float = HEARTBEAT_MAX_AGE_S) -> float:
    age = heartbeat_age_s(path, now)
    if age > max_age_s:
        shown = "missing" if age == float("inf") else f"{age:.0f}s"
        raise GuardError(f"REFUSED: watchdog heartbeat is {shown} old")
    return age


def ollama_digest(endpoint: str, model: str, *, opener=None) -> str | None:
    """The local Ollama's digest for a tag, from /api/tags. Loopback only."""

    url = require_local_endpoint(endpoint).rstrip("/") + "/api/tags"
    open_ = opener or urllib.request.urlopen
    with open_(url, timeout=30) as resp:  # noqa: S310 - loopback only, checked above
        data = json.loads(resp.read().decode("utf-8"))
    for m in data.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return m.get("digest")
    return None


def require_ollama_digest(endpoint: str, model: str, expected: str, *, opener=None) -> str:
    """A re-pull can change the weights under the same tag, so the critic's digest must match its pin."""

    if "cloud" in model.lower():
        raise GuardError(f"REFUSED: cloud model tags are never used ({model})")
    got = ollama_digest(endpoint, model, opener=opener)
    if got is None:
        raise GuardError(f"REFUSED: {model} is not present on the local Ollama")
    if got != expected:
        raise GuardError(f"REFUSED: {model} digest {got[:12]} differs from the pinned {expected[:12]}; re-check the licence and re-pin")
    return got
