"""The call journal and its replay: a resumed wave skips the model calls that already finished (R&D, 2026-10-17).

Every writer and critic call that returns is appended to the wave's call journal as soon as it returns: the request's
key and the completion, whatever its finish reason (a truncated call is kept as its outcome). A wave run with
--reuse-calls <an earlier journal> answers each request found there from the journal instead of the model, in the order
the journal holds them, and asks the model only for the rest. The wave's own logic runs again over the same
completions, so every state it builds (accept order, spent framings, the andon) comes out as one uninterrupted run
would make it. A call the cap cut off never reached the journal, so its slot runs again.

The replay is exact at --concurrency 1, the default and what every Colab window runs: the spent framings a prompt carries
depend on the order slots are accepted, and only one at a time fixes that order. At a higher concurrency a request can
differ from the cut run's and is simply asked again; the result's unused_from_reuse then counts the journaled calls left
over, and the window script reports any as a warning.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict, deque
from pathlib import Path

from .transport import Completion

JOURNAL_FILE = "calls-journal.jsonl"


def request_key(role: str, args: tuple, kwargs: dict) -> str:
    blob = json.dumps({"role": role, "args": args, "kwargs": kwargs}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _completion_dict(c: Completion) -> dict:
    out = {"text": c.text, "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens, "model": c.model,
           "seconds": c.seconds, "finish_reason": getattr(c, "finish_reason", None)}
    if getattr(c, "thinking_tokens", None) is not None:
        out["thinking_tokens"] = c.thinking_tokens
    return out


def load(path: Path) -> dict[str, deque]:
    """An earlier journal, as each request key's completions in journal order. A torn last line (the VM went mid-write)
    is left out: its call runs again."""

    reuse: dict[str, deque] = defaultdict(deque)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        c = d["completion"]
        reuse[d["key"]].append(Completion(text=c["text"], prompt_tokens=c.get("prompt_tokens"),
                                          completion_tokens=c.get("completion_tokens"), model=c.get("model") or "",
                                          seconds=c.get("seconds") or 0.0, finish_reason=c.get("finish_reason"),
                                          thinking_tokens=c.get("thinking_tokens")))
    return reuse


class CallJournal:
    """Wraps the writer and the critic: journals each finished call and, on a resume, replays the earlier ones."""

    def __init__(self, path: Path, reuse: dict[str, deque] | None = None):
        self.path = path
        self.reuse = reuse or {}
        self.lock = threading.Lock()
        self.counts = {"reused": 0, "called": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _append(self, key: str, role: str, c: Completion, reused: bool) -> None:
        line = json.dumps({"key": key, "role": role, "reused": reused, "completion": _completion_dict(c)}, ensure_ascii=False)
        with self.lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()

    def wrap(self, role: str, fn):
        if fn is None:
            return None

        def call(*args, **kwargs):
            key = request_key(role, args, kwargs)
            with self.lock:
                queue = self.reuse.get(key)
                c = queue.popleft() if queue else None
                if c is not None:
                    self.counts["reused"] += 1
            if c is not None:
                self._append(key, role, c, True)
                return c
            c = fn(*args, **kwargs)
            with self.lock:
                self.counts["called"] += 1
            self._append(key, role, c, False)
            return c

        call.takes_slot_id = getattr(fn, "takes_slot_id", False)  # the keyed mock is told the slot (Runner._writer_call)
        return call

    def summary(self) -> dict:
        left = sum(len(q) for q in self.reuse.values())
        return {"journal": self.path.name, **self.counts, "unused_from_reuse": left}
