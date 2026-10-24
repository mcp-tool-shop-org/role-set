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

from . import cards
from .guards import GuardError
from .transport import Completion

JOURNAL_FILE = "calls-journal.jsonl"
RETRY_SEED_BASE = 10_000
RETRY_SEED_STRIDE = 100
RETRY_SEED_BOUND = 100


def retry_seed(first_seed: int | None = 0, ask: int = 1) -> int:
    """Compute the disjoint retry seed `10_000 + first_seed * 100 + ask` for a VOID retry (`ask > 0`).

    Where the first ask (`ask == 0`) sent no `seed` kwarg at all, `first_seed` is `0` (or `None`, treated as `0`).
    The scheme is disjoint only while every `first_seed` is below 100 (`0 <= first_seed < 100`) and every `ask` is
    below 100 (`1 <= ask < 100`). Breaking either bound raises `ValueError` loudly (never clamping) so a later seed
    sweep or ask cap cannot collide silently.
    """
    base = 0 if first_seed is None else first_seed
    if isinstance(base, bool) or not isinstance(base, int) or not (0 <= base < RETRY_SEED_BOUND):
        raise ValueError(
            f"REFUSED: retry_seed requires 0 <= first_seed < {RETRY_SEED_BOUND}, got {first_seed!r}"
        )
    if isinstance(ask, bool) or not isinstance(ask, int) or not (1 <= ask < RETRY_SEED_BOUND):
        raise ValueError(
            f"REFUSED: retry_seed requires 1 <= ask < {RETRY_SEED_BOUND}, got {ask!r}"
        )
    return RETRY_SEED_BASE + base * RETRY_SEED_STRIDE + ask


def request_key(
    role: str,
    args: tuple,
    kwargs: dict,
    *,
    verdict_prompt: str | None = None,
    cannot_tell_construction: str | None = None,
) -> str:
    payload: dict = {"role": role, "args": args, "kwargs": kwargs}
    if verdict_prompt not in (None, "old"):
        payload["verdict_prompt"] = verdict_prompt
    if cannot_tell_construction not in (None, "default"):
        payload["cannot_tell_construction"] = cannot_tell_construction
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def messages_sha256(messages: object) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _completion_dict(c: Completion) -> dict:
    out = {"text": c.text, "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens, "model": c.model,
           "seconds": c.seconds, "finish_reason": getattr(c, "finish_reason", None)}
    if getattr(c, "thinking_tokens", None) is not None:
        out["thinking_tokens"] = c.thinking_tokens
    return out


def load(
    path: Path,
    *,
    expected_flaw_prompt: str | None = None,
    expected_verdict_prompt: str | None = None,
    expected_checks: str | None = None,
    expected_cannot_tell_construction: str | None = None,
) -> dict[str, deque]:
    """An earlier journal, as each request key's completions in journal order. A torn last line (the VM went mid-write)
    is left out: its call runs again. When `expected_flaw_prompt`, `expected_verdict_prompt`, `expected_checks` or
    `expected_cannot_tell_construction` is given, any journal entry (or adjacent report) written under a different
    setting refuses the resume."""

    if (
        expected_flaw_prompt is not None
        or expected_verdict_prompt is not None
        or expected_checks is not None
        or expected_cannot_tell_construction is not None
    ):
        if not path.is_file():
            raise GuardError(f"REFUSED: cannot read call journal {path}")
        rep_path = path.parent / "critic-eval-report.json"
        if rep_path.is_file():
            try:
                rep_data = json.loads(rep_path.read_text(encoding="utf-8"))
            except ValueError:
                rep_data = {}
            if isinstance(rep_data, dict) and (
                "flaw_prompt" in rep_data
                or "verdict_prompt" in rep_data
                or "checks" in rep_data
                or "critic_model" in rep_data
            ):
                if expected_flaw_prompt is not None:
                    rep_fp = str(rep_data.get("flaw_prompt") or "old")
                    if rep_fp != expected_flaw_prompt:
                        raise GuardError(
                            f"REFUSED: call journal {path} was written under flaw_prompt={rep_fp!r}, "
                            f"cannot reuse under flaw_prompt={expected_flaw_prompt!r}"
                        )
                if expected_verdict_prompt is not None:
                    rep_vp = str(rep_data.get("verdict_prompt") or "old")
                    if rep_vp != expected_verdict_prompt:
                        raise GuardError(
                            f"REFUSED: call journal {path} was written under verdict_prompt={rep_vp!r}, "
                            f"cannot reuse under verdict_prompt={expected_verdict_prompt!r}"
                        )
                if expected_checks is not None:
                    rep_ck = str(rep_data.get("checks") or "all")
                    if rep_ck != expected_checks:
                        raise GuardError(
                            f"REFUSED: call journal {path} was written under checks={rep_ck!r}, "
                            f"cannot reuse under checks={expected_checks!r}"
                        )

    reuse: dict[str, deque] = defaultdict(deque)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if expected_flaw_prompt is not None:
            rec_fp = str(d.get("flaw_prompt") or "old")
            if rec_fp != expected_flaw_prompt:
                raise GuardError(
                    f"REFUSED: call journal {path} was written under flaw_prompt={rec_fp!r}, "
                    f"cannot reuse under flaw_prompt={expected_flaw_prompt!r}"
                )
        if expected_verdict_prompt is not None:
            rec_vp = str(d.get("verdict_prompt") or "old")
            if rec_vp != expected_verdict_prompt:
                raise GuardError(
                    f"REFUSED: call journal {path} was written under verdict_prompt={rec_vp!r}, "
                    f"cannot reuse under verdict_prompt={expected_verdict_prompt!r}"
                )
        if expected_checks is not None:
            rec_ck = str(d.get("checks") or "all")
            if rec_ck != expected_checks:
                raise GuardError(
                    f"REFUSED: call journal {path} was written under checks={rec_ck!r}, "
                    f"cannot reuse under checks={expected_checks!r}"
                )
        if expected_cannot_tell_construction is not None:
            rec_ctc = str(d.get("cannot_tell_construction") or "default")
            if rec_ctc != expected_cannot_tell_construction:
                raise GuardError(
                    f"REFUSED: call journal {path} was written under cannot_tell_construction={rec_ctc!r}, "
                    f"cannot reuse under cannot_tell_construction={expected_cannot_tell_construction!r}"
                )
        c = d["completion"]
        reuse[d["key"]].append(Completion(text=c["text"], prompt_tokens=c.get("prompt_tokens"),
                                          completion_tokens=c.get("completion_tokens"), model=c.get("model") or "",
                                          seconds=c.get("seconds") or 0.0, finish_reason=c.get("finish_reason"),
                                          thinking_tokens=c.get("thinking_tokens"),
                                          messages_sha256=d.get("messages_sha256")))
    return reuse


class CallJournal:
    """Wraps the writer and the critic: journals each finished call and, on a resume, replays the earlier ones."""

    def __init__(
        self,
        path: Path,
        reuse: dict[str, deque] | None = None,
        *,
        flaw_prompt: str | None = None,
        verdict_prompt: str | None = None,
        checks: str | None = None,
        cannot_tell_construction: str | None = None,
    ):
        if (
            (
                flaw_prompt is not None
                or verdict_prompt is not None
                or checks is not None
                or cannot_tell_construction is not None
            )
            and path.is_file()
            and path.stat().st_size > 0
        ):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    if flaw_prompt is not None:
                        rec_fp = str(d.get("flaw_prompt") or "old")
                        if rec_fp != flaw_prompt:
                            raise GuardError(
                                f"REFUSED: call journal {path} was written under flaw_prompt={rec_fp!r}, "
                                f"cannot append under flaw_prompt={flaw_prompt!r}"
                            )
                    if verdict_prompt is not None:
                        rec_vp = str(d.get("verdict_prompt") or "old")
                        if rec_vp != verdict_prompt:
                            raise GuardError(
                                f"REFUSED: call journal {path} was written under verdict_prompt={rec_vp!r}, "
                                f"cannot append under verdict_prompt={verdict_prompt!r}"
                            )
                    if checks is not None:
                        rec_ck = str(d.get("checks") or "all")
                        if rec_ck != checks:
                            raise GuardError(
                                f"REFUSED: call journal {path} was written under checks={rec_ck!r}, "
                                f"cannot append under checks={checks!r}"
                            )
                    if cannot_tell_construction is not None:
                        rec_ctc = str(d.get("cannot_tell_construction") or "default")
                        if rec_ctc != cannot_tell_construction:
                            raise GuardError(
                                f"REFUSED: call journal {path} was written under cannot_tell_construction={rec_ctc!r}, "
                                f"cannot append under cannot_tell_construction={cannot_tell_construction!r}"
                            )
        self.path = path
        self.reuse = reuse or {}
        self.flaw_prompt = flaw_prompt
        self.verdict_prompt = verdict_prompt
        self.checks = checks
        self.cannot_tell_construction = cannot_tell_construction
        self.lock = threading.Lock()
        self.counts = {"reused": 0, "called": 0}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def _append(
        self,
        key: str,
        role: str,
        c: Completion,
        reused: bool,
        *,
        check: str | None = None,
        messages_sha256: str | None = None,
        slot: str | None = None,
        attempt: int | None = None,
    ) -> None:
        payload: dict = {
            "key": key,
            "role": role,
            "reused": reused,
            "messages_sha256": messages_sha256,
            "slot": slot,
            "attempt": attempt,
        }
        if self.flaw_prompt is not None:
            payload["flaw_prompt"] = self.flaw_prompt
        if self.verdict_prompt is not None:
            payload["verdict_prompt"] = self.verdict_prompt
        if self.checks is not None:
            payload["checks"] = self.checks
        if self.cannot_tell_construction is not None:
            payload["cannot_tell_construction"] = self.cannot_tell_construction
        if check is not None:
            payload["check"] = check
        payload["completion"] = _completion_dict(c)
        line = json.dumps(payload, ensure_ascii=False)
        with self.lock, self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()

    def wrap(self, role: str, fn, *, card: dict | None = None):
        if fn is None:
            return None
        eff_card = card if card is not None else getattr(fn, "card", None)

        def call(*args, **kwargs):
            check = kwargs.pop("check", None)
            slot = kwargs.pop("slot", None)
            attempt = kwargs.pop("attempt", None)
            key = request_key(
                role,
                args,
                kwargs,
                verdict_prompt=self.verdict_prompt,
                cannot_tell_construction=self.cannot_tell_construction,
            )
            if slot is None:
                slot = kwargs.get("slot_id")
            if attempt is None:
                attempt = kwargs.get("seed")
            messages = args[0] if args else kwargs.get("messages")
            sent = cards.apply(eff_card, messages) if (eff_card is not None and messages is not None) else messages
            default_sha = messages_sha256(sent) if sent is not None else None
            with self.lock:
                queue = self.reuse.get(key)
                c = queue.popleft() if queue else None
                if c is not None:
                    self.counts["reused"] += 1
            if c is not None:
                msg_sha = default_sha if eff_card is not None else (getattr(c, "messages_sha256", None) or default_sha)
                self._append(key, role, c, True, check=check, messages_sha256=msg_sha, slot=slot, attempt=attempt)
                return c
            c = fn(*args, **kwargs)
            with self.lock:
                self.counts["called"] += 1
            msg_sha = getattr(c, "messages_sha256", None) or default_sha
            self._append(key, role, c, False, check=check, messages_sha256=msg_sha, slot=slot, attempt=attempt)
            return c

        call.card = eff_card
        call.takes_slot_id = getattr(fn, "takes_slot_id", False)  # the keyed mock is told the slot (Runner._writer_call)
        call.takes_check = True
        return call

    def summary(self) -> dict:
        left = sum(len(q) for q in self.reuse.values())
        out = {"journal": self.path.name, **self.counts, "unused_from_reuse": left}
        if self.flaw_prompt is not None:
            out["flaw_prompt"] = self.flaw_prompt
        if self.verdict_prompt is not None:
            out["verdict_prompt"] = self.verdict_prompt
        if self.checks is not None:
            out["checks"] = self.checks
        if self.cannot_tell_construction not in (None, "default"):
            out["cannot_tell_construction"] = self.cannot_tell_construction
        return out
