"""Replay client for the batching-drift study (DESIGN-v1.md §3, §4, §5, §8 item 1).

Replays `requests-60.jsonl` in index order (0..59) with N in {1, 2, 4, 8} requests in flight across the six
arms (A1, A2, B2, B4, B8, B4r) through rnd's logging proxy, tagging every row of `calls-{arm}.jsonl` with `key`
and `request_index`, and joining `generations-{arm}.jsonl` by `call_index == request_index + 1` (never by line
position or completion order).

Can run either as `PIN_ROLE=batching-drift` inside `scripts/colab/pin_window.py` or as a standalone
`PIN_JOB_SCRIPT` (`python scripts/batchdrift/replay.py`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import dataclasses
import hashlib
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts" / "colab",
    REPO_ROOT / "scripts" / "batchdrift",
    REPO_ROOT / "tests",
):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_requests60 import canonical_payload_sha256, file_sha256  # noqa: E402
from role_set import callcache, transport  # noqa: E402
from role_set.mock import MockCritic  # noqa: E402

REQUESTS60_DEFAULT_PATH = Path(__file__).resolve().parent / "requests-60.jsonl"
REQUESTS60_SHA256 = "150576231c6b0792760f568af5eec2843c03d8b7a491024250f2f50efc447cb6"
REQUIRED_ARMS: tuple[tuple[str, int], ...] = (
    ("A1", 1),
    ("A2", 1),
    ("B2", 2),
    ("B4", 4),
    ("B8", 8),
    ("B4r", 4),
)
PER_SLOT_CTX = 32768
BATCHDRIFT_EXTRA_ARGS: tuple[str, ...] = ("--cache-ram", "0", "--slot-prompt-similarity", "0.0")
DEFAULT_CRITIC = "muse-glimmer-30b"
CRITIC_MAX_TOKENS = 16384
RESUME_ENV_VARS: tuple[str, ...] = (
    "PIN_RESUME_WAVE",
    "PIN_RESUME_DIR",
    "PIN_RESUME_CALLS_DIR",
    "PIN_RESUME_PRIOR_DIR",
    "PIN_CRITIC_EVAL_RESUME",
)
EXPECTED_CARD_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "cache_prompt": False,
}
EXPECTED_CARD_KWARGS = {"reasoning_strength": "high"}


def require_no_resume_env() -> None:
    """Batching-drift forbids journal replay; refuse if any resume env var is set (DESIGN-v1.md §5 check 3)."""

    active = [k for k in RESUME_ENV_VARS if (os.environ.get(k) or "").strip()]
    if active:
        raise RuntimeError(f"REFUSED: batching-drift forbids journal replay, but resume env var(s) are set: {', '.join(active)}")


def validate_request_row(row: dict, expected_index: int) -> None:
    """Validate one row of requests-60.jsonl (request_index, seed, key, messages_sha256, payload_sha256)."""

    if row.get("request_index") != expected_index:
        raise RuntimeError(
            f"REFUSED: row {expected_index} has request_index={row.get('request_index')!r}, expected {expected_index}"
        )
    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RuntimeError(f"REFUSED: request_index {expected_index} has missing or invalid seed {seed!r}")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RuntimeError(f"REFUSED: request_index {expected_index} has empty or non-list messages")
    got_msg_sha = callcache.messages_sha256(msgs)
    if got_msg_sha != row.get("messages_sha256"):
        raise RuntimeError(
            f"REFUSED: request_index {expected_index} messages_sha256 mismatch: {got_msg_sha} != {row.get('messages_sha256')}"
        )
    got_key = callcache.request_key("critic", (msgs,), {})
    if got_key != row.get("key"):
        raise RuntimeError(
            f"REFUSED: request_index {expected_index} key mismatch: {got_key} != {row.get('key')}"
        )
    got_payload_sha = canonical_payload_sha256(msgs, seed)
    if row.get("payload_sha256") and got_payload_sha != row.get("payload_sha256"):
        raise RuntimeError(
            f"REFUSED: request_index {expected_index} payload_sha256 mismatch: {got_payload_sha} != {row.get('payload_sha256')}"
        )


def load_and_verify_requests60(
    path: Path,
    *,
    expected_sha256: str | None = REQUESTS60_SHA256,
    expected_count: int | None = 60,
) -> tuple[list[dict], str]:
    """Load requests-60.jsonl, verify its sha256 pin, and validate every row's index, seed, and hashes."""

    if not path.is_file():
        raise RuntimeError(f"REFUSED: requests file not found: {path}")
    got_sha = file_sha256(path)
    if expected_sha256 is not None and got_sha.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"REFUSED: {path.name} sha256 {got_sha} != pinned {expected_sha256}"
        )
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if expected_count is not None and len(rows) != expected_count:
        raise RuntimeError(f"REFUSED: {path.name} has {len(rows)} rows, expected {expected_count}")
    for idx, r in enumerate(rows):
        validate_request_row(r, idx)
    return rows, got_sha


def build_stub_answers(requests: list[dict]) -> dict:
    """Populate MockCritic.answers from requests-60.jsonl so StubLlama is stateless across concurrent slots."""

    answers: dict = {}
    for r in requests:
        msgs = r.get("messages") or []
        user = msgs[-1].get("content") or "" if msgs else ""
        check = r.get("check")
        if check in ("verdict", "twin_verdict"):
            stmt_i = user.find("Statement: ")
            mat_i = user.find("Material:\n")
            statement = user[stmt_i + len("Statement: "):].split("\n", 1)[0].strip() if stmt_i >= 0 else ""
            first_line = user[mat_i + len("Material:\n"):].split("\n", 1)[0].strip() if mat_i >= 0 else ""
            kv = r.get("key_verdict") or "supported"
            kd = list(r.get("key_deciding") or [])
            if not kd and kv != "cannot_tell" and first_line:
                kd = [first_line]
            answers[(statement, first_line)] = (kv, kd)
        elif check == "flaw":
            trace_i = user.find("The other checker wrote: ")
            trace = user[trace_i + len("The other checker wrote: "):].split("\n", 1)[0].strip() if trace_i >= 0 else ""
            answers[("flaw", trace)] = r.get("key_flaw") or "none"
    return answers


class _FallbackProxyState:
    """Stdlib logging proxy state mirroring mcptoolshop_rnd.localbuilders.proxy.ProxyState for CPU stub tests
    when mcptoolshop_rnd is not installed in the role-set test environment."""

    def __init__(
        self,
        *,
        upstream_url: str,
        config: dict | str | Path,
        generations_path: str | Path,
        candidate: str | None = None,
        max_calls: int = 10**6,
        wall_limit_s: float = 86400.0,
    ) -> None:
        self.upstream_url = upstream_url.rstrip("/")
        self.generations_path = Path(generations_path)
        self.generations_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.generations_path.exists():
            self.generations_path.write_text("", encoding="utf-8")
        self.max_calls = int(max_calls)
        self.wall_limit_s = float(wall_limit_s)
        self._lock = threading.Lock()
        self.call_count = 0

        if isinstance(config, (str, Path)):
            raw = json.loads(Path(config).read_text(encoding="utf-8"))
        else:
            raw = copy.deepcopy(config)
        if "candidates" in raw:
            cands = raw["candidates"]
            entry = next(
                (c for c in cands if candidate is None or c.get("candidate") == candidate or c.get("arm") == candidate),
                cands[0],
            )
        else:
            entry = raw
        self.arm = str(entry.get("candidate") or entry.get("arm") or candidate or "critic")
        self.sampling = copy.deepcopy(entry.get("sampling") or {})
        self.chat_template_kwargs = copy.deepcopy(entry.get("chat_template_kwargs") or {})
        self.transforms = [t for t in (entry.get("transforms") or []) if t != "reattach_thinking"]
        self.injected = {
            "sampling": copy.deepcopy(self.sampling),
            "chat_template_kwargs": copy.deepcopy(self.chat_template_kwargs),
            "transforms": list(self.transforms),
            "reattach_max_turns": entry.get("reattach_max_turns"),
        }

    def handle_completion(self, path: str, body: dict) -> tuple[int, dict]:
        with self._lock:
            prompt_idx = self.call_count
            self.call_count += 1
            call_number = self.call_count

        up_body = copy.deepcopy(body)
        up_body["stream"] = False
        up_body["logprobs"] = True
        up_body["top_logprobs"] = 5
        for k, v in self.sampling.items():
            up_body[k] = copy.deepcopy(v)
        up_body["chat_template_kwargs"] = copy.deepcopy(self.chat_template_kwargs)

        req = urllib.request.Request(
            f"{self.upstream_url}/{path.lstrip('/')}",
            data=json.dumps(up_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.wall_limit_s) as resp:  # noqa: S310 - loopback
            status = resp.status
            up_json = json.loads(resp.read().decode("utf-8"))
        wall_s = time.monotonic() - t0

        with self._lock:
            choices = up_json.get("choices") or [{}]
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content")
            lp_content = ((choices[0].get("logprobs") or {}).get("content")) or []
            tokens = [s.get("token") for s in lp_content]
            top = [[[t.get("token"), t.get("logprob")] for t in (s.get("top_logprobs") or [])] for s in lp_content]
            usage = up_json.get("usage") or {}
            timings = up_json.get("timings") or {}
            finish_reason = choices[0].get("finish_reason")
            rec = {
                "kind": "result",
                "arm": self.arm,
                "prompt": prompt_idx,
                "call_index": call_number,
                "response": content,
                "reasoning_content": reasoning,
                "tokens": tokens,
                "top": top,
                "logprobs": choices[0].get("logprobs"),
                "timings": timings,
                "usage": usage,
                "wall_time_s": round(wall_s, 6),
                "thinking_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or len(tokens),
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "transforms": list(self.transforms),
                "injected": copy.deepcopy(self.injected),
            }
            with self.generations_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            stripped = copy.deepcopy(up_json)
            if stripped.get("choices"):
                stripped["choices"][0].pop("logprobs", None)
                if isinstance(stripped["choices"][0].get("message"), dict):
                    stripped["choices"][0]["message"].pop("reasoning_content", None)
            return status, stripped


class _FallbackProxyHandler(http.server.BaseHTTPRequestHandler):
    state: _FallbackProxyState

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        status, reply = self.state.handle_completion(self.path, body)
        raw = json.dumps(reply).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _FallbackLoggingProxyServer:
    def __init__(self, state: _FallbackProxyState, host: str = "127.0.0.1", port: int = 0) -> None:
        self.state = state
        handler_cls = type("BoundFallbackProxyHandler", (_FallbackProxyHandler,), {"state": state})
        self.httpd = http.server.ThreadingHTTPServer((host, port), handler_cls)
        self.host, self.port = self.httpd.server_address[0], self.httpd.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> _FallbackLoggingProxyServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


def _import_proxy():
    """Import LoggingProxyServer and ProxyState from mcptoolshop_rnd when available (Colab),
    or fall back to the stdlib implementation in unit tests."""
    try:
        from mcptoolshop_rnd.localbuilders.proxy import LoggingProxyServer, ProxyState  # noqa: PLC0415

        return LoggingProxyServer, ProxyState
    except ModuleNotFoundError:
        return _FallbackLoggingProxyServer, _FallbackProxyState


class _ArrivalGate:
    """Serializes proxy arrival order so request_index 0..59 gets call_index == request_index + 1,
    releasing request_index + 1 the instant request_index increments ProxyState.call_count (before urlopen)."""

    def __init__(self, state):
        self.state = state
        self._inner = state._lock
        self._cond = threading.Condition()

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        res = self._inner.__exit__(*exc)
        with self._cond:
            self._cond.notify_all()
        return res

    def wait_turn(self, request_index: int, stop_event: threading.Event | None = None, timeout: float = 120.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.state.call_count < request_index:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError(f"aborted waiting for proxy turn at request_index={request_index}")
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise RuntimeError(
                        f"timed out waiting for proxy turn at request_index={request_index} (call_count={self.state.call_count})"
                    )
                self._cond.wait(timeout=min(0.05, rem))


def join_calls_and_generations(calls_rows: list[dict], gen_rows: list[dict]) -> list[tuple[dict, dict]]:
    """Join calls-cN.jsonl and generations-cN.jsonl by `call_index == request_index + 1` (DESIGN-v1.md §4).
    Never joins by line position or completion order."""

    refused = [g for g in gen_rows if g.get("kind") != "result"]
    if refused:
        raise RuntimeError(f"REFUSED: generations log contains {len(refused)} non-result row(s): {refused[0]}")

    by_call_index: dict[int, dict] = {}
    for g in gen_rows:
        cidx = g.get("call_index")
        if not isinstance(cidx, int) or isinstance(cidx, bool) or cidx < 1:
            raise RuntimeError(f"REFUSED: generation row has invalid call_index {cidx!r}")
        if cidx in by_call_index:
            raise RuntimeError(f"REFUSED: duplicate call_index {cidx} in generations log")
        by_call_index[cidx] = g

    by_req_index: dict[int, dict] = {}
    for c in calls_rows:
        ridx = c.get("request_index")
        if not isinstance(ridx, int) or isinstance(ridx, bool) or ridx < 0:
            raise RuntimeError(f"REFUSED: call row has invalid request_index {ridx!r}")
        if ridx in by_req_index:
            raise RuntimeError(f"REFUSED: duplicate request_index {ridx} in calls journal")
        by_req_index[ridx] = c

    if len(by_req_index) != len(by_call_index):
        raise RuntimeError(
            f"REFUSED: count mismatch between calls ({len(by_req_index)}) and generations ({len(by_call_index)})"
        )

    joined: list[tuple[dict, dict]] = []
    for ridx in sorted(by_req_index):
        c_row = by_req_index[ridx]
        want_call_index = ridx + 1
        if want_call_index not in by_call_index:
            raise RuntimeError(
                f"REFUSED: request_index {ridx} has no matching call_index {want_call_index} in generations log"
            )
        g_row = by_call_index[want_call_index]
        c_text = (c_row.get("completion") or {}).get("text") if isinstance(c_row.get("completion"), dict) else c_row.get("completion")
        g_text = g_row.get("response")
        if (c_text or "") != (g_text or ""):
            raise RuntimeError(
                f"REFUSED: joined text mismatch at request_index={ridx} (call_index={want_call_index}): "
                f"calls has {str(c_text)[:60]!r} vs generations {str(g_text)[:60]!r}"
            )
        joined.append((c_row, g_row))
    return joined


def _cached_tokens_in_generation(gen_row: dict) -> int:
    timings = gen_row.get("timings") if isinstance(gen_row.get("timings"), dict) else {}
    usage = gen_row.get("usage") if isinstance(gen_row.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return max(
        int(timings.get("cache_n") or 0),
        int(details.get("cached_tokens") or 0),
        int(usage.get("cached_tokens") or 0),
    )


def _align_stub_requests(received_bodies: list[dict], requests: list[dict]) -> list[dict]:
    """Align concurrent StubLlama received POST bodies into `request_index` 0..N-1 order by `messages_sha256`."""

    if len(received_bodies) != len(requests):
        return list(received_bodies)
    buckets: dict[str, list[dict]] = {}
    for b in received_bodies:
        m_sha = callcache.messages_sha256(b.get("messages") or [])
        buckets.setdefault(m_sha, []).append(b)
    aligned: list[dict | None] = [None] * len(requests)
    for idx, req in enumerate(requests):
        q = buckets.get(req["messages_sha256"])
        if q:
            aligned[idx] = q.pop(0)
    leftover = [b for q in buckets.values() for b in q]
    for idx in range(len(aligned)):
        if aligned[idx] is None and leftover:
            aligned[idx] = leftover.pop(0)
    return [b if b is not None else received_bodies[idx] for idx, b in enumerate(aligned)]


def verify_stub_received_bodies(
    received_bodies: list[dict],
    requests: list[dict],
    *,
    expected_sampling: dict = EXPECTED_CARD_SAMPLING,
    expected_kwargs: dict = EXPECTED_CARD_KWARGS,
) -> dict:
    """Verify that the card's sampling parameters and each row's explicit integer seed reached StubLlama (check 4)."""

    if len(received_bodies) != len(requests):
        raise RuntimeError(
            f"REFUSED: StubLlama received {len(received_bodies)} POST bodies, expected {len(requests)}"
        )
    ordered = _align_stub_requests(received_bodies, requests)
    dropped_seed_rows: list[int] = []
    sampling_mismatches: list[dict] = []
    for idx, (body, req) in enumerate(zip(ordered, requests)):
        if "seed" not in body or body["seed"] is None or isinstance(body["seed"], bool) or not isinstance(body["seed"], int):
            dropped_seed_rows.append(idx)
            continue
        if body["seed"] != req["seed"]:
            dropped_seed_rows.append(idx)
            continue
        got_msg_sha = callcache.messages_sha256(body.get("messages") or [])
        if got_msg_sha != req["messages_sha256"]:
            raise RuntimeError(
                f"REFUSED: StubLlama received mutated messages at request_index={idx}: {got_msg_sha} != {req['messages_sha256']}"
            )
        for k, want_v in expected_sampling.items():
            if body.get(k) != want_v:
                sampling_mismatches.append({"request_index": idx, "field": k, "got": body.get(k), "want": want_v})
        if (body.get("chat_template_kwargs") or {}) != expected_kwargs:
            sampling_mismatches.append({
                "request_index": idx,
                "field": "chat_template_kwargs",
                "got": body.get("chat_template_kwargs"),
                "want": expected_kwargs,
            })
    if dropped_seed_rows:
        raise RuntimeError(
            f"REFUSED: seed missing or mismatched in server received POST body at request_index: {dropped_seed_rows}"
        )
    if sampling_mismatches:
        raise RuntimeError(
            f"REFUSED: card sampling mismatch in server received POST body: {sampling_mismatches[0]}"
        )
    return {
        "checked": len(received_bodies),
        "dropped_seed_rows": dropped_seed_rows,
        "sampling_mismatches": sampling_mismatches,
        "ok": True,
    }


def _poll_slots_busy(port: int) -> int:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=5) as r:  # noqa: S310 - loopback
            data = json.loads(r.read())
        if isinstance(data, list):
            return sum(1 for s in data if isinstance(s, dict) and (s.get("is_processing") or s.get("state") == 1))
    except Exception:  # noqa: BLE001
        pass
    return 0


def _kv_buffer_lines(server_log: Path | None) -> list[str]:
    if server_log is None or not server_log.is_file():
        return []
    text = server_log.read_text(encoding="utf-8", errors="replace")
    return [ln.strip()[:240] for ln in text.splitlines() if "KV buffer" in ln or "kv_cache" in ln.lower()][:20]


def replay_arm(
    *,
    arm_name: str,
    parallel: int,
    requests: list[dict],
    requests_sha256: str,
    built: dict,
    model_info: dict,
    proxy_cfg: dict,
    out_dir: Path,
    pw_mod,
    port: int | None = None,
    out_of_order_stub: bool = False,
    stub_call_delay_s: float | None = None,
) -> dict:
    """Run one batching-drift arm (fresh server start, /props check, N-way dispatch through LoggingProxyServer)."""

    require_no_resume_env()
    pw_mod.check_deadline(f"batching-drift arm {arm_name} (parallel={parallel})")
    eff_port = int(port if port is not None else pw_mod.PILOT_PORT)
    ctx_total = PER_SLOT_CTX * parallel
    extra_args = list(BATCHDRIFT_EXTRA_ARGS)

    out_dir.mkdir(parents=True, exist_ok=True)
    critic_name = model_info["served_as"]
    calls_path = out_dir / f"calls-{arm_name}-{critic_name}.jsonl"
    gen_path = out_dir / f"generations-{arm_name}-{critic_name}.jsonl"
    smi_csv = out_dir / f"smi-{arm_name}.csv"
    calls_path.unlink(missing_ok=True)
    gen_path.unlink(missing_ok=True)

    srv = pw_mod.Served(
        built,
        model_info,
        model_info["gguf_sha256"],
        parallel=parallel,
        ctx=ctx_total,
        extra_args=extra_args,
    )

    LoggingProxyServer, ProxyState = _import_proxy()

    with srv:
        if srv.stub is not None:
            ans = build_stub_answers(requests)
            inner_critic = MockCritic(answers=ans)
            delay_s = (
                float(stub_call_delay_s)
                if stub_call_delay_s is not None
                else float(os.environ.get("PIN_STUB_CALL_S") or 0.0)
            )
            srv.stub.critic = pw_mod._SlowStub(inner_critic, delay_s) if delay_s > 0 else inner_critic
            if out_of_order_stub or os.environ.get("PIN_STUB_OUT_OF_ORDER") == "1":
                srv.stub.out_of_order = True

        # Explicit /props assertion before the arm starts (DESIGN-v1.md §3, §5 check 2).
        props_slots = pw_mod.slots(eff_port, PER_SLOT_CTX, want_slots=parallel)
        if not props_slots["ok"] or props_slots["total_slots"] != parallel or props_slots["n_ctx_per_slot"] != PER_SLOT_CTX:
            raise RuntimeError(
                f"REFUSED: arm {arm_name} /props check failed: expected total_slots={parallel}, "
                f"n_ctx_per_slot={PER_SLOT_CTX}, got {props_slots}"
            )
        start_rec = srv.starts[-1] if srv.starts else {}
        if list(start_rec.get("extra_args") or []) != extra_args:
            raise RuntimeError(
                f"REFUSED: arm {arm_name} launch extra_args {start_rec.get('extra_args')} != {extra_args}"
            )

        state = ProxyState(
            upstream_url=f"http://127.0.0.1:{eff_port}",
            config=proxy_cfg["config"],
            candidate=proxy_cfg["candidate"],
            generations_path=gen_path,
            max_calls=10**6,
            wall_limit_s=86400,
        )
        arrival_gate = _ArrivalGate(state)
        state._lock = arrival_gate
        px = LoggingProxyServer(state).start()
        endpoint = px.openai_base_url

        stop_tail = threading.Event()
        tail_thread = threading.Thread(
            target=pw_mod.stream_calls,
            args=(calls_path, arm_name, f"batchdrift-{arm_name}", stop_tail),
            daemon=True,
        )
        tail_thread.start()

        stop_smi = threading.Event()
        smi_stats = {"samples": 0, "peak_vram_mib": 0}
        smi_thread = None
        if hasattr(pw_mod, "sample_smi_every_2s"):
            smi_thread = threading.Thread(
                target=pw_mod.sample_smi_every_2s,
                args=(smi_csv, stop_smi, smi_stats),
                daemon=True,
            )
            smi_thread.start()

        write_lock = threading.Lock()
        in_flight_lock = threading.Lock()
        client_in_flight = 0
        client_max_in_flight = 0
        slots_busy_peak = 0
        completed_count = 0
        cut_triggered = threading.Event()
        t0 = time.time()

        def _watch_deadline() -> None:
            while not stop_tail.is_set():
                left = pw_mod.deadline_left()
                if left is not None and left <= 0:
                    cut_triggered.set()
                    try:
                        px.close()
                    except Exception:  # noqa: BLE001
                        pass
                    if srv.stub is not None:
                        try:
                            srv.stub.close()
                        except Exception:  # noqa: BLE001
                            pass
                    return
                stop_tail.wait(0.05)

        watcher = threading.Thread(target=_watch_deadline, daemon=True)
        watcher.start()

        def _run_one(row: dict) -> dict:
            nonlocal client_in_flight, client_max_in_flight, slots_busy_peak, completed_count
            idx = row["request_index"]
            validate_request_row(row, idx)
            if cut_triggered.is_set():
                raise pw_mod.DeadlineReached(f"batching-drift arm {arm_name} cut at deadline")
            pw_mod.check_deadline(f"batching-drift {arm_name} request {idx}")

            arrival_gate.wait_turn(idx, stop_event=cut_triggered)
            with in_flight_lock:
                client_in_flight += 1
                if client_in_flight > client_max_in_flight:
                    client_max_in_flight = client_in_flight
                if client_in_flight == parallel:
                    polled = _poll_slots_busy(eff_port)
                    if polled > slots_busy_peak:
                        slots_busy_peak = polled
            try:
                comp = transport.chat(
                    endpoint,
                    critic_name,
                    row["messages"],
                    seed=row["seed"],
                    max_tokens=CRITIC_MAX_TOKENS,
                )
            finally:
                with in_flight_lock:
                    polled = _poll_slots_busy(eff_port)
                    if polled > slots_busy_peak:
                        slots_busy_peak = polled
                    client_in_flight -= 1

            if cut_triggered.is_set():
                raise pw_mod.DeadlineReached(f"batching-drift arm {arm_name} cut at deadline")
            left = pw_mod.deadline_left()
            if left is not None and left <= 0:
                cut_triggered.set()
                raise pw_mod.DeadlineReached(f"batching-drift arm {arm_name} cut at deadline")

            payload_sha = canonical_payload_sha256(row["messages"], row["seed"])
            if payload_sha != row["payload_sha256"]:
                raise RuntimeError(
                    f"REFUSED: request_index {idx} payload_sha256 {payload_sha} != {row['payload_sha256']}"
                )
            if comp.messages_sha256 != row["messages_sha256"]:
                raise RuntimeError(
                    f"REFUSED: request_index {idx} messages_sha256 {comp.messages_sha256} != {row['messages_sha256']}"
                )

            call_row = {
                "role": "critic",
                "arm": arm_name,
                "parallel": parallel,
                "key": row["key"],
                "request_index": idx,
                "slot": row["slot"],
                "check": row["check"],
                "call_occurrence": row.get("call_occurrence", 0),
                "attempt": row["attempt"],
                "seed": row["seed"],
                "messages_sha256": row["messages_sha256"],
                "payload_sha256": payload_sha,
                "reused": False,
                "completion": dataclasses.asdict(comp),
            }
            with write_lock:
                if cut_triggered.is_set():
                    raise pw_mod.DeadlineReached(f"batching-drift arm {arm_name} cut at deadline")
                with calls_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(call_row, ensure_ascii=False) + "\n")
                    fh.flush()
                completed_count += 1
            return call_row

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(_run_one, r) for r in requests]
                for fut in futures:
                    try:
                        fut.result()
                    except BaseException as exc:  # noqa: BLE001
                        if cut_triggered.is_set() or isinstance(exc, pw_mod.DeadlineReached):
                            for pending_fut in futures:
                                pending_fut.cancel()
                            pw_mod.CUT_AT = {
                                "wave": "batching-drift",
                                "arm": arm_name,
                                "parallel": parallel,
                                "out": str(out_dir),
                                "journal": str(calls_path),
                                "completed_calls": completed_count,
                                "seconds": round(time.time() - t0, 2),
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                            pw_mod.say(
                                f"CUT AT THE DEADLINE batching-drift {arm_name} after {pw_mod.CUT_AT['seconds']} s; "
                                f"{completed_count} finished calls are in {calls_path.name}"
                            )
                            raise pw_mod.DeadlineReached(
                                f"batching-drift arm {arm_name} was cut at the window's deadline"
                            ) from None
                        raise
        finally:
            stop_tail.set()
            tail_thread.join(timeout=30)
            stop_smi.set()
            if smi_thread is not None:
                smi_thread.join(timeout=15)
            try:
                px.close()
            except Exception:  # noqa: BLE001
                pass

        wall_s = time.time() - t0
        calls_rows = [json.loads(ln) for ln in calls_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        gen_rows = [json.loads(ln) for ln in gen_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(calls_rows) != len(requests):
            raise RuntimeError(
                f"REFUSED: arm {arm_name} wrote {len(calls_rows)} call rows, expected {len(requests)}"
            )
        joined = join_calls_and_generations(calls_rows, gen_rows)

        # Verify 0 cached prompt tokens and 0 reused rows (DESIGN-v1.md §4, §5 check 3).
        cached_total = sum(_cached_tokens_in_generation(g) for _, g in joined)
        if cached_total != 0:
            raise RuntimeError(
                f"REFUSED: arm {arm_name} had {cached_total} cached prompt tokens (must be 0)"
            )
        if any(c.get("reused") for c, _ in joined):
            raise RuntimeError(f"REFUSED: arm {arm_name} had reused call journal rows")

        out_of_order_count = sum(
            1 for line_idx, c in enumerate(calls_rows) if c.get("request_index") != line_idx
        )
        stub_meta = None
        if srv.stub is not None:
            if srv.stub.max_in_flight > slots_busy_peak:
                slots_busy_peak = srv.stub.max_in_flight
            if srv.stub.slots_polls and max(srv.stub.slots_polls) > slots_busy_peak:
                slots_busy_peak = max(srv.stub.slots_polls)
            aligned_stub_reqs = _align_stub_requests(srv.stub.requests, requests)
            stub_req_path = out_dir / f"stub-requests-{arm_name}-{critic_name}.jsonl"
            stub_req_path.write_text(
                "".join(json.dumps(b, ensure_ascii=False) + "\n" for b in aligned_stub_reqs),
                encoding="utf-8",
                newline="\n",
            )
            stub_check = verify_stub_received_bodies(aligned_stub_reqs, requests)
            if len(requests) >= parallel and srv.stub.max_in_flight != parallel:
                raise RuntimeError(
                    f"REFUSED: stub arm {arm_name} max_in_flight={srv.stub.max_in_flight}, expected {parallel}"
                )
            stub_meta = {
                "max_in_flight": srv.stub.max_in_flight,
                "slots_busy_peak": slots_busy_peak,
                "slot_log_count": len(srv.stub.slot_log),
                "slot_log_sample": srv.stub.slot_log[:8],
                "received_check": stub_check,
                "stub_requests_path": str(stub_req_path),
            }
        server_log_path = getattr(srv, "log", None)
        if server_log_path and Path(server_log_path).is_file():
            shutil.copy(server_log_path, out_dir / f"server-{arm_name}.log")

        return {
            "arm": arm_name,
            "parallel": parallel,
            "ctx": ctx_total,
            "n_ctx_per_slot": PER_SLOT_CTX,
            "extra_args": extra_args,
            "props_slots": props_slots,
            "requests": len(requests),
            "requests_sha256": requests_sha256,
            "calls_completed": len(calls_rows),
            "calls_path": str(calls_path),
            "calls_sha256": file_sha256(calls_path),
            "generations_path": str(gen_path),
            "generations": pw_mod.generations_summary(gen_path),
            "cached_prompt_tokens": cached_total,
            "reused_calls": 0,
            "out_of_order_completions": out_of_order_count,
            "client_max_in_flight": client_max_in_flight,
            "slots_busy_peak": max(slots_busy_peak, client_max_in_flight),
            "seconds": round(wall_s, 3),
            "items_per_hour": round(len(requests) * 3600.0 / max(wall_s, 0.001), 2),
            "smi": smi_stats,
            "kv_buffer_lines": _kv_buffer_lines(Path(server_log_path) if server_log_path else None),
            "exit": 0,
            **({"stub": stub_meta} if stub_meta is not None else {}),
        }


def run_all_arms(
    *,
    built: dict,
    model_info: dict,
    proxy_cfg: dict,
    requests_path: Path,
    out_dir: Path,
    pw_mod,
    arms: tuple[tuple[str, int], ...] = REQUIRED_ARMS,
    expected_requests_sha256: str | None = REQUESTS60_SHA256,
    expected_requests_count: int | None = 60,
    port: int | None = None,
    out_of_order_stub: bool = False,
    stub_call_delay_s: float | None = None,
) -> dict:
    """Run all six batching-drift arms in order (A1, A2, B2, B4, B8, B4r) and verify every arm completed."""

    require_no_resume_env()
    requests, req_sha = load_and_verify_requests60(
        requests_path,
        expected_sha256=expected_requests_sha256,
        expected_count=expected_requests_count,
    )

    arm_results: dict[str, dict] = {}
    for arm_name, parallel in arms:
        res = replay_arm(
            arm_name=arm_name,
            parallel=parallel,
            requests=requests,
            requests_sha256=req_sha,
            built=built,
            model_info=model_info,
            proxy_cfg=proxy_cfg,
            out_dir=out_dir,
            pw_mod=pw_mod,
            port=port,
            out_of_order_stub=out_of_order_stub,
            stub_call_delay_s=stub_call_delay_s,
        )
        arm_results[arm_name] = res

    required_names = [a for a, _ in REQUIRED_ARMS]
    missing_arms = [
        a
        for a in required_names
        if a not in arm_results
        or arm_results[a].get("exit") != 0
        or arm_results[a].get("calls_completed") != len(requests)
    ]
    if missing_arms:
        raise RuntimeError(
            f"REFUSED: batching-drift did not complete all six required arms; missing or incomplete: {', '.join(missing_arms)}"
        )

    import compare_batch  # noqa: PLC0415

    comparison = compare_batch.compare_study_dir(
        out_dir,
        requests_path=requests_path,
        critic_model=model_info["served_as"],
        expected_requests_sha256=expected_requests_sha256,
        expected_requests_count=expected_requests_count,
    )
    comp_path = out_dir / "comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8", newline="\n")

    return {
        "requests_path": str(requests_path),
        "requests_sha256": req_sha,
        "requests_count": len(requests),
        "arms": arm_results,
        "comparison_path": str(comp_path),
        "comparison": comparison,
        "all_six_arms_completed": True,
    }


def batching_drift_window(built: dict, result: dict, pw_mod=None) -> None:
    """Entry point for `PIN_ROLE=batching-drift` inside `pin_window.py` (or standalone)."""

    if pw_mod is None:
        import pin_window as pw_mod  # noqa: PLC0415

    require_no_resume_env()
    held_token = pw_mod.take_token()
    try:
        pw_mod.check_token_clean(result)
        if not os.environ.get("PIN_GGUF_REPO"):
            held_token = None
        rs = result["role_set"] = pw_mod.role_set_env(os.environ["PIN_ROLESET_COMMIT"])
        result["rnd"] = pw_mod.install_rnd(os.environ["PIN_RND_COMMIT"])
        # Preflight: the comparison at the end needs the gates (aspire_si_gates); fail here, before any paid arm, not
        # after six of them (review ko, 2026-10-28).
        from role_set import gates as _gates  # noqa: PLC0415

        _gates.load_gates()
        result["gates_preflight"] = "ok"
        critic_name = (
            os.environ.get("PIN_BATCHDRIFT_CRITIC")
            or os.environ.get("PIN_CRITIC_MODEL")
            or DEFAULT_CRITIC
        ).strip()
        by_name = {m["served_as"]: m for m in pw_mod.MODELS}
        if critic_name not in by_name:
            raise RuntimeError(f"REFUSED: {critic_name!r} is not in pin_window.MODELS")

        raw_req_path = (os.environ.get("PIN_BATCHDRIFT_REQUESTS") or "").strip()
        if raw_req_path:
            requests_path = Path(raw_req_path)
        elif (pw_mod.CONTENT / "requests-60.jsonl").is_file():
            requests_path = pw_mod.CONTENT / "requests-60.jsonl"
        else:
            requests_path = Path(rs["src"]) / "scripts" / "batchdrift" / "requests-60.jsonl"
        expect_sha = (os.environ.get("PIN_REQUESTS60_SHA256") or REQUESTS60_SHA256).strip()

        cards_file = pw_mod.CONTENT / "card-values.json"
        if os.environ.get("PIN_GGUF_REPO"):
            pw_mod._TOKEN, held_token = held_token, None
        pw_mod.prefetch_hosted([critic_name], by_name, result)
        pw_mod.check_token_clean(result)
    finally:
        held_token = None
        pw_mod._TOKEN = None

    out_dir = pw_mod.WORK / "batchdrift"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        pool = pw_mod.Pool(built, result, {}, cards_file, pw_mod.WORK / "cards-j1")
        m = pool.get(critic_name)
        result["critic"] = {"served_as": m["served_as"], "gguf_sha256": m["gguf_sha256"]}
        proxy_cfg = pool.proxy(critic_name)
        if proxy_cfg is None:
            raise RuntimeError(f"REFUSED: no proxy card configuration for {critic_name}")

        raw_arms = (os.environ.get("PIN_BATCHDRIFT_ARMS") or "").strip()
        if raw_arms:
            arm_map = dict(REQUIRED_ARMS)
            arms = tuple((a.strip(), arm_map[a.strip()]) for a in raw_arms.split(",") if a.strip())
        else:
            arms = REQUIRED_ARMS

        summary = run_all_arms(
            built=built,
            model_info=m,
            proxy_cfg=proxy_cfg,
            requests_path=requests_path,
            out_dir=out_dir,
            pw_mod=pw_mod,
            arms=arms,
            expected_requests_sha256=expect_sha,
        )
        result["batching_drift"] = summary
        result["model_errors"] = 0
    finally:
        if (out_dir.exists()):
            proc = subprocess.run(
                ["tar", "-C", str(pw_mod.WORK), "-cf", str(pw_mod.WORK / "batchdrift-out.tar"), "batchdrift"],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                result["tar_error"] = (proc.stderr or proc.stdout)[:300]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the batching-drift replay client.")
    ap.add_argument("--requests", type=Path, default=None, help="Path to requests-60.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="Output directory")
    args = ap.parse_args(argv)

    if args.requests is not None:
        os.environ["PIN_BATCHDRIFT_REQUESTS"] = str(args.requests)
    if args.out is not None:
        os.environ["PIN_WORK"] = str(args.out)
    os.environ.setdefault("PIN_ROLE", "batching-drift")
    import pin_window as pw  # noqa: PLC0415

    return pw.main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
