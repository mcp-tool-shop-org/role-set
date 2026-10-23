"""A stub llama-server for tests that need role-set's real (non-mock) path: an OpenAI-compatible /v1/chat/completions on
127.0.0.1 that answers with the mock writer and the mock critic by the request's model name. Everything above the
transport (build, cards, pins, the waves, the gates) runs as it does on a paid window."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from role_set.mock import MockCritic, MockWriter


class _ReuseHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class StubLlama:
    def __init__(self, writer_model: str = "stub", critic_model: str = "stub", *, critic: MockCritic | None = None,
                 writer: MockWriter | None = None, port: int = 0, ollama_model: str | None = None,
                 ollama_digest: str | None = None, ollama_version: str = "0.40.4"):
        self.writer_model, self.critic_model = writer_model, critic_model
        self.writer = writer or MockWriter()
        self.critic = critic or MockCritic(answers=getattr(self.writer, "answers", None))
        self.requests: list[dict] = []
        self.ollama_requests: list[tuple[str, dict]] = []
        self.ollama_model = ollama_model or writer_model
        self.ollama_digest = (ollama_digest or ("6" * 64)).removeprefix("sha256:")
        self.ollama_version = ollama_version
        self.ollama_loaded = False
        self.ollama_size = 20281829620
        self.ollama_size_vram = 20281829620
        self.ollama_context_length = 4096
        self.ollama_on_generate = None
        self.ollama_on_chat = None
        self.in_apply = False
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send_json(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802 - http.server's name
                if self.path == "/api/version":
                    self._send_json(200, {"version": stub.ollama_version})
                elif self.path == "/api/tags":
                    self._send_json(
                        200,
                        {
                            "models": [
                                {
                                    "name": stub.ollama_model,
                                    "model": stub.ollama_model,
                                    "size": stub.ollama_size,
                                    "digest": stub.ollama_digest,
                                }
                            ]
                        },
                    )
                elif self.path == "/api/ps":
                    models = (
                        [
                            {
                                "name": stub.ollama_model,
                                "model": stub.ollama_model,
                                "size": stub.ollama_size,
                                "digest": stub.ollama_digest,
                                "size_vram": stub.ollama_size_vram,
                                "context_length": stub.ollama_context_length,
                            }
                        ]
                        if stub.ollama_loaded
                        else []
                    )
                    self._send_json(200, {"models": models})
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):  # noqa: N802 - http.server's name
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                if self.path.startswith("/api/"):
                    stub.ollama_requests.append((self.path, body))
                    if self.path == "/api/show":
                        self._send_json(
                            200,
                            {
                                "capabilities": ["completion", "thinking"],
                                "model_info": {"gemma4.context_length": 262144},
                            },
                        )
                        return
                    if self.path == "/api/generate":
                        opts = body.get("options") or {}
                        if opts.get("num_ctx"):
                            stub.ollama_context_length = int(opts["num_ctx"])
                        keep_alive = body.get("keep_alive")
                        stub.ollama_loaded = keep_alive not in (0, "0", "0s")
                        if stub.ollama_on_generate is not None:
                            stub.ollama_on_generate(body)
                        self._send_json(
                            200,
                            {
                                "model": body.get("model") or stub.ollama_model,
                                "response": "",
                                "done": True,
                                "done_reason": "load" if stub.ollama_loaded else "unload",
                            },
                        )
                        return
                    if self.path == "/api/chat":
                        stub.ollama_loaded = True
                        opts = body.get("options") or {}
                        if opts.get("num_ctx"):
                            stub.ollama_context_length = int(opts["num_ctx"])
                        if stub.ollama_on_chat is not None:
                            stub.ollama_on_chat(body)
                        if opts.get("num_predict") == 1:
                            self._send_json(
                                200,
                                {
                                    "model": body.get("model") or stub.ollama_model,
                                    "message": {"role": "assistant", "content": "ok"},
                                    "done": True,
                                    "done_reason": "length",
                                    "prompt_eval_count": 30,
                                    "eval_count": 1,
                                    "total_duration": 5,
                                },
                            )
                            return
                        stub.writer(body.get("messages") or [], seed=opts.get("seed") or 0)
                        content = json.dumps(
                            {
                                "reasoning": "stub check",
                                "verdict": "unsupported",
                                "evidence_quote": "",
                                "evidence_source": "",
                            }
                        )
                        self._send_json(
                            200,
                            {
                                "model": body.get("model") or stub.ollama_model,
                                "message": {
                                    "role": "assistant",
                                    "content": content,
                                    "thinking": "stub thinking",
                                },
                                "done": True,
                                "done_reason": "stop",
                                "prompt_eval_count": 30,
                                "eval_count": 3,
                                "total_duration": 5,
                            },
                        )
                        return
                stub.requests.append(body)
                msgs = body.get("messages") or []
                system = msgs[0].get("content") or "" if msgs else ""
                user = msgs[-1].get("content") or "" if msgs else ""
                if "## The trait lesson\n" in user:
                    stub.in_apply = "# 4." in user or "## 4." in user or "Lesson T04" in user
                elif "Lesson T04," in user:
                    stub.in_apply = True
                elif "Lesson T02," in user:
                    stub.in_apply = False
                is_writer = ("training items" in system or "training item" in system or "everyday material" in system)
                if body["model"] == stub.writer_model and (body["model"] != stub.critic_model or is_writer):
                    c = stub.writer(msgs, seed=body.get("seed") or 0, slot_id="T04-AP" if stub.in_apply else None)
                elif body["model"] == stub.critic_model:
                    c = stub.critic(msgs, seed=body.get("seed") or 0)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                out = {"model": body["model"], "choices": [{"message": {"content": c.text}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": len(c.text.split())}}
                self._send_json(200, out)

        self.server = _ReuseHTTPServer(("127.0.0.1", port), Handler)
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.url = f"{self.base_url}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


FAKE_OFFRIG_PY = r"""#!/usr/bin/env python3
import json, sys, urllib.request
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["verify", "calibrate"], f"unexpected subcommand: {args}"
gold = Path(args[2])
opts = {}
i = 3
while i < len(args):
    opts[args[i]] = args[i + 1]
    i += 2
url = opts.get("--url", "http://127.0.0.1:11434")
model = opts["--model"]
out_dir = Path(opts["--out"])
proj_dir = Path(opts["--project"])
out_dir.mkdir(parents=True, exist_ok=True)
(proj_dir / ".offrig").mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(f"{url}/api/version", timeout=10) as r:
    ver = json.loads(r.read())["version"]
with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as r:
    tags = json.loads(r.read())["models"]
with urllib.request.urlopen(f"{url}/api/ps", timeout=10) as r:
    ps = json.loads(r.read())["models"]
assert len(ps) == 1, f"expected model already warmed up in /api/ps before offrig starts, got {ps}"
show_req = urllib.request.Request(
    f"{url}/api/show",
    data=json.dumps({"name": model}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(show_req, timeout=10) as r:
    show = json.loads(r.read())

rows = [json.loads(ln) for ln in gold.read_text(encoding="utf-8").splitlines() if ln.strip()]
tune_rows = [row for idx, row in enumerate(rows) if row.get("split", "tune" if idx % 2 == 0 else "heldout") == "tune"] or rows
# Probe call (num_predict=1) then one chat per claim
probe_req = urllib.request.Request(
    f"{url}/api/chat",
    data=json.dumps({"model": model, "messages": [{"role": "user", "content": "probe"}], "options": {"num_predict": 1}}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(probe_req, timeout=10) as r:
    json.loads(r.read())

verdicts = []
for row in tune_rows:
    cid = row.get("id") or row.get("claim_id") or "c1"
    chat_req = urllib.request.Request(
        f"{url}/api/chat",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": row.get("claim", "claim")}],
            "think": opts.get("--think") == "on",
            "options": {"seed": 0, "temperature": 0.0, "num_ctx": 8192, "num_predict": 4096},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(chat_req, timeout=30) as r:
        resp = json.loads(r.read())
    parsed = json.loads(resp["message"]["content"])
    verdicts.append({
        "claim_id": cid,
        "model_verdict": parsed["verdict"],
        "final_verdict": parsed["verdict"],
        "reasoning": parsed["reasoning"],
        "evidence_quote": parsed["evidence_quote"],
        "evidence_source": parsed["evidence_source"],
    })

(out_dir / "verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in verdicts), encoding="utf-8")
digest = tags[0]["digest"]
(out_dir / "manifest.json").write_text(
    json.dumps({
        "model": model,
        "model_digest": digest,
        "ollama_version": ver,
        "think": opts.get("--think"),
        "structured": opts.get("--structured"),
        "split": opts.get("--split"),
        "jobs": int(opts.get("--jobs", "1")),
    }, indent=1) + "\n",
    encoding="utf-8",
)
(out_dir / "metrics.json").write_text(
    json.dumps({
        "model": model,
        "model_digest": digest,
        "claims_selected": len(verdicts),
        "settings": {
            "think": opts.get("--think"),
            "structured": opts.get("--structured"),
            "split": opts.get("--split"),
            "jobs": int(opts.get("--jobs", "1")),
        },
    }, indent=1) + "\n",
    encoding="utf-8",
)
print(f"calibrated {len(verdicts)} claims on {model} ({ver})")
"""
