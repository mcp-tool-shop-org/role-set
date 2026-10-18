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
    def __init__(self, writer_model: str, critic_model: str, *, critic: MockCritic | None = None,
                 writer: MockWriter | None = None, port: int = 0):
        self.writer_model, self.critic_model = writer_model, critic_model
        self.writer = writer or MockWriter()
        self.critic = critic or MockCritic(answers=self.writer.answers)
        self.requests: list[dict] = []
        self.in_apply = False
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_POST(self):  # noqa: N802 - http.server's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
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
                data = json.dumps(out).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = _ReuseHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
