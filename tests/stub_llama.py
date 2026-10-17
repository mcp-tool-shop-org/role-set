"""A stub llama-server for tests that need role-set's real (non-mock) path: an OpenAI-compatible /v1/chat/completions on
127.0.0.1 that answers with the mock writer and the mock critic by the request's model name. Everything above the
transport (build, cards, pins, the waves, the gates) runs as it does on a paid window."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from role_set.mock import MockCritic, MockWriter


class StubLlama:
    def __init__(self, writer_model: str, critic_model: str, *, critic: MockCritic | None = None):
        self.writer_model, self.critic_model = writer_model, critic_model
        self.writer = MockWriter()
        self.critic = critic or MockCritic(answers=self.writer.answers)
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_POST(self):  # noqa: N802 - http.server's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append(body)
                if body["model"] == stub.writer_model:
                    c = stub.writer(body["messages"], seed=body.get("seed") or 0)
                elif body["model"] == stub.critic_model:
                    c = stub.critic(body["messages"], seed=body.get("seed") or 0)
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

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
