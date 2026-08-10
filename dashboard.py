"""Dispute Desk dashboard — trace viewer + approval panel.

    python dashboard.py
    python dashboard.py --provider openai
    python dashboard.py --port 8080

Opens http://localhost:7777. Pick a dispute, watch the agent triage it
in real-time, then approve or deny the proposal.

Architecture (same pattern as Waku):

    browser ──POST /api/triage──▶ Handler.do_POST
                                      │
                                      ▼
                              asyncio.new_event_loop()
                              loop.run_until_complete(
                                  run_agent(..., observer=emit)
                              )
                                      │
                        emit(kind, data) writes SSE frames
                        to self.wfile as the loop runs
                                      │
    browser ◀──data: {"kind":...}──── │

The observer callback is synchronous and called from inside the async
agent loop. It writes to the HTTP response socket directly. This works
because the event loop is dedicated to this one request — no other async
tasks are waiting on it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent.approval import decide, list_pending
from agent.loop_manual import run
from agent.providers import KINDS, get_provider
from server import db

_sessions: dict[str, dict] = {}

PORT = 7777
STATIC = Path(__file__).parent / "static"


class Handler(BaseHTTPRequestHandler):
    provider: str = "local"
    model_override: str | None = None

    def log_message(self, format, *args):
        pass

    def _send_json(self, data: dict | list) -> None:
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # ── GET ────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        if self.path == "/api/disputes":
            rows = db.query(
                "SELECT d.*, c.name, c.account_type, c.status AS account_status, "
                "c.prior_claims_12m "
                "FROM disputes d JOIN customers c ON c.id = d.customer_id "
                "ORDER BY d.id"
            )
            self._send_json(rows)

        elif self.path == "/api/proposals":
            rows = db.query(
                "SELECT p.*, d.category, d.customer_id "
                "FROM proposals p JOIN disputes d ON d.id = p.dispute_id "
                "ORDER BY p.id DESC"
            )
            self._send_json(rows)

        elif self.path == "/api/pending":
            self._send_json(list_pending())

        elif self.path.startswith("/static/"):
            rel = self.path[len("/static/"):]
            file = STATIC / rel
            if file.is_file() and file.resolve().is_relative_to(STATIC.resolve()):
                mime = "text/css" if rel.endswith(".css") else "application/javascript"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.end_headers()
                self.wfile.write(file.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()

        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write((STATIC / "index.html").read_bytes())

        else:
            self.send_response(404)
            self.end_headers()

    # ── POST ───────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        if self.path == "/api/triage":
            self._handle_triage()
        elif self.path == "/api/chat":
            self._handle_chat()
        elif self.path == "/api/approve":
            self._handle_approve()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_triage(self) -> None:
        """Run triage, streaming events as SSE."""
        payload = self._read_body()
        dispute_id = (payload.get("dispute_id") or "").strip()

        if not dispute_id:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"dispute_id required"}')
            return

        # SSE headers — the connection stays open while events stream.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(kind: str, data: dict) -> None:
            """Write one SSE frame. This is the entire real-time channel."""
            try:
                frame = json.dumps({"kind": kind, **data}, default=str)
                self.wfile.write(f"data: {frame}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        # Bridge async → sync: one event loop per request. The observer
        # writes SSE frames synchronously from inside the async loop.
        # This works because the loop is dedicated to this request.
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                run(
                    dispute_id,
                    self.provider,
                    verbose=False,
                    model_override=self.model_override,
                    observer=emit,
                )
            )
            sid = str(uuid.uuid4())
            if result.get("messages"):
                _sessions[sid] = {
                    "messages": result["messages"],
                    "provider_kind": self.provider,
                    "model_override": self.model_override,
                }
            emit("complete", {
                "session_id": sid,
                "steps": result.get("steps", 0),
                "seconds": result.get("seconds", 0),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "trace": result.get("trace", []),
                "repairs": result.get("repairs", []),
                "error": result.get("error"),
            })
        except Exception as exc:
            emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            loop.close()

    def _handle_approve(self) -> None:
        """Approve or deny a pending proposal."""
        payload = self._read_body()
        proposal_id = payload.get("proposal_id")
        decision = payload.get("decision")
        by = payload.get("by", "")

        if not all([proposal_id, decision, by]):
            self._send_json({"ok": False, "error": "proposal_id, decision, and by are required"})
            return

        result = decide(int(proposal_id), decision, by)
        self._send_json(result)

    def _handle_chat(self) -> None:
        """Follow-up chat — LLM only, no tools. Uses stored message history."""
        payload = self._read_body()
        sid = payload.get("session_id", "")
        message = (payload.get("message") or "").strip()

        state = _sessions.get(sid)
        if not state:
            self._send_json({"error": "session expired — run a new triage first"})
            return
        if not message:
            self._send_json({"error": "message required"})
            return

        state["messages"].append({"role": "user", "content": message})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def emit(kind: str, data: dict) -> None:
            try:
                frame = json.dumps({"kind": kind, **data}, default=str)
                self.wfile.write(f"data: {frame}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        # Build a one-off message list with a no-tools instruction appended
        # to the system prompt. This prevents the model from emitting textual
        # tool calls (DeepSeek's <DSML> tags, GPT's ```json blocks, etc.).
        # The stored state is NOT modified — only this LLM call sees it.
        chat_msgs = []
        for m in state["messages"]:
            if m.get("role") == "system":
                chat_msgs.append({
                    "role": "system",
                    "content": m["content"] + (
                        "\n\nYou are now answering follow-up questions about the "
                        "triage above. Answer from the evidence already retrieved "
                        "in this conversation. You cannot call tools or functions "
                        "— do not attempt to. Respond in plain text."
                    ),
                })
            else:
                chat_msgs.append(m)

        loop = asyncio.new_event_loop()
        try:
            provider = get_provider(state["provider_kind"], state["model_override"])
            reply = loop.run_until_complete(
                provider.client.chat.completions.create(
                    model=provider.model,
                    messages=chat_msgs,
                )
            )
            text = (reply.choices[0].message.content or "").strip()
            state["messages"].append({"role": "assistant", "content": text})
            emit("chat_reply", {"text": text})
        except Exception as exc:
            emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            loop.close()

    # ── OPTIONS (CORS preflight) ──────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dispute Desk dashboard.")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", PORT)))
    ap.add_argument("--provider", default="local", choices=KINDS)
    ap.add_argument("--model", default=None, help="override the model name")
    args = ap.parse_args()

    Handler.provider = args.provider
    Handler.model_override = args.model

    for port in range(args.port, args.port + 5):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError:
            print(f"port {port} busy, trying {port + 1}...")
            continue

        print(f"Dispute Desk → http://localhost:{port}  (Ctrl-C to stop)")
        print(f"provider: {args.provider}" + (f", model: {args.model}" if args.model else ""))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")
        finally:
            server.server_close()
        return

    raise SystemExit(f"no free port in {args.port}–{args.port + 4}")


if __name__ == "__main__":
    main()
