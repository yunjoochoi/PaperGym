"""Host-side metering proxy: the ONLY LLM path for sandboxed agent code.
Holds the real LLMClient (and provider keys) on the host; the sandbox sees
only an http URL. Records tokens/cost and enforces budget via UsageMeter."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .metering import BudgetExceeded, UsageMeter


def handle_payload(meter: UsageMeter, payload: dict) -> tuple[int, str]:
    """Pure request logic (unit-tested). Returns (status, json_body)."""
    messages = payload.get("messages") or []
    temperature = float(payload.get("temperature", 0.0))
    try:
        content = meter.call(messages, temperature=temperature)
    except BudgetExceeded as exc:
        return 429, json.dumps({"error": str(exc)})
    except Exception as exc:                     # provider/transport error
        return 500, json.dumps({"error": str(exc)})
    return 200, json.dumps({"content": content, "usage": meter.usage()})


def run_proxy(meter: UsageMeter, host: str = "127.0.0.1", port: int = 0):
    """Start the proxy in a background thread. Returns (server, url, thread)."""
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            status, body = handle_payload(meter, payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{server.server_address[1]}"
    return server, url, thread
