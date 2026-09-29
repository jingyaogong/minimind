"""Loopback-only web chat and same-origin proxy for the local vLLM API."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CHAT_PAGE = Path(__file__).with_name("chat.html")


class ChatHandler(BaseHTTPRequestHandler):
    backend = "http://127.0.0.1:8000"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, path: str, body: bytes | None = None) -> None:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.backend}{path}",
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urlopen(request, timeout=600) as response:
                self._respond(
                    response.status,
                    response.read(),
                    response.headers.get("Content-Type", "application/json"),
                )
        except HTTPError as error:
            self._respond(error.code, error.read(), "application/json; charset=utf-8")
        except (TimeoutError, URLError, OSError) as error:
            message = json.dumps(
                {"error": f"无法连接到 vLLM 服务：{error}"}, ensure_ascii=False
            ).encode("utf-8")
            self._respond(502, message, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if self.path == "/":
            try:
                self._respond(200, CHAT_PAGE.read_bytes(), "text/html; charset=utf-8")
            except OSError as error:
                self._respond(500, str(error).encode(), "text/plain; charset=utf-8")
        elif self.path == "/health":
            self._proxy("/health")
        else:
            self._respond(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self._respond(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 1_048_576:
            self._respond(400, b'{"error":"Invalid request size"}', "application/json")
            return
        self._proxy("/v1/chat/completions", self.rfile.read(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    ChatHandler.backend = args.backend.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"MiniMind web chat: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
