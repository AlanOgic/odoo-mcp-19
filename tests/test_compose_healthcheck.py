"""The docker-compose healthcheck probe, run for real against a local HTTP server.

The probe is the Python one-liner in ``docker-compose.yml``. It must pass on any
answer below 500 (the server answers an unauthenticated probe with 401) and fail
on a 5xx or when nothing answers.
"""

import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

COMPOSE = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()


def _probe_source() -> str:
    block = COMPOSE.split("healthcheck:", 1)[1]
    match = re.search(r'^\s*- "(import http\.client.*)"\s*$', block, re.MULTILINE)
    assert match, "healthcheck probe not found in docker-compose.yml"
    return match.group(1)


def _run_probe(port: int) -> int:
    env = {"MCP_PORT": str(port), "PATH": ""}
    return subprocess.run([sys.executable, "-c", _probe_source()], env=env, capture_output=True, timeout=30).returncode


def _serve(status: int) -> HTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/mcp"
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("status,healthy", [(401, True), (200, True), (404, True), (500, False), (503, False)])
def test_probe_judges_the_http_status(status, healthy):
    server = _serve(status)
    try:
        assert (_run_probe(server.server_address[1]) == 0) is healthy
    finally:
        server.shutdown()
        server.server_close()


def test_probe_fails_when_nothing_listens():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert _run_probe(port) != 0


def test_healthcheck_is_not_in_the_dockerfile():
    """The image's default command is STDIO, where an HTTP probe would always fail."""
    dockerfile = (Path(__file__).resolve().parent.parent / "Dockerfile").read_text()
    assert "HEALTHCHECK" not in dockerfile
