#!/usr/bin/env python3
"""End-to-end test: the backend CLI against a local stub endpoint.

The stub serves /v1/chat/completions, so this exercises the real CLI, the real
HTTP client and the real page loop without touching a live OCR endpoint.
"""

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from tests.support import TempDirTestCase, write_route

REPO_ROOT = Path(__file__).parent.parent
BACKEND = REPO_ROOT / "tool" / "pdf_ocr_backend.py"
PDF = Path(__file__).parent / "test_data" / "multipage.pdf"
EXPECTED_PAGES = 10


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")

        if body.get("model") == "reject-me":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "model does not support images"}}')
            return

        payload = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "STUB PAGE TEXT"},
                    }
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestEndToEnd(TempDirTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Stub)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        super().setUp()
        if not PDF.exists():
            self.skipTest(f"fixture PDF missing: {PDF}")
        self.route_file = write_route(
            self.temp_dir,
            {"model": "stub-ocr", "base_url": f"http://127.0.0.1:{self.port}"},
        )

    def _run(self, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(BACKEND),
                str(PDF),
                "--routing-config",
                self.route_file,
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_multipage_pdf_is_ocrd_page_by_page(self):
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--- Page "), EXPECTED_PAGES)
        self.assertIn("--- Page 1 ---", result.stdout)
        self.assertIn(f"--- Page {EXPECTED_PAGES} ---", result.stdout)
        self.assertIn("STUB PAGE TEXT", result.stdout)

    def test_page_selection_limits_the_pages(self):
        result = self._run("--page", "1-2")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("--- Page "), 2)

    def test_endpoint_rejection_exits_3(self):
        result = self._run("--model", "reject-me")

        self.assertEqual(result.returncode, 3)
        self.assertIn("NO_OCR_SUPPORT", result.stderr)

    def _mcp(self, messages):
        """Run one MCP stdio session and return the responses keyed by id."""
        proc = subprocess.Popen(
            [sys.executable, str(BACKEND), "--mcp", "--routing-config", self.route_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            for message in messages:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()

            responses = {}
            expected = {m["id"] for m in messages if "id" in m}
            while expected - responses.keys():
                line = proc.stdout.readline()
                if not line:
                    break
                message = json.loads(line)
                if "id" in message:
                    responses[message["id"]] = message
            return responses
        finally:
            proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=30)

    def test_mcp_mode_lists_and_runs_the_tool(self):
        """--mcp serves one stdio tool, and that tool OCRs through the stub."""
        responses = self._mcp(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"},
                    },
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "pdf_ocr",
                        "arguments": {"pdf_path": str(PDF), "page": "1"},
                    },
                },
            ]
        )

        self.assertEqual(responses[1]["result"]["serverInfo"]["name"], "pdf-ocr")
        self.assertEqual(
            [tool["name"] for tool in responses[2]["result"]["tools"]], ["pdf_ocr"]
        )
        self.assertFalse(responses[3]["result"]["isError"])
        self.assertIn("STUB PAGE TEXT", responses[3]["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()