#!/usr/bin/env python3
"""Tests for the MCP stdio server (``pdf_ocr_backend.py --mcp``).

The server calls the library in-process, so these tests drive the registered
tool through ``MCPServer.call_tool``: no subprocess, and the typed errors are
what the tests assert on. The stdio handshake itself is covered by
test_integration.py.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "tool"))

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

import pdf_ocr_backend  # noqa: E402
from tests.support import TempDirTestCase, make_pdf, write_route  # noqa: E402


def call_tool(server, arguments):
    """Call the pdf_ocr tool on ``server`` from a synchronous test."""
    return asyncio.run(server.call_tool("pdf_ocr", arguments))


class TestMCPServer(TempDirTestCase):
    """The pdf_ocr tool, driven in-process through the server object."""

    def setUp(self):
        super().setUp()
        self.pdf = make_pdf(self.temp_dir, pages=2)
        self.route_file = write_route(
            self.temp_dir, {"model": "stub-model", "base_url": "http://stub:8080"}
        )
        self.server = pdf_ocr_backend.build_mcp_server(str(self.route_file))

    def test_exposes_one_tool(self):
        """The server exposes exactly the pdf_ocr tool."""
        tools = asyncio.run(self.server.list_tools())
        self.assertEqual([tool.name for tool in tools], ["pdf_ocr"])

    @patch("pdf_ocr_backend.process_with_model", return_value="--- Page 1 ---\nok")
    def test_tool_returns_the_ocr_text(self, mock_process):
        """A successful call returns the OCR text, routed by the given file."""
        result = call_tool(self.server, {"pdf_path": self.pdf, "page": "1"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.content[0].text, "--- Page 1 ---\nok")
        self.assertEqual(mock_process.call_args.args[1], "stub-model")
        self.assertEqual(mock_process.call_args.args[2], "http://stub:8080")

    @patch("pdf_ocr_backend.process_with_model", return_value="ok")
    def test_env_var_selects_the_route_file(self, mock_process):
        """With no explicit path, the tool reads PDF_OCR_ROUTING_CONFIG."""
        with patch.dict(os.environ, {"PDF_OCR_ROUTING_CONFIG": self.route_file}):
            server = pdf_ocr_backend.build_mcp_server()
            call_tool(server, {"pdf_path": self.pdf})

        self.assertEqual(mock_process.call_args.args[1], "stub-model")

    @patch("pdf_ocr_backend.cleanup_temp_file")
    @patch("pdf_ocr_backend.download_pdf", return_value="/tmp/downloaded.pdf")
    @patch("pdf_ocr_backend.route_ocr_request", return_value="text")
    def test_url_is_downloaded_and_cleaned_up(
        self, mock_route, mock_download, mock_cleanup
    ):
        """A URL argument is downloaded first, then removed after the call."""
        result = call_tool(self.server, {"pdf_path": "https://example.com/a.pdf"})

        self.assertEqual(result.content[0].text, "text")
        mock_download.assert_called_once_with("https://example.com/a.pdf")
        mock_route.assert_called_once_with(
            "/tmp/downloaded.pdf", None, routing_config_path=str(self.route_file)
        )
        mock_cleanup.assert_called_once_with("/tmp/downloaded.pdf")

    def test_missing_file_fails_the_call_with_a_reason(self):
        """A path that does not exist fails the call with the CLI's message."""
        missing = os.path.join(self.temp_dir, "nope.pdf")

        with self.assertRaises(ToolError) as cm:
            call_tool(self.server, {"pdf_path": missing})

        self.assertIn(f"PDF file not found: {missing}", str(cm.exception))

    @patch("pdf_ocr_backend._ocr_page")
    def test_endpoint_rejection_becomes_a_tool_error(self, mock_page):
        """A route-level rejection fails the call with the backend's message."""
        mock_page.side_effect = pdf_ocr_backend.NoOcrSupportError(
            "NO_OCR_SUPPORT: the endpoint rejected model 'x' for OCR."
        )

        with self.assertRaises(ToolError) as cm:
            call_tool(self.server, {"pdf_path": self.pdf, "page": "1"})

        self.assertIn("NO_OCR_SUPPORT", str(cm.exception))

    @patch("pdf_ocr_backend._ocr_page")
    def test_partial_pages_still_return_what_worked(self, mock_page):
        """A page failure fails the call but keeps the pages that succeeded."""
        mock_page.side_effect = [pdf_ocr_backend.OcrRequestError("boom"), "page two"]

        with self.assertRaises(ToolError) as cm:
            call_tool(self.server, {"pdf_path": self.pdf})

        self.assertIn("page two", str(cm.exception))
        self.assertIn("[OCR failed: boom]", str(cm.exception))

    @patch(
        "pdf_ocr_backend.route_ocr_request",
        side_effect=pdf_ocr_backend.OcrRequestError("transport died"),
    )
    def test_other_backend_errors_become_a_tool_error(self, mock_route):
        """A transport failure fails the call with the backend's message."""
        with self.assertRaises(ToolError) as cm:
            call_tool(self.server, {"pdf_path": self.pdf})

        self.assertIn("transport died", str(cm.exception))


if __name__ == "__main__":
    unittest.main()