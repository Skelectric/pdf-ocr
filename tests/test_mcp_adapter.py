#!/usr/bin/env python3
"""Unit tests for the MCP stdio adapter (tool/pdf_ocr_mcp.py).

The adapter is a thin wrapper around pdf_ocr_backend.py, so these tests only
cover command construction and exit-code translation; the OCR logic itself is
covered by test_backend.py.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "tool"))

from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

import pdf_ocr_mcp  # noqa: E402


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=["pdf_ocr_backend.py"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestMCPAdapter(unittest.TestCase):
    """Command construction and error translation for pdf_ocr()."""

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_base_url_raises(self):
        """Without DEEPSEEK_OCR_BASE_URL the tool fails fast with a clear error."""
        with self.assertRaises(ToolError) as cm:
            pdf_ocr_mcp.pdf_ocr("/tmp/x.pdf")
        self.assertIn("DEEPSEEK_OCR_BASE_URL", str(cm.exception))

    @patch("pdf_ocr_mcp.subprocess.run")
    @patch.dict(
        os.environ,
        {
            "DEEPSEEK_OCR_BASE_URL": "http://ocr-endpoint:8080",
            "PDF_OCR_ROUTING_CONFIG": "/tmp/ocr_routing.json",
        },
        clear=True,
    )
    def test_command_includes_base_url_routing_and_page(self, mock_run):
        """The backend is invoked with base-url, routing-config, and page args."""
        mock_run.return_value = _completed(stdout="--- Page 1 ---\nhello\n")

        result = pdf_ocr_mcp.pdf_ocr("/tmp/doc.pdf", page="1-2")

        self.assertEqual(result, "--- Page 1 ---\nhello")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], sys.executable)
        self.assertTrue(cmd[1].endswith("pdf_ocr_backend.py"))
        self.assertEqual(cmd[2:5], ["/tmp/doc.pdf", "markdown", "--base-url"])
        self.assertEqual(cmd[5], "http://ocr-endpoint:8080")
        self.assertIn("--routing-config", cmd)
        self.assertIn("/tmp/ocr_routing.json", cmd)
        self.assertEqual(cmd[-2:], ["--page", "1-2"])
        self.assertTrue(mock_run.call_args.kwargs["capture_output"])
        self.assertTrue(mock_run.call_args.kwargs["text"])

    @patch("pdf_ocr_mcp.subprocess.run")
    @patch.dict(
        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://host:8080"}, clear=True
    )
    def test_page_and_routing_omitted_when_absent(self, mock_run):
        """With no page and no routing env, only base-url is passed."""
        mock_run.return_value = _completed(stdout="text")

        pdf_ocr_mcp.pdf_ocr("/tmp/doc.pdf")

        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--page", cmd)
        self.assertNotIn("--routing-config", cmd)
        self.assertIn("--base-url", cmd)

    @patch("pdf_ocr_mcp.subprocess.run")
    @patch.dict(
        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://host:8080"}, clear=True
    )
    def test_exit_3_translates_to_no_ocr_support(self, mock_run):
        """Exit 3 becomes a NO_OCR_SUPPORT ToolError."""
        mock_run.return_value = _completed(
            returncode=3, stderr="NO_OCR_SUPPORT: model lacks vision"
        )

        with self.assertRaises(ToolError) as cm:
            pdf_ocr_mcp.pdf_ocr("/tmp/doc.pdf")

        self.assertIn("NO_OCR_SUPPORT", str(cm.exception))
        self.assertIn("model lacks vision", str(cm.exception))

    @patch("pdf_ocr_mcp.subprocess.run")
    @patch.dict(
        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://host:8080"}, clear=True
    )
    def test_generic_failure_surfaces_stderr(self, mock_run):
        """A non-zero exit surfaces the backend's stderr."""
        mock_run.return_value = _completed(returncode=1, stderr="boom")

        with self.assertRaises(ToolError) as cm:
            pdf_ocr_mcp.pdf_ocr("/tmp/doc.pdf")

        self.assertIn("exit 1", str(cm.exception))
        self.assertIn("boom", str(cm.exception))

    @patch("pdf_ocr_mcp.subprocess.run")
    @patch.dict(
        os.environ, {"DEEPSEEK_OCR_BASE_URL": "http://host:8080"}, clear=True
    )
    def test_timeout_raises(self, mock_run):
        """A hung backend is reported as a timeout, not an unhandled error."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pdf_ocr_backend.py", timeout=1)

        with self.assertRaises(ToolError) as cm:
            pdf_ocr_mcp.pdf_ocr("/tmp/doc.pdf")

        self.assertIn("timed out", str(cm.exception))


if __name__ == "__main__":
    unittest.main()