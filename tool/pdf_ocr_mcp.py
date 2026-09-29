#!/usr/bin/env python3
"""MCP stdio server for PDF OCR.

Exposes one tool, ``pdf_ocr``, which runs ``pdf_ocr_backend.py`` and returns its
markdown output. The backend owns the routing and prompt logic; this module only
handles the MCP transport and maps the backend's exit codes to tool errors.

Environment:
    DEEPSEEK_OCR_BASE_URL   Root URL of the OCR endpoint, without a /v1 suffix.
                            Required.
    PDF_OCR_ROUTING_CONFIG  Path to the routing JSON. Optional; defaults to
                            ``ocr_routing.json`` next to the backend.

Run:
    uv run --frozen python tool/pdf_ocr_mcp.py
"""

import os
import subprocess
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

BACKEND = Path(__file__).with_name("pdf_ocr_backend.py")
BASE_URL_ENV = "DEEPSEEK_OCR_BASE_URL"
ROUTING_CONFIG_ENV = "PDF_OCR_ROUTING_CONFIG"
OCR_TIMEOUT_SECONDS = 3600

mcp = MCPServer("pdf-ocr")


@mcp.tool()
def pdf_ocr(pdf_path: str, page: str | None = None) -> str:
    """Extract text from a PDF using the configured OCR route.

    Args:
        pdf_path: Absolute path to a local PDF, or an http(s) URL to download.
        page: Optional page selection: '5', '1-5', '1,3,5', or '1-3,5,7-9'.
            Omit to process every page.
    """
    base_url = os.environ.get(BASE_URL_ENV)
    if not base_url:
        raise ToolError(f"{BASE_URL_ENV} is not set for the pdf-ocr MCP server")

    cmd = [sys.executable, str(BACKEND), pdf_path, "markdown", "--base-url", base_url]
    routing_config = os.environ.get(ROUTING_CONFIG_ENV)
    if routing_config:
        cmd += ["--routing-config", routing_config]
    if page:
        cmd += ["--page", page]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"OCR timed out after {OCR_TIMEOUT_SECONDS}s: {pdf_path}")

    if proc.returncode == 3:
        raise ToolError(
            "NO_OCR_SUPPORT: the routed OCR method is current_model but the loaded "
            "model declares no vision support. Add a routing rule for the loaded "
            f"model set.\n{proc.stderr.strip()}"
        )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown error"
        raise ToolError(f"OCR failed (exit {proc.returncode}): {detail}")

    return proc.stdout.strip()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()