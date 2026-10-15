#!/usr/bin/env python3
import sys
import os
import json
from typing import Optional
import pymupdf
from openai import (
    BadRequestError,
    NotFoundError,
    OpenAI,
    UnprocessableEntityError,
)
from pathlib import Path
import base64
import argparse
import logging
import requests
import tempfile
import urllib.parse

class PdfOcrError(Exception):
    """Base class for every pdf-ocr failure."""


class ConfigurationError(PdfOcrError):
    """The endpoint or the routing configuration is missing or invalid."""


class DownloadError(PdfOcrError):
    """The PDF could not be downloaded, or it is not a usable PDF."""


class NoOcrSupportError(PdfOcrError):
    """The endpoint rejected the configured model for OCR."""


class OcrRequestError(PdfOcrError):
    """The OCR request to the endpoint failed."""


class PageFailureError(OcrRequestError):
    """Some pages failed OCR. ``results`` carries the pages that succeeded."""

    def __init__(self, message: str, results: list):
        super().__init__(message)
        self.results = results


# Name of the environment variable that points at a non-default routing config.
# An MCP server (or another host adapter) sets this so it can ship a
# deployment-specific ocr_routing.json without copying the whole tool.
ROUTING_CONFIG_ENV = "PDF_OCR_ROUTING_CONFIG"

# Maximum downloadable PDF size (50 MB)
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024

# OCR request settings.
OCR_DPI = 144
OCR_TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 3600

# DeepSeek-OCR grounding. The ngram and window constrain the whitelist to the
# model's special text tokens, so the layout markers survive generation.
DEEPSEEK_OCR_NGRAM_SIZE = 30
DEEPSEEK_OCR_WINDOW_SIZE = 90
DEEPSEEK_OCR_WHITELIST_TOKEN_IDS = [128821, 128822]

logger = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    """Configure root logging. Called by main(), not on import."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_routing_config_path(config_path: Optional[Path] = None) -> Path:
    """Resolve the routing config path: explicit arg > env var > tool dir."""
    if config_path is not None:
        return Path(config_path)
    env_path = os.getenv(ROUTING_CONFIG_ENV)
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "ocr_routing.json"


def is_url(path: str) -> bool:
    """Check if a path is a URL."""
    parsed = urllib.parse.urlparse(path)
    return parsed.scheme in ("http", "https")


def download_pdf(url: str, max_size: int = MAX_DOWNLOAD_SIZE) -> str:
    """
    Download a PDF from a URL to a temporary file.

    Args:
        url: The URL to download from
        max_size: Maximum allowed file size in bytes

    Returns:
        str: Path to the downloaded temporary PDF file

    Raises:
        DownloadError: If the download fails, the size exceeds the limit, or the
            content is not a valid PDF.
    """
    logger.info(f"Downloading PDF from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise DownloadError(f"Failed to download PDF from {url}: {e}")

    # Check Content-Length if available
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            total_size = int(content_length)
        except ValueError:
            total_size = 0  # Ignore invalid Content-Length
        if total_size > max_size:
            raise DownloadError(
                f"PDF file too large ({total_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed size is {max_size / 1024 / 1024:.0f} MB."
            )

    # Stream download with size limit
    downloaded_size = 0
    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                raise DownloadError(
                    f"PDF file exceeds maximum allowed size of {max_size / 1024 / 1024:.0f} MB."
                )
            chunks.append(chunk)

    pdf_bytes = b"".join(chunks)

    if not pdf_bytes:
        raise DownloadError(f"Downloaded PDF from {url} is empty.")

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        temp_file.write(pdf_bytes)
        temp_file.flush()
        temp_file_path = temp_file.name
    finally:
        temp_file.close()

    # Validate that it's a valid PDF
    doc = None
    page_count = 0
    try:
        doc = pymupdf.open(temp_file_path)
        page_count = doc.page_count
        if page_count == 0:
            raise DownloadError("Downloaded file is not a valid PDF (no pages).")
    except DownloadError:
        cleanup_temp_file(temp_file_path)
        raise
    except Exception as e:
        cleanup_temp_file(temp_file_path)
        raise DownloadError(f"Downloaded file is not a valid PDF: {e}")
    finally:
        if doc is not None:
            doc.close()

    logger.info(
        f"PDF downloaded successfully: {temp_file_path} "
        f"({downloaded_size / 1024:.1f} KB, {page_count} pages)"
    )
    return temp_file_path


def cleanup_temp_file(path: str) -> None:
    """Delete a temporary file if it exists."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.debug(f"Cleaned up temporary file: {path}")
    except OSError as e:
        logger.warning(f"Failed to clean up temporary file {path}: {e}")


# Prompt templates, selected by the route's prompt family.
PROMPT_TEMPLATES = {
    "deepseek-ocr": {
        "system": None,
        "user": "Free OCR.",
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": DEEPSEEK_OCR_NGRAM_SIZE,
                "window_size": DEEPSEEK_OCR_WINDOW_SIZE,
                "whitelist_token_ids": DEEPSEEK_OCR_WHITELIST_TOKEN_IDS,
            },
        },
    },
    "default_multimodal": {
        "system": "You are an OCR assistant. Extract all text from the image accurately.",
        "user": "Please perform OCR on this image and extract all visible text. Format the output as markdown.",
        "extra_body": None,
    },
}

# Route prompt families. "generic" is the friendly name for the default template.
PROMPT_FAMILY_ALIASES = {"generic": "default_multimodal"}


def get_prompt_template(model_id: str, prompt_family: Optional[str] = None) -> dict:
    """
    Get the prompt template for a route.

    The route's ``prompt`` family wins when set. Without one the model id decides:
    a deepseek-ocr id gets the special-token template, anything else gets the
    generic one.

    Args:
        model_id: The model identifier string
        prompt_family: Optional family name from the route ("deepseek-ocr" or
            "generic")

    Returns:
        dict: Prompt template with system, user and extra_body keys

    Raises:
        ConfigurationError: If the named family is unknown.
    """
    if prompt_family:
        key = PROMPT_FAMILY_ALIASES.get(prompt_family, prompt_family)
        if key not in PROMPT_TEMPLATES:
            raise ConfigurationError(f"unknown prompt family: {prompt_family}")
        return PROMPT_TEMPLATES[key]

    if "deepseek-ocr" in model_id.lower():
        return PROMPT_TEMPLATES["deepseek-ocr"]
    return PROMPT_TEMPLATES["default_multimodal"]


def _parse_page_number(text: str, part: str) -> int:
    """Parse a positive page number, raising ValueError with the offending part."""
    try:
        number = int(text)
    except ValueError:
        raise ValueError(f"Invalid page number: {part}")
    if number < 1:
        raise ValueError(f"Page numbers must be positive: {part}")
    return number


def parse_page_spec(page_spec: Optional[str], total_pages: int) -> list:
    """
    Parse a page specification string into a list of 0-based page indices.

    Supports formats:
    - Single page: "5"
    - Page range: "1-5"
    - Multiple pages: "1,3,5"
    - Mixed: "1-3,5,7-9"

    Args:
        page_spec: The page specification string
        total_pages: Total number of pages in the document

    Returns:
        list: List of 0-based page indices

    Raises:
        ValueError: If the page specification is invalid
    """
    if not page_spec:
        return list(range(total_pages))

    pages = set()

    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError("Empty page specification part")

        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid range specification: {part}")
            start = _parse_page_number(bounds[0].strip(), part)
            end = _parse_page_number(bounds[1].strip(), part)
            if start > end:
                raise ValueError(f"Invalid range (start > end): {part}")
            numbers = range(start, end + 1)
        else:
            numbers = [_parse_page_number(part, part)]

        for page_num in numbers:
            if page_num > total_pages:
                raise ValueError(
                    f"Page number {page_num} exceeds document length ({total_pages} pages)"
                )
            pages.add(page_num - 1)

    return sorted(list(pages))


def _ocr_page(client: OpenAI, page, model_id: str, prompt_template: dict) -> str:
    """
    OCR one page image and return its text.

    Raises:
        NoOcrSupportError: If the endpoint rejects the model for OCR.
        OcrRequestError: If the endpoint returns no usable text.
    """
    system_prompt = prompt_template.get("system")
    user_prompt = prompt_template["user"]
    extra_body = prompt_template.get("extra_body")

    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csRGB)
    img_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    img_data_url = f"data:image/png;base64,{img_base64}"

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_data_url}},
                {"type": "text", "text": user_prompt},
            ],
        }
    )

    # Build API call parameters
    api_params = {
        "model": model_id,
        "messages": messages,
        "temperature": OCR_TEMPERATURE,
    }

    if extra_body:
        api_params["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**api_params)
    except (BadRequestError, NotFoundError, UnprocessableEntityError) as e:
        raise NoOcrSupportError(
            f"NO_OCR_SUPPORT: the endpoint rejected model '{model_id}' for OCR "
            f"({e}). Point the route at a model that can serve OCR."
        ) from e

    if not response.choices:
        raise OcrRequestError("OCR response contained no choices")

    result_text = response.choices[0].message.content

    if result_text is None:
        raise OcrRequestError("OCR response contained no text")

    # The endpoint returns the byte-fallback encodings of space (U+0120) and
    # newline (U+010A) without decoding them, for every prompt family and with or
    # without skip_special_tokens. Decode them here.
    result_text = result_text.replace("\u0120", " ").replace("\u010a", "\n")

    return result_text


def process_pdf_pages(
    doc: pymupdf.Document,
    model_id: str,
    base_url: str,
    prompt_template: dict,
    page_indices: Optional[list] = None,
) -> list:
    """
    Process pages in a PDF document using the specified model and prompt template.

    One page that fails does not abort the run: its place in the output carries an
    explicit failure marker and the other pages are still processed. A route-level
    failure does abort the run, because every page would fail the same way.

    Args:
        doc: The PDF document object
        model_id: The model identifier to use
        base_url: OCR endpoint root URL; requests go to ``{base_url}/v1``
        prompt_template: The prompt template with system, user and extra_body
        page_indices: List of 0-based page indices to process. If None, all pages are processed.

    Returns:
        list: List of extracted text for each page

    Raises:
        NoOcrSupportError: If the endpoint rejects the model for OCR.
    """
    if page_indices is None:
        page_indices = list(range(doc.page_count))

    client = OpenAI(
        api_key="EMPTY", base_url=f"{base_url}/v1", timeout=REQUEST_TIMEOUT_SECONDS
    )

    results = []
    failed_pages = []
    for page_num in page_indices:
        try:
            result_text = _ocr_page(client, doc[page_num], model_id, prompt_template)
        except NoOcrSupportError:
            raise
        except Exception as e:
            logger.error(f"Page {page_num + 1} failed: {e}")
            results.append(f"--- Page {page_num + 1} ---\n[OCR failed: {e}]")
            failed_pages.append(page_num + 1)
            continue

        results.append(f"--- Page {page_num + 1} ---\n{result_text}")

    if failed_pages:
        logger.warning(f"Pages that failed OCR: {failed_pages}")
        raise PageFailureError(
            f"OCR failed for pages: {failed_pages}", results
        )

    return results


def process_with_model(
    pdf_path: str,
    model_id: str,
    base_url: str,
    page_spec: Optional[str] = None,
    prompt_family: Optional[str] = None,
) -> str:
    """
    Process a PDF through a route on an OpenAI-compatible endpoint.

    The request goes to ``{base_url}/v1/chat/completions`` with the model in the
    body, so the endpoint is any OpenAI-compatible service. The only difference
    between OCR models is the prompt template, which the route's prompt family
    selects (get_prompt_template()).

    Args:
        pdf_path: Path to the PDF file
        model_id: The model ID to use
        base_url: Base URL for the OCR endpoint
        page_spec: Page specification string (e.g., "1-5", "1,3,5"). If None, all pages are processed.
        prompt_family: Optional prompt family from the route.

    Returns:
        str: Extracted text from the PDF

    Raises:
        PdfOcrError: If PDF processing fails.
    """
    logger.info(f"Processing PDF with model ({model_id}): {pdf_path}")

    doc = None
    try:
        doc = pymupdf.open(pdf_path)
        if doc.page_count == 0:
            raise PdfOcrError("PDF has no pages")

        page_indices = parse_page_spec(page_spec, doc.page_count)
        prompt_template = get_prompt_template(model_id, prompt_family)

        return "\n\n".join(
            process_pdf_pages(doc, model_id, base_url, prompt_template, page_indices)
        )

    finally:
        if doc is not None:
            doc.close()


def load_route(config_path: Optional[Path] = None) -> dict:
    """
    Load the OCR route from the route file.

    The file holds one route:

        { "model": "...", "base_url": "...", "prompt": "deepseek-ocr" | "generic" }

    Args:
        config_path: Path to the route file. If None, uses the default location.

    Returns:
        dict: The route

    Raises:
        ConfigurationError: If the file is missing, unreadable or not an object.
    """
    path = resolve_routing_config_path(config_path)

    if not path.exists():
        raise ConfigurationError(f"no OCR route configured: {path} not found")

    try:
        with open(path, "r") as f:
            route = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"cannot read {path}: {e}") from e

    if not isinstance(route, dict):
        raise ConfigurationError(f"{path} must contain a route object")

    return route


def parse_route_spec(spec: str) -> tuple:
    """
    Parse an ad-hoc route specification.

    ``model@base_url`` sets both; a bare value sets the model only.

    Returns:
        tuple: ``(model, base_url)``, either of which may be None

    Raises:
        ConfigurationError: If the specification is empty.
    """
    model, _, base_url = spec.partition("@")
    model = model.strip() or None
    base_url = base_url.strip().rstrip("/") or None
    if model is None and base_url is None:
        raise ConfigurationError("empty --route value")
    return model, base_url


def resolve_route(
    config_path: Optional[Path] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional[str] = None,
) -> tuple:
    """
    Resolve the route to ``(model_id, base_url, prompt_family)``.

    Explicit values override the route file, so an ad-hoc run needs no file edit.
    There is no discovery and no selection: the model and the endpoint both come
    from the route or from the caller.

    Raises:
        ConfigurationError: If the resolved route is incomplete.
    """
    route = load_route(config_path)

    model_id = model or route.get("model")
    route_base_url = base_url or route.get("base_url")
    prompt_family = prompt or route.get("prompt")

    if not model_id:
        raise ConfigurationError("the OCR route needs a 'model'")
    if not route_base_url:
        raise ConfigurationError("the OCR route needs a 'base_url'")

    return model_id, str(route_base_url).rstrip("/"), prompt_family


def route_ocr_request(
    pdf_path: str,
    page_spec: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    prompt: Optional[str] = None,
    routing_config_path: Optional[str] = None,
) -> str:
    """
    OCR a PDF through the configured route.

    The route comes from the route file, overridden by any explicit values. The
    tool never asks the endpoint what it has loaded, so a run is deterministic:
    the same route every time.

    Args:
        pdf_path: Path to the PDF file
        page_spec: Page specification string (e.g., "1-5", "1,3,5"). If None, all pages are processed.
        model: Model id, overriding the route file.
        base_url: OCR endpoint root URL, overriding the route file.
        prompt: Prompt family, overriding the route file.
        routing_config_path: Path to the route file. Overrides the
            PDF_OCR_ROUTING_CONFIG env var and the tool-dir default when set.

    Returns:
        str: OCR result text

    Raises:
        ConfigurationError: If no usable route is configured.
        NoOcrSupportError: If the endpoint rejects the configured model for OCR.
        OcrRequestError: If the endpoint fails.
    """
    model_id, route_base_url, prompt_family = resolve_route(
        Path(routing_config_path) if routing_config_path else None,
        model=model,
        base_url=base_url,
        prompt=prompt,
    )

    logger.info(f"Routing to {model_id} at {route_base_url}")

    return process_with_model(
        pdf_path, model_id, route_base_url, page_spec, prompt_family
    )


def build_mcp_server(routing_config_path: Optional[str] = None):
    """Build the stdio MCP server that exposes the ``pdf_ocr`` tool.

    The tool calls the library in-process, so there is no subprocess hop and no
    exit-code translation: the typed errors map straight to tool errors. The MCP
    import stays here, so a plain CLI run needs no MCP package.
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = MCPServer("pdf-ocr")

    @mcp.tool()
    def pdf_ocr(pdf_path: str, page: str | None = None) -> str:
        """Extract text from a PDF using the configured OCR route.

        Args:
            pdf_path: Absolute path to a local PDF, or an http(s) URL to download.
            page: Optional page selection: '5', '1-5', '1,3,5', or '1-3,5,7-9'.
                Omit to process every page.
        """
        temp_pdf_path = None
        try:
            if is_url(pdf_path):
                temp_pdf_path = download_pdf(pdf_path)
                pdf_path = temp_pdf_path
            elif not Path(pdf_path).exists():
                raise ToolError(f"PDF file not found: {pdf_path}")
            return route_ocr_request(pdf_path, page, routing_config_path=routing_config_path)
        except PageFailureError as e:
            # Some pages worked. The tool call fails, but the good pages are still
            # returned, because discarding them loses real OCR output.
            partial = "\n\n".join(e.results)
            raise ToolError(f"{e}\n\nPartial output:\n{partial}") from e
        except PdfOcrError as e:
            raise ToolError(str(e)) from e
        finally:
            if temp_pdf_path:
                cleanup_temp_file(temp_pdf_path)

    return mcp


def run_mcp_server(routing_config_path: Optional[str] = None) -> None:
    """Serve the ``pdf_ocr`` tool over stdio. Selected by ``--mcp``."""
    build_mcp_server(routing_config_path).run()


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF through an OpenAI-compatible OCR endpoint"
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        help="Absolute path to PDF file or URL to download from",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Serve the pdf_ocr tool over stdio instead of OCRing one PDF",
    )
    parser.add_argument(
        "--page",
        help="Page(s) to OCR: single page ('5'), range ('1-5'), multiple ('1,3,5'), or mixed ('1-3,5,7-9'). If omitted, all pages are processed.",
    )
    parser.add_argument(
        "--route",
        help="Ad-hoc route as model@base_url, or a bare model id (overrides the route file)",
    )
    parser.add_argument(
        "--model",
        help="Model id (overrides the route file)",
    )
    parser.add_argument(
        "--base-url",
        help="OCR endpoint root URL, without a /v1 suffix (overrides the route file)",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt family: deepseek-ocr or generic (overrides the route file)",
    )
    parser.add_argument(
        "--routing-config",
        help=(
            "Path to the route file (overrides the PDF_OCR_ROUTING_CONFIG "
            "env var and the ocr_routing.json next to this script)"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    if args.mcp:
        run_mcp_server(args.routing_config)
        return

    if not args.pdf_path:
        parser.error("pdf_path is required unless --mcp is given")

    pdf_path = args.pdf_path
    temp_pdf_path = None

    # Handle URL downloads
    if is_url(pdf_path):
        try:
            temp_pdf_path = download_pdf(pdf_path)
            pdf_path = temp_pdf_path
        except DownloadError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    try:
        model = args.model
        base_url = args.base_url
        if args.route:
            route_model, route_base_url = parse_route_spec(args.route)
            model = route_model or model
            base_url = route_base_url or base_url

        output = route_ocr_request(
            pdf_path,
            args.page,
            model=model,
            base_url=base_url,
            prompt=args.prompt,
            routing_config_path=args.routing_config,
        )
        print(output)
    except NoOcrSupportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)
    except PageFailureError as e:
        # The pages that succeeded are still worth printing.
        print("\n\n".join(e.results))
        print(f"Error processing PDF: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing PDF: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if temp_pdf_path:
            cleanup_temp_file(temp_pdf_path)


if __name__ == "__main__":
    main()
