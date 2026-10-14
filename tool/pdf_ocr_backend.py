#!/usr/bin/env python3
import sys
import os
import json
from typing import Optional
import fitz
from openai import OpenAI
from pathlib import Path
import base64
import argparse
from dotenv import load_dotenv
import logging
import requests
import tempfile
import urllib.parse

# Load environment variables from .env file in the tool directory
tool_dir = Path(__file__).parent
env_path = tool_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Name of the environment variable that points at a non-default routing config.
# An MCP server (or another host adapter) sets this so it can ship a
# deployment-specific ocr_routing.json without copying the whole tool.
ROUTING_CONFIG_ENV = "PDF_OCR_ROUTING_CONFIG"


def resolve_routing_config_path(config_path: Optional[Path] = None) -> Path:
    """Resolve the routing config path: explicit arg > env var > tool dir."""
    if config_path is not None:
        return Path(config_path)
    env_path = os.getenv(ROUTING_CONFIG_ENV)
    if env_path:
        return Path(env_path)
    return Path(__file__).parent / "ocr_routing.json"


def validate_environment_variables():
    """Validate environment variables for configuration."""
    errors = []
    warnings = []

    # Validate DEEPSEEK_OCR_BASE_URL (required)
    deepseek_url = os.getenv("DEEPSEEK_OCR_BASE_URL")
    if deepseek_url and not deepseek_url.startswith(("http://", "https://")):
        errors.append(
            f"DEEPSEEK_OCR_BASE_URL must be a valid URL starting with http:// or https://: {deepseek_url}"
        )

    # Check for routing config file (warning only if missing)
    config_path = resolve_routing_config_path()
    if not config_path.exists():
        warnings.append(f"Routing config not found at {config_path}, using defaults")

    return errors, warnings


# Run validation on module load
_validation_errors, _validation_warnings = validate_environment_variables()

if _validation_warnings:
    for warning in _validation_warnings:
        print(f"Warning: {warning}", file=sys.stderr)

if _validation_errors:
    for error in _validation_errors:
        print(f"Configuration Error: {error}", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Maximum downloadable PDF size (50 MB)
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024


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
        Exception: If download fails, size exceeds limit, or content is not a valid PDF
    """
    logger.info(f"Downloading PDF from: {url}")

    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download PDF from {url}: {e}")

    # Check Content-Length if available
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            total_size = int(content_length)
            if total_size > max_size:
                raise Exception(
                    f"PDF file too large ({total_size / 1024 / 1024:.1f} MB). "
                    f"Maximum allowed size is {max_size / 1024 / 1024:.0f} MB."
                )
        except ValueError:
            pass  # Ignore invalid Content-Length

    # Stream download with size limit
    downloaded_size = 0
    chunks = []
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            downloaded_size += len(chunk)
            if downloaded_size > max_size:
                raise Exception(
                    f"PDF file exceeds maximum allowed size of {max_size / 1024 / 1024:.0f} MB."
                )
            chunks.append(chunk)

    pdf_bytes = b"".join(chunks)

    if not pdf_bytes:
        raise Exception(f"Downloaded PDF from {url} is empty.")

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
        doc = fitz.open(temp_file_path)
        page_count = doc.page_count
        if page_count == 0:
            raise Exception("Downloaded file is not a valid PDF (no pages).")
    except Exception as e:
        # Clean up temp file on validation failure
        try:
            os.unlink(temp_file_path)
        except OSError:
            pass
        if "no pages" in str(e).lower():
            raise
        raise Exception(f"Downloaded file is not a valid PDF: {e}")
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


def get_all_models(base_url: str) -> list:
    """
    Query the endpoint's /running API for all currently loaded models.

    Args:
        base_url: Base URL for the OCR endpoint

    Returns:
        list: List of model ID strings currently loaded (e.g.,
              ["kimi-k2.6", "qwen3.6-35b-a3b-nvfp4"]).
              Returns empty list if no models are loaded or on error.
    """
    try:
        url = f"{base_url}/running"
        logger.info(f"Querying all models from: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        models = []

        # Handle different response formats
        if isinstance(data, list):
            # List format - extract model from each item
            for item in data:
                model_id = item.get("model") or item.get("id")
                if model_id:
                    models.append(model_id)
        elif isinstance(data, dict):
            # Dict format: check for "running" key (proxy response)
            if "running" in data and isinstance(data["running"], list):
                for item in data["running"]:
                    model_id = item.get("model") or item.get("id")
                    if model_id:
                        models.append(model_id)
            # Direct dict format - check for model field
            else:
                model_id = data.get("model") or data.get("id")
                if model_id:
                    models.append(model_id)

        if models:
            logger.info(f"Loaded models detected: {models}")
        else:
            logger.warning("No models currently loaded")

        return models

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection failed to {base_url}/running: {e}")
        return []
    except requests.exceptions.Timeout:
        logger.error(f"Timeout querying {base_url}/running")
        return []
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except (KeyError, ValueError) as e:
        logger.error(f"Failed to parse response: {e}")
        return []


def get_current_model(base_url: str) -> Optional[str]:
    """
    Query ik_llama.cpp /running endpoint for currently loaded model.

    Args:
        base_url: Base URL for the ik_llama.cpp endpoint

    Returns:
        Optional[str]: Model ID string if a model is loaded, None otherwise
    """
    models = get_all_models(base_url)
    if models:
        model_id = models[0]
        logger.info(f"Current model detected: {model_id}")
        return model_id
    return None


def check_multimodal_support(base_url: str, model_id: str) -> bool:
    """
    Query /upstream/{model_id}/props endpoint to check for modalities.vision support.

    Args:
        base_url: Base URL for the ik_llama.cpp endpoint
        model_id: ID of the model to check

    Returns:
        bool: True if model supports vision/multimodal, False otherwise
    """
    try:
        url = f"{base_url}/upstream/{model_id}/props"
        logger.info(f"Checking multimodal support for {model_id} at: {url}")

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Check modalities.vision field
        modalities = data.get("modalities", {})
        vision_support = modalities.get("vision", False)

        if vision_support:
            logger.info(f"Model {model_id} supports vision/multimodal")
        else:
            logger.info(f"Model {model_id} does not support vision/multimodal")

        return vision_support

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection failed to {base_url}/upstream/{model_id}/props: {e}")
        return False
    except requests.exceptions.Timeout:
        logger.error(f"Timeout querying {base_url}/upstream/{model_id}/props")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False
    except (KeyError, ValueError) as e:
        logger.error(f"Failed to parse response: {e}")
        return False


# Define prompt templates for different models
PROMPT_TEMPLATES = {
    "deepseek-ocr": {
        "system": None,
        "user": "Free OCR.",
        "extra_body": {
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 30,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    },
    "default_multimodal": {
        "system": "You are an OCR assistant. Extract all text from the image accurately.",
        "user": "Please perform OCR on this image and extract all visible text. Format the output as markdown.",
        "extra_body": None,
    },
}


def get_prompt_template(model_id: str) -> dict:
    """
    Get the appropriate prompt template for a given model ID.

    Args:
        model_id: The model identifier string

    Returns:
        dict: Prompt template with system, user, and extra_body keys
    """
    if "deepseek-ocr" in model_id.lower():
        return PROMPT_TEMPLATES["deepseek-ocr"]
    return PROMPT_TEMPLATES["default_multimodal"]


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
    parts = page_spec.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Range specification
            range_parts = part.split("-")
            if len(range_parts) != 2:
                raise ValueError(f"Invalid range specification: {part}")
            try:
                start = int(range_parts[0].strip())
                end = int(range_parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid page numbers in range: {part}")

            if start < 1 or end < 1:
                raise ValueError(f"Page numbers must be positive: {part}")
            if start > end:
                raise ValueError(f"Invalid range (start > end): {part}")
            if start > total_pages or end > total_pages:
                raise ValueError(
                    f"Page number exceeds document length ({total_pages} pages): {part}"
                )

            # Convert to 0-based indices
            for page_num in range(start, end + 1):
                pages.add(page_num - 1)
        else:
            # Single page
            try:
                page_num = int(part)
            except ValueError:
                raise ValueError(f"Invalid page number: {part}")

            if page_num < 1:
                raise ValueError(f"Page number must be positive: {page_num}")
            if page_num > total_pages:
                raise ValueError(
                    f"Page number {page_num} exceeds document length ({total_pages} pages)"
                )

            pages.add(page_num - 1)

    return sorted(list(pages))


def process_pdf_pages(
    doc: fitz.Document,
    client: OpenAI,
    model_id: str,
    prompt_template: dict,
    page_indices: Optional[list] = None,
) -> list:
    """
    Process pages in a PDF document using the specified model and prompt template.

    Args:
        doc: The PDF document object
        client: The OpenAI client instance
        model_id: The model identifier to use
        prompt_template: The prompt template with system, user, and extra_body
        page_indices: List of 0-based page indices to process. If None, all pages are processed.

    Returns:
        list: List of extracted text for each page
    """
    results = []
    system_prompt = prompt_template.get("system")
    user_prompt = prompt_template["user"]
    extra_body = prompt_template.get("extra_body")

    if page_indices is None:
        page_indices = list(range(doc.page_count))

    for page_num in page_indices:
        page = doc[page_num]
        pix = page.get_pixmap(dpi=144, colorspace=fitz.csRGB)
        img_bytes = pix.tobytes("png")
        pix = None
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
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
            "temperature": 0.0,
        }

        if extra_body:
            api_params["extra_body"] = extra_body

        response = client.chat.completions.create(**api_params)

        if not response.choices or len(response.choices) == 0:
            raise Exception(f"No OCR response for page {page_num + 1}")

        result_text = response.choices[0].message.content
        
        if result_text and extra_body and extra_body.get("skip_special_tokens") is False:
            result_text = result_text.replace("\u0120", " ").replace("\u010a", "\n")

        results.append(f"--- Page {page_num + 1} ---\n{result_text}")

        img_data_url = None
        img_base64 = None

    return results


def process_with_model(
    pdf_path: str,
    output_format: str,
    model_id: str,
    base_url: str,
    page_indices: Optional[list] = None,
) -> str:
    """
    Process PDF using any model through an OpenAI-compatible endpoint.

    The request goes to ``{base_url}/v1/chat/completions`` with the model in the
    body. That is the OpenAI-compatible route every provider serves, and the
    model id in the body is what selects the model.

    The only difference between OCR models is the prompt template
    (handled by get_prompt_template()), not the endpoint or processing logic.

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format (markdown or text)
        model_id: The model ID to use (e.g., "deepseek-ocr", "deepseek-ocr-2", "glm-ocr")
        base_url: Base URL for the OCR endpoint
        page_indices: List of 0-based page indices to process. If None, all pages are processed.

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If PDF processing fails
    """
    logger.info(f"Processing PDF with model ({model_id}): {pdf_path}")

    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            raise Exception("PDF has no pages")

        client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1", timeout=3600)
        prompt_template = get_prompt_template(model_id)

        results = process_pdf_pages(
            doc, client, model_id, prompt_template, page_indices
        )

        output = "\n\n".join(results)
        return output

    finally:
        if doc is not None:
            doc.close()


def process_with_deepseek_ocr(
    pdf_path: str,
    output_format: str,
    page_indices: Optional[list] = None,
    model_name: str = "deepseek-ocr",
) -> str:
    """
    Process PDF using DeepSeek-OCR model.

    Kept for backward compatibility. Delegates to process_with_model().

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format (markdown or text)
        page_indices: List of 0-based page indices to process. If None, all pages are processed.
        model_name: The name of the DeepSeek-OCR model to use (e.g., "deepseek-ocr" or "deepseek-ocr-2")

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If PDF processing fails
    """
    base_url = os.getenv("DEEPSEEK_OCR_BASE_URL")
    if not base_url:
        raise Exception(
            "DEEPSEEK_OCR_BASE_URL not set. Set it via environment variable or .env file"
        )

    return process_with_model(pdf_path, output_format, model_name, base_url, page_indices)


def process_with_current_model(
    pdf_path: str,
    output_format: str,
    model_id: str,
    base_url: str,
    page_indices: Optional[list] = None,
) -> str:
    """
    Process PDF using the currently loaded multimodal model.

    Kept for backward compatibility. Delegates to process_with_model().

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format (markdown or text)
        model_id: The ID of the currently loaded model
        base_url: Base URL for the API endpoint
        page_indices: List of 0-based page indices to process. If None, all pages are processed.

    Returns:
        str: Extracted text from the PDF

    Raises:
        Exception: If PDF processing fails
    """
    return process_with_model(pdf_path, output_format, model_id, base_url, page_indices)


def load_ocr_routing_config(config_path: Optional[Path] = None) -> dict:
    """
    Load OCR routing configuration from JSON file.

    Args:
        config_path: Path to the routing config file. If None, uses default location.

    Returns:
        dict: Routing configuration with model mappings
    """
    config_path = resolve_routing_config_path(config_path)

    default_config = {"ocr_routing": {}, "default": "current_model"}

    if not config_path.exists():
        logger.warning(f"Routing config not found at {config_path}, using defaults")
        return default_config

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate config structure
        if "ocr_routing" not in config:
            config["ocr_routing"] = {}
        if "default" not in config:
            config["default"] = "current_model"

        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in routing config: {e}")
        return default_config
    except Exception as e:
        logger.error(f"Error loading routing config: {e}")
        return default_config


def get_ocr_method_for_model(model_id: str, config: dict):
    """
    Determine OCR route for a given model based on routing configuration.

    Args:
        model_id: The current model ID
        config: The routing configuration dictionary

    Returns:
        str | dict: the configured route - a model id string, a route object, or
            "current_model"
    """
    routing = config.get("ocr_routing", {})
    default = config.get("default", "current_model")

    # Check for exact match
    if model_id in routing:
        return routing[model_id]

    # Check for partial match (e.g., "kimi-k2.5" matches "ik_llama.cpp/kimi-k2.5-experimental")
    for pattern, method in routing.items():
        if pattern in model_id or model_id in pattern:
            return method

    # Return default
    return default


def get_ocr_method_for_model_set(model_ids: list, config: dict):
    """
    Determine OCR route for a set of loaded models.

    Matching priority:
    1. Exact model-set match (comma-separated key in config)
    2. Single-model partial match (for each loaded model)
    3. Default fallback

    Args:
        model_ids: List of currently loaded model IDs
        config: The routing configuration dictionary

    Returns:
        str | dict: OCR route to use (see get_ocr_method_for_model)
    """
    routing = config.get("ocr_routing", {})
    default = config.get("default", "current_model")

    # Normalize loaded models to lowercase set
    loaded_set = set(m.strip().lower() for m in model_ids)

    # 1. Try exact model-set match (multi-model keys)
    for pattern, method in routing.items():
        if "," in pattern:
            pattern_set = set(m.strip().lower() for m in pattern.split(","))
            if loaded_set == pattern_set:
                return method

    # 2. Single-model matching (backward compatible)
    for model_id in model_ids:
        method = get_ocr_method_for_model(model_id, config)
        if method != default:
            return method

    # 3. Return default
    return default


def resolve_ocr_route(ocr_method, default_base_url: Optional[str] = None):
    """
    Normalise a routing-config value into ``(model_id, base_url)``.

    A string value is a model id served by ``default_base_url``. An object value
    may override the endpoint:

        { "model": "...", "base_url": "..." }

    ``base_url`` defaults to ``default_base_url`` (``DEEPSEEK_OCR_BASE_URL``).
    The request goes to ``{base_url}/v1`` with the model in the body, so the tool
    is generic: the endpoint is any OpenAI-compatible service.

    Args:
        ocr_method: A routing-config value: a model id string, or a route object.
        default_base_url: Base URL used when the route does not set one.

    Returns:
        tuple: ``(model_id, base_url)``
    """
    if isinstance(ocr_method, dict):
        model_id = ocr_method.get("model")
        if not model_id:
            raise Exception(f"routing config route object is missing 'model': {ocr_method!r}")
        base_url = ocr_method.get("base_url") or default_base_url
    else:
        model_id = ocr_method
        base_url = default_base_url
    if not base_url:
        raise Exception("no base URL for the OCR route (set DEEPSEEK_OCR_BASE_URL or the route's base_url)")
    return model_id, base_url


def route_ocr_request(
    pdf_path: str,
    output_format: str,
    page_spec: Optional[str] = None,
    base_url: Optional[str] = None,
    routing_config_path: Optional[str] = None,
) -> str:
    """
    Route OCR request based on model-set-based configuration.

    Args:
        pdf_path: Path to the PDF file
        output_format: Output format (markdown or text)
        page_spec: Page specification string (e.g., "1-5", "1,3,5"). If None, all pages are processed.
        base_url: OCR endpoint root URL. Overrides DEEPSEEK_OCR_BASE_URL when set.
        routing_config_path: Path to the routing JSON. Overrides the
            PDF_OCR_ROUTING_CONFIG env var and the tool-dir default when set.

    Returns:
        str: OCR result text

    Exits:
        0: Success (OCR model or current multimodal model used)
        1: General error (file not found, processing error, OCR model failure, etc.)
        3: NO_OCR_SUPPORT (current_model routing but model lacks vision support)
    """
    if not base_url:
        base_url = os.getenv("DEEPSEEK_OCR_BASE_URL")
    if not base_url:
        raise Exception("DEEPSEEK_OCR_BASE_URL not set")

    # Load routing configuration
    routing_config = load_ocr_routing_config(
        Path(routing_config_path) if routing_config_path else None
    )

    # Get ALL loaded models (not just the first one)
    loaded_models = get_all_models(base_url)
    if not loaded_models:
        raise Exception("No model currently loaded")

    # Determine OCR method based on model SET
    ocr_method = get_ocr_method_for_model_set(loaded_models, routing_config)

    logger.info(f"Loaded models: {loaded_models}, OCR method: {ocr_method}")

    # Parse page specification
    doc = fitz.open(pdf_path)
    try:
        page_indices = parse_page_spec(page_spec, doc.page_count)
    finally:
        doc.close()

    if ocr_method == "current_model":
        # Use current model - check if it supports vision
        current_model = loaded_models[0]  # Primary model for OCR
        logger.info(f"Routing to current model ({current_model})")

        if check_multimodal_support(base_url, current_model):
            return process_with_model(
                pdf_path, output_format, current_model, base_url, page_indices
            )
        else:
            # Current model doesn't support vision - Exit 3
            error_msg = (
                f"NO_OCR_SUPPORT: Model '{current_model}' is configured to use "
                f"current_model for OCR but does not support multimodal/vision capabilities. "
                f"Add a routing rule for the loaded models ({', '.join(loaded_models)}) "
                f"to ocr_routing.json to use an OCR model instead."
            )
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            sys.exit(3)
    else:
        # A configured OCR route: a model id (string) served by the default base
        # URL, or an object that also sets the endpoint (base_url). The route
        # object is how a non-default endpoint (e.g. the routing coordinator's
        # ocr pool) is configured - with no tool-side special case.
        model_id, route_base_url = resolve_ocr_route(ocr_method, base_url)
        logger.info(f"Routing to {model_id} at {route_base_url} for loaded models: {loaded_models}")
        try:
            return process_with_model(
                pdf_path, output_format, model_id, route_base_url, page_indices
            )
        except Exception as e:
            logger.error(f"{model_id} failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Process PDF using DeepSeek-OCR")
    parser.add_argument("pdf_path", help="Absolute path to PDF file or URL to download from")
    parser.add_argument(
        "output_format",
        nargs="?",
        default="markdown",
        choices=["markdown", "text"],
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--page",
        help="Page(s) to OCR: single page ('5'), range ('1-5'), multiple ('1,3,5'), or mixed ('1-3,5,7-9'). If omitted, all pages are processed.",
    )
    parser.add_argument(
        "--base-url",
        help="OCR endpoint root URL, without a /v1 suffix (overrides DEEPSEEK_OCR_BASE_URL env var)",
    )
    parser.add_argument(
        "--routing-config",
        help=(
            "Path to the OCR routing JSON (overrides the PDF_OCR_ROUTING_CONFIG "
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

    # Configure verbose logging if requested
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    pdf_path = args.pdf_path
    temp_pdf_path = None

    # Handle URL downloads
    if is_url(pdf_path):
        try:
            temp_pdf_path = download_pdf(pdf_path)
            pdf_path = temp_pdf_path
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif not Path(pdf_path).exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        # Use the new routing system
        output = route_ocr_request(
            pdf_path,
            args.output_format,
            args.page,
            base_url=args.base_url,
            routing_config_path=args.routing_config,
        )
        print(output)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)
    finally:
        if temp_pdf_path:
            cleanup_temp_file(temp_pdf_path)


if __name__ == "__main__":
    main()
