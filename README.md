# PDF OCR Tool

An OCR backend for PDF files. Converts PDFs to high-quality page images, performs
OCR on each page via an OpenAI-compatible model endpoint, and returns markdown
output. Supports dedicated OCR models and multimodal vision models, selected per
loaded-model set.

`tool/pdf_ocr_backend.py` is the tool. Host adapters live beside it and
only wrap the backend; see [Compatibility](#compatibility) for which adapter goes
where.

## Installation

The backend is a `uv` project at the repository root:

```bash
uv sync
```

That is all the backend needs. A host adapter may also copy the tool into the
directory that host expects — see [Compatibility](#compatibility).

## Usage

**Important**: Python scripts must be run using `uv run` to ensure proper dependency management:

```bash
# Direct backend execution (with .env file)
uv run --env-file .env python tool/pdf_ocr_backend.py <pdf_path> <output_format>
```

A host adapter normally invokes this for you (see [Compatibility](#compatibility)).

### MCP server

`tool/pdf_ocr_mcp.py` is a stdio MCP server exposing one tool,
`pdf_ocr(pdf_path, page=None)`. It runs the backend, so the routing and prompt
logic stay in one place.

```bash
# The venv must already exist (host-side `uv sync`); MCP hosts cannot install at runtime.
uv run --frozen python tool/pdf_ocr_mcp.py
```

Environment:

- `DEEPSEEK_OCR_BASE_URL` — OCR endpoint **root** URL, no `/v1` suffix (required).
- `PDF_OCR_ROUTING_CONFIG` — path to the routing JSON (optional; defaults to
  `ocr_routing.json` next to the backend).

If the endpoint is single-tenant, pin the routing table to `current_model` so OCR
reuses the already-loaded model instead of requesting a different one.

## Parameters

- `pdf_path`: Absolute path to a PDF file, or an `http(s)` URL to download
- `output_format`: Output format - "markdown" or "text" (defaults to "markdown")
- `page`: Optional page selection - `5`, `1-5`, `1,3,5`, or `1-3,5,7-9` (defaults to all pages)

## Dependencies

- openai>=1.0.0
- PyMuPDF>=1.23.0
- Pillow>=10.0.0

## Configuration

### Endpoint Configuration

The tool connects to an OpenAI-compatible endpoint. The endpoint can be configured in three ways:

1. **.env file** (recommended for persistent configuration):
   Copy `.env.example` to `.env` and edit it:
   ```bash
   cp .env.example .env
   # Edit .env with your endpoint URL
   ```
   Then run with `uv run --env-file .env`.

2. **Environment variable**:
   ```bash
   export DEEPSEEK_OCR_BASE_URL="http://your-endpoint:8080"
   ```

3. **Command-line argument** (overrides both above):
   ```bash
   uv run python tool/pdf_ocr_backend.py <pdf_path> <output_format> --base-url http://your-endpoint:8080
   ```

The base URL is the endpoint **root** — do **not** append `/v1`. The backend
derives `/running`, `/upstream/{model}/props`, and `/upstream/{model}/v1` itself;
a `/v1` suffix makes model detection 404.

If none of these are set, the tool will throw an error.

### OCR Routing Configuration

The tool uses model-based routing to determine which OCR method to use. This is configured in `ocr_routing.json`:

**Location**: `ocr_routing.json` next to the backend. An alternative path can be
supplied via the `PDF_OCR_ROUTING_CONFIG` environment variable or the
`--routing-config` argument (the argument wins).

**Structure**:
```json
{
  "_comment": "OCR Routing Configuration - Maps loaded model sets to preferred OCR method. Single-model keys work as before. Comma-separated keys match ALL loaded models (order-independent).",
  "_routing_options": {
    "deepseek-ocr": "Use DeepSeek-OCR model",
    "deepseek-ocr-2": "Use DeepSeek-OCR-2 model",
    "glm-ocr": "Use GLM-OCR model",
    "current_model": "Use the currently loaded model (requires vision support)"
  },
  "ocr_routing": {
    "_comment": "Model-set examples: comma-separated keys match ALL loaded models",
    "primary-model,secondary-model-a": "deepseek-ocr-2",
    "primary-model,secondary-model-b": "glm-ocr",
    "primary-model": "current_model"
  },
  "default": "current_model"
}
```

**Routing Options**:
- `deepseek-ocr`: Use the dedicated DeepSeek-OCR model for OCR tasks
- `deepseek-ocr-2`: Use the DeepSeek-OCR-2 model (improved accuracy)
- `glm-ocr`: Use the GLM-OCR model
- `current_model`: Use the currently loaded model (requires vision/multimodal support)

**Matching Logic** (most specific first):
1. **Exact model-set match**: Comma-separated key matches all loaded models exactly (order-independent)
2. **Single-model match**: Full model ID or partial match in `ocr_routing`
3. **Default**: Falls back to `default` value if no match found

**Single-Model Keys** (backward compatible):
Keys without commas work exactly as before:
```json
{
  "ocr_routing": {
    "primary-model": "deepseek-ocr-2",
    "another-model": "glm-ocr"
  },
  "default": "current_model"
}
```

**Model-Set Keys**:
When multiple models are loaded simultaneously, you can route based on the complete set:
```json
{
  "ocr_routing": {
    "primary-model,secondary-model-a": "deepseek-ocr-2",
    "primary-model,secondary-model-b": "glm-ocr",
    "primary-model": "current_model"
  },
  "default": "current_model"
}
```
In this example:
- When `primary-model` + `secondary-model-a` are loaded → uses `deepseek-ocr-2`
- When `primary-model` + `secondary-model-b` are loaded → uses `glm-ocr`
- When only `primary-model` is loaded → uses `current_model` (vision check required)

Always use current model (disable dedicated OCR):
```json
{
  "ocr_routing": {},
  "default": "current_model"
}
```

Enable DeepSeek-OCR for all models by default:
```json
{
  "ocr_routing": {},
  "default": "deepseek-ocr"
}
```

For more examples, see `ocr_routing.json.example` in the repository.

## Technical Details

- PDF-to-image conversion at 144 DPI (high quality for OCR)
- PNG format with RGB color space
- Sequential page processing for memory management
- OCR parameters: temperature=0.0, max_tokens=8192, ngram_size=30, window_size=90

## Exit Codes

| Exit Code | Meaning | Scenario |
|-----------|---------|----------|
| **0** | Success | OCR completed successfully using either a dedicated OCR model or a multimodal model |
| **1** | General Error | File not found, API error, processing error, or OCR model failure |
| **3** | NO_OCR_SUPPORT | Current model is configured to use `current_model` routing but lacks vision/multimodal support |

## Troubleshooting

### Exit Code 3: NO_OCR_SUPPORT

This error occurs when:
- The current model is configured to use `current_model` for OCR routing
- The model does not support multimodal/vision capabilities

**Solution**: Add a routing rule for the loaded model(s) to `ocr_routing.json`:

For a single model:
```json
{
  "ocr_routing": {
    "your-model-name": "deepseek-ocr-2"
  },
  "default": "current_model"
}
```

For a model set (when multiple models are loaded):
```json
{
  "ocr_routing": {
    "your-model,secondary-model": "glm-ocr"
  },
  "default": "current_model"
}
```

Or switch to a model with vision support.

### OCR Model Failures (Exit Code 1)

If a dedicated OCR model is configured but fails:
- Verify the endpoint is running and accessible
- Check that the OCR model is loaded at the endpoint
- Review endpoint logs for errors

### Configuration File Not Found

If `ocr_routing.json` is missing, the tool will use defaults (`current_model`).
Create it next to the backend, using `ocr_routing.json.example` as a starting
point.

## Compatibility

The backend is host-agnostic. This is the one place host-specific names appear:

| Host | Adapter | Install location |
|---|---|---|
| OpenCode | `tool/pdf-ocr.ts` | `~/.config/opencode/tool/` |
| MCP clients (dsh, ...) | `tool/pdf_ocr_mcp.py` | anywhere; the host points at it |
| pi | `pi/` | `~/.pi/agent/extensions/pdf-ocr/` + `~/.config/pi/tool/` |

`deploy-tool.sh` deploys to either or both hosts. Each host owns its own `.env`
and `ocr_routing.json`; nothing is shared at runtime.

```bash
./deploy-tool.sh                  # both (default)
./deploy-tool.sh --target opencode
./deploy-tool.sh --target pi
./deploy-tool.sh --force          # force reinstall
./deploy-tool.sh --repo <path>    # use a different checkout
```

### OpenCode

Copies the tool into `~/.config/opencode/tool/` and runs `uv sync` there:

```bash
mkdir -p ~/.config/opencode/tool/
cp tool/pdf-ocr.ts tool/pdf_ocr_backend.py pyproject.toml ~/.config/opencode/tool/
cd ~/.config/opencode/tool && uv sync
```

### pi

Copies the backend into `~/.config/pi/tool/` (override with `PDF_OCR_TOOL_DIR`)
and the extension into `~/.pi/agent/extensions/pdf-ocr/`. Run `/reload` in pi
afterwards to load the `pdf_ocr` tool. See `pi/README.md`.

### MCP clients

Register `tool/pdf_ocr_mcp.py` as a stdio server (see
[MCP server](#mcp-server)). The venv must already exist; MCP hosts cannot install
dependencies at runtime.

## Future Enhancements

- Partial page range support (e.g., process pages 5-10 only)
- Progress reporting during OCR processing
- Batch processing of multiple PDFs
- Additional output formats (e.g., JSON, HTML)
- Image quality settings configuration
