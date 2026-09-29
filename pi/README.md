# pi pdf-ocr

A [pi](https://pi.dev) extension that registers a `pdf_ocr` tool for transcribing
PDF documents via OCR. It shells out to the harness-agnostic Python backend
(`pdf_ocr_backend.py`), which converts PDF pages to images and routes them to an
OCR model (DeepSeek-OCR / GLM-OCR / the currently loaded multimodal model) via a
llama-swap proxy.

This is the pi packaging of the same backend [opencode-ocr](../) uses. Each
harness deploys its own private copy of the backend — they share nothing at
runtime.

## Install

From the opencode-ocr repo root:

```bash
# pi only (skip OpenCode)
./deploy-tool.sh --target pi

# or both harnesses (default)
./deploy-tool.sh
```

This deploys:

- the backend → `~/.config/pi/tool/` (`pdf_ocr_backend.py`, `pyproject.toml`,
  a generated `.env`, and `ocr_routing.json`)
- this extension → `~/.pi/agent/extensions/pdf-ocr/` (pi's global auto-discovery
  location)

Then `/reload` in pi to load the `pdf_ocr` tool.

To force a reinstall of the tool files:

```bash
./deploy-tool.sh --target pi --force
```

## Tool parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `pdf_path` | yes | Path to PDF (absolute, relative to cwd, or a `http(s)://` URL to download) |
| `output_format` | no | `"markdown"` (default) or `"text"` |
| `page` | no | Pages to OCR: `"5"`, `"1-5"`, `"1,3,5"`, or `"1-3,5,7-9"`. Omit = all pages |

## Configuration

The backend reads its configuration from `~/.config/pi/tool/`:

- **`.env`** — `DEEPSEEK_OCR_BASE_URL` (the llama-swap proxy endpoint).
- **`ocr_routing.json`** — maps loaded model sets to an OCR method. See
  `ocr_routing.json.example` for the full schema and examples.

Override the backend directory with `PDF_OCR_TOOL_DIR` (the extension and the
backend both honor it for discovery).

## Exit codes

The backend's exit codes surface as tool failures (pi marks a non-zero exit as
an error):

| Exit | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (file not found, API error, OCR model failure) |
| 3 | `NO_OCR_SUPPORT` — routed to `current_model` but the model lacks vision |

See the repo [README](../../README.md) for routing details and troubleshooting.