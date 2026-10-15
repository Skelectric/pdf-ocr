# pi pdf-ocr

A [pi](https://pi.dev) extension that registers a `pdf_ocr` tool for transcribing
PDF documents via OCR. It shells out to the harness-agnostic Python backend
(`pdf_ocr_backend.py`), which converts PDF pages to images and sends them to the
OpenAI-compatible endpoint named in its route file.

This is the pi packaging of the same backend [pdf-ocr](../) uses. Each
harness deploys its own private copy of the backend — they share nothing at
runtime.

## Install

From the pdf-ocr repo root:

```bash
# pi only (skip OpenCode)
./deploy-tool.sh --target pi

# or both harnesses (default)
./deploy-tool.sh
```

This deploys:

- the backend → `~/.config/pi/tool/` (`pdf_ocr_backend.py`, `pyproject.toml`,
  and `ocr_routing.json`)
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
| `page` | no | Pages to OCR: `"5"`, `"1-5"`, `"1,3,5"`, or `"1-3,5,7-9"`. Omit = all pages |

## Configuration

The backend reads its route from `~/.config/pi/tool/ocr_routing.json`:

```json
{
  "model": "your-ocr-model",
  "base_url": "http://your-endpoint:8080",
  "prompt": "generic"
}
```

One file, one route: the model and the endpoint. There is no discovery, so the
same PDF always takes the same route. Point it at a gateway by naming the pool
label and the gateway URL.

Override the backend directory with `PDF_OCR_TOOL_DIR` (the extension and the
backend both honor it for discovery).

## Exit codes

The backend's exit codes surface as tool failures (pi marks a non-zero exit as
an error):

| Exit | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (no usable route, file not found, endpoint error, some pages failed) |
| 3 | `NO_OCR_SUPPORT` — the endpoint rejected the configured model for OCR |

See the repo [README](../README.md) for route details and troubleshooting.