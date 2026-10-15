# PDF OCR Tool

An OCR backend for PDF files. Converts PDFs to high-quality page images, sends
each page to an OpenAI-compatible endpoint, and returns markdown. The endpoint
and the model come from a route file, so the tool never inspects what the
endpoint has loaded.

`tool/pdf_ocr_backend.py` is the tool. Host adapters live beside it and only wrap
the backend; see [Compatibility](#compatibility) for which adapter goes where.

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
uv run python tool/pdf_ocr_backend.py <pdf_path> [--page <spec>]
```

A host adapter normally invokes this for you (see [Compatibility](#compatibility)).

## Parameters

- `pdf_path`: Absolute path to a PDF file, or an `http(s)` URL to download
- `--page`: Optional page selection - `5`, `1-5`, `1,3,5`, or `1-3,5,7-9` (defaults to all pages)

Route overrides, all optional:

- `--route model@base_url`: an ad-hoc route; a bare value sets the model only
- `--model`, `--base-url`: set either half of the route
- `--prompt`: prompt family, `deepseek-ocr` or `generic`
- `--routing-config`: path to the route file
- `--mcp`: serve the `pdf_ocr` tool over stdio instead of OCRing one PDF
- `--verbose`, `-v`: debug logging

## Configuration

### The route file

One file names one route: the model and the endpoint the tool sends pages to.
By default the tool reads `ocr_routing.json` next to the backend; override the
path with the `PDF_OCR_ROUTING_CONFIG` environment variable or the
`--routing-config` argument (the argument wins).

```json
{
  "model": "your-ocr-model",
  "base_url": "http://your-endpoint:8080",
  "prompt": "generic"
}
```

- `model`: the model id to request.
- `base_url`: the endpoint root, without a `/v1` suffix. Requests go to
  `{base_url}/v1/chat/completions` with the model in the body, which is what
  every OpenAI-compatible server expects.
- `prompt` (optional): the prompt family, `deepseek-ocr` or `generic`. Without
  it, the model id decides: an id containing `deepseek-ocr` gets the
  special-token template, anything else gets the generic one.

There is no discovery and no selection. The tool does not query the endpoint for
loaded models, so the same PDF takes the same route every run. To point the tool
somewhere else, edit the file or pass an override:

```bash
# Ad-hoc route, no file edit
uv run python tool/pdf_ocr_backend.py doc.pdf --route ocr-model-pool@http://other:8080

# One half of the route
uv run python tool/pdf_ocr_backend.py doc.pdf --base-url http://other:8080
```

A missing or incomplete route file is a configuration error (exit 1) that names
the file it looked for.

### MCP server

`--mcp` turns the backend itself into a stdio MCP server exposing one tool,
`pdf_ocr(pdf_path, page=None)`. The tool calls the same code the CLI runs, in
process, so the route and prompt logic stay in one place.

```bash
# The venv must already exist (host-side `uv sync`); MCP hosts cannot install at runtime.
uv run --frozen python tool/pdf_ocr_backend.py --mcp
```

`--routing-config` also applies in this mode. Environment:

- `PDF_OCR_ROUTING_CONFIG` — path to the route file (optional; defaults to
  `ocr_routing.json` next to the backend).

### Prompt families

The route's `prompt` names a template family:

- `generic` — the default multimodal template.
- `deepseek-ocr` — the special-token template: `skip_special_tokens` off plus a
  token whitelist (`ngram_size`, `window_size`) that keeps the model's layout
  markers in the output.

Name the family explicitly when the route points at a model id the tool cannot
recognise, such as a gateway's pool label. A `deepseek-ocr` family is only
reachable through the route, not by guessing at the model id.

## Dependencies

- openai>=1.0.0
- PyMuPDF>=1.23.0
- Pillow>=10.0.0

## Technical Details

- PDF-to-image conversion at 144 DPI (high quality for OCR)
- PNG format with RGB color space
- Sequential page processing for memory management
- OCR parameters: temperature=0.0, ngram_size=30, window_size=90

## Exit Codes

| Exit Code | Meaning | Scenario |
|-----------|---------|----------|
| **0** | Success | Every page was OCR'd |
| **1** | General Error | No usable route, file not found, endpoint error, or some pages failed |
| **3** | NO_OCR_SUPPORT | The endpoint rejected the configured model for OCR |

When some pages fail, the pages that succeeded are still printed and the exit
code is 1, so a partial result is never mistaken for a complete one. The host
adapters pass those pages on with the failure, so the text is not lost.

## Troubleshooting

### Exit Code 3: NO_OCR_SUPPORT

The endpoint rejected the configured model for OCR. That usually means the route
names a model that cannot see images.

**Solution**: point the route at a model that can serve OCR:

```json
{
  "model": "your-ocr-model",
  "base_url": "http://your-endpoint:8080"
}
```

### Exit Code 1: no usable route

The route file is missing, unreadable, or does not name both a model and an
endpoint. The error message names the file the tool looked for. Create it next to
the backend, using `ocr_routing.json.example` as a starting point, or pass
`--route`, `--model` and `--base-url`.

### OCR Model Failures (Exit Code 1)

If the configured model fails:
- Verify the endpoint is running and accessible
- Check that the model is available at the endpoint
- Review endpoint logs for errors

## Compatibility

The backend is host-agnostic. This is the one place host-specific names appear:

| Host | Adapter | Install location |
|---|---|---|
| OpenCode | `tool/pdf-ocr.ts` | `~/.config/opencode/tool/` |
| MCP clients (dsh, ...) | `tool/pdf_ocr_backend.py --mcp` | anywhere; the host points at it |
| pi | `pi/` | `~/.pi/agent/extensions/pdf-ocr/` + `~/.config/pi/tool/` |

`deploy-tool.sh` deploys to either or both hosts. Each host owns its own route
file; nothing is shared at runtime.

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

Register `tool/pdf_ocr_backend.py --mcp` as a stdio server (see
[MCP server](#mcp-server)). The venv must already exist; MCP hosts cannot install
dependencies at runtime.

## Future Enhancements

- Progress reporting during OCR processing
- Batch processing of multiple PDFs
- Additional output formats (e.g., JSON, HTML)
- Image quality settings configuration