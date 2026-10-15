#!/usr/bin/env bash
set -euo pipefail

# Installation/Update script for the PDF OCR tool (deploys to opencode and/or pi).
# Usage: ./deploy-tool.sh [options]
# Options:
#   --target {opencode,pi,both}  Which harness to deploy to (default: both)
#   --force    Force reinstallation even if already installed
#   --repo <path>  Specify custom repository path (default: ~/pdf-ocr)
#
# Deploys:
#   - opencode tool  -> ~/.config/opencode/tool/      (pdf-ocr.ts + backend)
#   - pi extension   -> ~/.pi/agent/extensions/pdf-ocr/ + ~/.config/pi/tool/ (backend)
# Each harness owns its own ocr_routing.json (one route); nothing is shared at
# runtime.
# Single-harness users: use --target opencode or --target pi to avoid deploying
# the other harness's files and running a second uv sync.

REPO_DIR="."
TOOL_DIR="$HOME/.config/opencode/tool"
FORCE=false
TARGET="both"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --repo)
            REPO_DIR="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            case "$TARGET" in
                opencode|pi|both) ;;
                *)
                    echo "Error: --target must be one of: opencode, pi, both (got: $TARGET)"
                    echo "Usage: $0 [--target {opencode,pi,both}] [--force] [--repo <path>]"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--target {opencode,pi,both}] [--force] [--repo <path>]"
            exit 1
            ;;
    esac
done

# Resolve REPO_DIR to an absolute path so subsequent `cd` calls (git pull,
# uv sync) don't break relative source lookups later in the script.
REPO_DIR="$(cd "$REPO_DIR" && pwd)"

# Check if source exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Error: Repository not found at $REPO_DIR"
    echo "Please clone the repository first or provide a valid path with --repo"
    exit 1
fi

# Check if already installed, per target (the pi extension is pi's marker).
case "$TARGET" in
    opencode) INSTALLED_MARKER="$TOOL_DIR/pdf-ocr.ts" ;;
    pi) INSTALLED_MARKER="$HOME/.pi/agent/extensions/pdf-ocr/index.ts" ;;
    both) INSTALLED_MARKER="$TOOL_DIR/pdf-ocr.ts" ;;
esac
if [ -f "$INSTALLED_MARKER" ] && [ "$FORCE" = false ]; then
    echo "Tool already installed. Updating..."
    ACTION="updating"
else
    echo "Installing PDF OCR Tool..."
    ACTION="installing"
fi

echo "Source: $REPO_DIR"
echo "Target: $TARGET"
echo "Action: $ACTION"

# Pull latest changes if it's a git repo (skip a dirty tree rather than merge).
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    if [ -n "$(git status --porcelain)" ]; then
        echo "Working tree has local changes; skipping git pull"
    else
        echo "Pulling latest changes..."
        git pull || echo "Warning: git pull failed, continuing with current state"
    fi
fi

# ---------------------------------------------------------------------------
# Shared helpers: write the route file, prompt for it when absent, and deploy
# one harness. The chosen values land in $SHARED_MODEL / $SHARED_ENDPOINT so a
# --target both run prompts once and reuses them for the second harness.
# ---------------------------------------------------------------------------
SHARED_MODEL=""
SHARED_ENDPOINT=""

write_route_file() {
    local route_file="$1" model="$2" endpoint="$3"
    cat > "$route_file" << EOF
{
  "model": "$model",
  "base_url": "$endpoint",
  "prompt": "generic"
}
EOF
}

ensure_route_file() {
    local route_file="$1" label="$2"
    if [ -f "$route_file" ]; then
        echo "  route file already exists at $route_file"
        return
    fi
    echo ""
    echo "No route file found for $label. It names one route:"
    echo "  - the OCR model id"
    echo "  - the OCR endpoint root URL (any OpenAI-compatible service)"
    read -p "Enter the OCR model id: " MODEL
    read -p "Enter the OCR endpoint URL (e.g., http://localhost:8080): " ENDPOINT
    MODEL="${MODEL:-your-ocr-model}"
    ENDPOINT="${ENDPOINT:-http://localhost:8080}"
    write_route_file "$route_file" "$MODEL" "$ENDPOINT"
    echo "Route file created at $route_file"
    if [ "$MODEL" = "your-ocr-model" ]; then
        echo "  The model id is a placeholder. Replace it in $route_file with a model your endpoint serves."
    fi
    SHARED_MODEL="$MODEL"
    SHARED_ENDPOINT="$ENDPOINT"
}

# Deploy one harness: copy <source> <destination> pairs, ship the example route
# file, settle the route file, install the venv, and verify the copies landed.
# route_mode is "own" (prompt for this harness's route) or "seed" (reuse
# opencode's route file when it exists).
deploy_target() {
    local label="$1" tool_dir="$2" route_mode="$3"
    shift 3

    echo ""
    echo "Deploying $label..."
    mkdir -p "$tool_dir"

    echo "Copying $label files..."
    local -a deployed=()
    while [ $# -gt 0 ]; do
        local src="$1" dest="$2"
        shift 2
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        deployed+=("$dest")
    done
    chmod +x "$tool_dir/pdf_ocr_backend.py"

    # Ship the example route file as a reference; never overwrite a user's file.
    if [ ! -f "$tool_dir/ocr_routing.json.example" ]; then
        cp "$REPO_DIR/tool/ocr_routing.json.example" "$tool_dir/"
    fi
    echo "Reference route file: $tool_dir/ocr_routing.json.example"

    # The route file: one route, with its own endpoint.
    local route_file="$tool_dir/ocr_routing.json"
    if [ -f "$route_file" ]; then
        echo "  route file already exists at $route_file"
    elif [ "$route_mode" = "seed" ] && [ -f "$TOOL_DIR/ocr_routing.json" ]; then
        cp "$TOOL_DIR/ocr_routing.json" "$route_file"
        echo "  seeded ocr_routing.json from opencode's route file"
    elif [ -n "$SHARED_ENDPOINT" ]; then
        write_route_file "$route_file" "$SHARED_MODEL" "$SHARED_ENDPOINT"
        echo "  created ocr_routing.json from the route just entered"
    else
        ensure_route_file "$route_file" "$label"
    fi

    echo "Installing $label Python dependencies..."
    cd "$tool_dir"
    uv sync --no-dev

    echo "Verifying $label installation..."
    local missing=""
    local f
    for f in "${deployed[@]}"; do
        [ -f "$f" ] || missing="$missing $f"
    done
    if [ -z "$missing" ]; then
        echo "All $label files installed successfully"
    else
        echo "Warning: missing files:$missing"
    fi
}

# ---------------------------------------------------------------------------
# opencode deployment
# ---------------------------------------------------------------------------
if [ "$TARGET" = "opencode" ] || [ "$TARGET" = "both" ]; then
    deploy_target "opencode" "$TOOL_DIR" own \
        "$REPO_DIR/tool/pdf-ocr.ts" "$TOOL_DIR/pdf-ocr.ts" \
        "$REPO_DIR/tool/pdf_ocr_backend.py" "$TOOL_DIR/pdf_ocr_backend.py" \
        "$REPO_DIR/pyproject.toml" "$TOOL_DIR/pyproject.toml"
fi

# ---------------------------------------------------------------------------
# pi deployment (independent of opencode; deploys a private backend copy).
# Backend  -> ~/.config/pi/tool/      (parallel to ~/.config/opencode/tool/)
# Extension -> ~/.pi/agent/extensions/pdf-ocr/  (pi global auto-discovery)
# Each harness owns its own ocr_routing.json (one route); nothing is shared at
# runtime. Override the backend dir with $PDF_OCR_TOOL_DIR.
# ---------------------------------------------------------------------------
if [ "$TARGET" = "pi" ] || [ "$TARGET" = "both" ]; then
    PI_TOOL_DIR="$HOME/.config/pi/tool"
    PI_EXT_DIR="$HOME/.pi/agent/extensions/pdf-ocr"
    deploy_target "pi" "$PI_TOOL_DIR" seed \
        "$REPO_DIR/tool/pdf_ocr_backend.py" "$PI_TOOL_DIR/pdf_ocr_backend.py" \
        "$REPO_DIR/pyproject.toml" "$PI_TOOL_DIR/pyproject.toml" \
        "$REPO_DIR/pi/index.ts" "$PI_EXT_DIR/index.ts" \
        "$REPO_DIR/pi/package.json" "$PI_EXT_DIR/package.json" \
        "$REPO_DIR/pi/README.md" "$PI_EXT_DIR/README.md"
fi

echo ""
if [ "$ACTION" = "updating" ]; then
    echo "Update complete!"
    if [ "$TARGET" = "opencode" ] || [ "$TARGET" = "both" ]; then
        echo "  opencode tool: $TOOL_DIR (restart opencode to apply)"
    fi
    if [ "$TARGET" = "pi" ] || [ "$TARGET" = "both" ]; then
        echo "  pi extension:  $HOME/.pi/agent/extensions/pdf-ocr (run /reload in pi to load the pdf_ocr tool)"
    fi
else
    echo "Installation complete!"
    if [ "$TARGET" = "opencode" ] || [ "$TARGET" = "both" ]; then
        echo "  opencode tool: $TOOL_DIR"
    fi
    if [ "$TARGET" = "pi" ] || [ "$TARGET" = "both" ]; then
        echo "  pi extension:  $HOME/.pi/agent/extensions/pdf-ocr (run /reload in pi to load the pdf_ocr tool)"
    fi
    echo "To update later, run: ./deploy-tool.sh"
fi