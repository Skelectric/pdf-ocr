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
# Each harness owns its own .env + ocr_routing.json; nothing is shared at runtime.
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

# Check if already installed (opencode side; pi has no pre-existing marker)
if [ -f "$TOOL_DIR/pdf-ocr.ts" ] && [ "$FORCE" = false ]; then
    echo "Tool already installed. Updating..."
    ACTION="updating"
else
    echo "Installing DeepSeek-OCR PDF Tool..."
    ACTION="installing"
fi

echo "Source: $REPO_DIR"
echo "Target: $TARGET"
echo "Action: $ACTION"

# Pull latest changes if it's a git repo
if [ -d "$REPO_DIR/.git" ]; then
    echo "Pulling latest changes..."
    cd "$REPO_DIR"
    git pull || echo "Warning: git pull failed, continuing with current state"
fi

# ---------------------------------------------------------------------------
# Shared helper: prompt for the OCR endpoint and write a .env file if
# one doesn't exist. Returns the chosen endpoint in $SHARED_ENDPOINT (so a
# --target both run can prompt once and reuse it for the pi side without a
# second prompt).
# ---------------------------------------------------------------------------
SHARED_ENDPOINT=""
ensure_env_file() {
    local env_file="$1" label="$2"
    if [ -f "$env_file" ]; then
        echo "  .env file already exists at $env_file"
        # Capture the existing endpoint for cross-seeding (best-effort).
        if [ -z "$SHARED_ENDPOINT" ]; then
            SHARED_ENDPOINT=$(grep -E '^DEEPSEEK_OCR_BASE_URL=' "$env_file" 2>/dev/null | head -1 | sed -E 's/^DEEPSEEK_OCR_BASE_URL="?([^"]*)"?.*$/\1/' || true)
        fi
        return
    fi
    echo ""
    echo "No .env found for $label. This tool requires an OCR proxy endpoint that provides:"
    echo "  - OCR model serving (DeepSeek-OCR / GLM-OCR)"
    echo "  - Model status endpoint (/running)"
    echo "  - Model capability checks (/upstream/{model}/props)"
    read -p "Enter OCR endpoint URL (e.g., http://localhost:8080): " ENDPOINT
    if [ -z "$ENDPOINT" ]; then
        echo "Warning: No endpoint provided. Using default http://localhost:8080"
        ENDPOINT="http://localhost:8080"
    fi
    echo "DEEPSEEK_OCR_BASE_URL=\"$ENDPOINT\"" > "$env_file"
    echo ".env file created at $env_file"
    SHARED_ENDPOINT="$ENDPOINT"
}

# ---------------------------------------------------------------------------
# opencode deployment
# ---------------------------------------------------------------------------
if [ "$TARGET" = "opencode" ] || [ "$TARGET" = "both" ]; then
    echo ""
    echo "Deploying opencode tool..."

    # Create tool directory if it doesn't exist
    mkdir -p "$TOOL_DIR"

    # Copy tool files
    echo "Copying tool files..."
    cp "$REPO_DIR/pdf-ocr/tool/pdf-ocr.ts" "$TOOL_DIR/"
    cp "$REPO_DIR/pdf-ocr/tool/pdf_ocr_backend.py" "$TOOL_DIR/"
    cp "$REPO_DIR/pdf-ocr/pyproject.toml" "$TOOL_DIR/"

    # Copy example routing config as reference (never overwrite user config)
    if [ ! -f "$TOOL_DIR/ocr_routing.json.example" ]; then
        cp "$REPO_DIR/pdf-ocr/tool/ocr_routing.json.example" "$TOOL_DIR/"
    fi

    # Make Python script executable
    chmod +x "$TOOL_DIR/pdf_ocr_backend.py"

    # .env: prompt if absent (sets $SHARED_ENDPOINT for the pi cross-seed).
    ENV_FILE="$TOOL_DIR/.env"
    ensure_env_file "$ENV_FILE" "opencode"

    # During updates, remind users about the example config.
    if [ -f "$TOOL_DIR/ocr_routing.json.example" ]; then
        echo "Reference config with model-set routing examples: $TOOL_DIR/ocr_routing.json.example"
    fi

    # Create OCR routing configuration if missing.
    ROUTING_CONFIG="$TOOL_DIR/ocr_routing.json"
    if [ ! -f "$ROUTING_CONFIG" ]; then
        echo ""
        echo "Creating OCR routing configuration..."
        echo "This file maps loaded model sets to their preferred OCR method."
        echo "Single-model keys work as before. Comma-separated keys match ALL loaded models."

        cat > "$ROUTING_CONFIG" << 'EOF'
{
  "_comment": "OCR Routing Configuration - Maps loaded model sets to preferred OCR method. Single-model keys work as before. Comma-separated keys match ALL loaded models (order-independent).",
  "_routing_options": {
    "deepseek-ocr": "Use DeepSeek-OCR model (requires sufficient VRAM)",
    "deepseek-ocr-2": "Use DeepSeek-OCR-2 model (requires sufficient VRAM)",
    "glm-ocr": "Use GLM-OCR model on GPU (lighter VRAM, for constrained setups)",
    "glm-ocr-cpu": "Use GLM-OCR model on CPU (zero VRAM, uses ~3 GiB RAM)",
    "current_model": "Use the currently loaded model (requires vision support)"
  },
  "ocr_routing": {
    "_comment": "Model-set examples: comma-separated keys match ALL loaded models",
    "kimi-k2.6,qwen3.6-35b-a3b-nvfp4": "deepseek-ocr-2",
    "kimi-k2.6,qwen3.6-27b-nvfp4,qwen3.5-4b": "glm-ocr",
    "kimi-k2.6,qwen3.6-27b-nvfp4": "glm-ocr",
    "kimi-k2.6": "current_model",
    "kimi-k2.7-code,qwen3.6-27b-nvfp4": "glm-ocr-cpu",
    "kimi-k2.7-code": "glm-ocr-cpu",
    "qwen3.6-27b-mtp": "glm-ocr-cpu",
    "qwen3.6-27b-mtp-cpu": "glm-ocr-cpu"
  },
  "default": "current_model"
}
EOF

        echo "Created default routing config at $ROUTING_CONFIG"
        echo "Edit this file to enable DeepSeek-OCR for specific models."
    fi

    # Install/update Python dependencies
    echo "Installing opencode Python dependencies..."
    cd "$TOOL_DIR"
    uv sync

    # Verify installation
    echo "Verifying opencode installation..."
    if [ -f "$TOOL_DIR/pdf-ocr.ts" ] && [ -f "$TOOL_DIR/pdf_ocr_backend.py" ] && [ -f "$TOOL_DIR/pyproject.toml" ]; then
        echo "All opencode tool files installed successfully"
    else
        echo "Warning: Some opencode tool files may be missing"
    fi
fi

# ---------------------------------------------------------------------------
# pi deployment (independent of opencode; deploys a private backend copy).
# Backend  -> ~/.config/pi/tool/      (parallel to ~/.config/opencode/tool/)
# Extension -> ~/.pi/agent/extensions/pdf-ocr/  (pi global auto-discovery)
# Each harness owns its own .env + ocr_routing.json; nothing is shared at
# runtime. Override the backend dir with $PDF_OCR_TOOL_DIR.
# ---------------------------------------------------------------------------
if [ "$TARGET" = "pi" ] || [ "$TARGET" = "both" ]; then
    PI_TOOL_DIR="$HOME/.config/pi/tool"
    PI_EXT_DIR="$HOME/.pi/agent/extensions/pdf-ocr"

    echo ""
    echo "Deploying pi extension..."
    mkdir -p "$PI_TOOL_DIR" "$PI_EXT_DIR"

    # Backend files (same source as opencode; private deployed copy).
    echo "Copying pi backend files..."
    cp "$REPO_DIR/pdf-ocr/tool/pdf_ocr_backend.py" "$PI_TOOL_DIR/"
    cp "$REPO_DIR/pdf-ocr/pyproject.toml" "$PI_TOOL_DIR/"
    chmod +x "$PI_TOOL_DIR/pdf_ocr_backend.py"

    # Extension files.
    echo "Copying pi extension files..."
    cp "$REPO_DIR/pdf-ocr/pi/index.ts" "$PI_EXT_DIR/"
    cp "$REPO_DIR/pdf-ocr/pi/package.json" "$PI_EXT_DIR/"
    cp "$REPO_DIR/pdf-ocr/pi/README.md" "$PI_EXT_DIR/"

    # Reference routing config (never overwrites user config).
    if [ ! -f "$PI_TOOL_DIR/ocr_routing.json.example" ]; then
        cp "$REPO_DIR/pdf-ocr/tool/ocr_routing.json.example" "$PI_TOOL_DIR/"
    fi

    # pi .env: when deploying both, reuse the endpoint already captured in
    # $SHARED_ENDPOINT (from the opencode prompt) so the user is prompted once.
    # When deploying pi alone, prompt for pi's own .env. pi stays decoupled at
    # runtime — it reads its own .env, never opencode's.
    PI_ENV_FILE="$PI_TOOL_DIR/.env"
    if [ -f "$PI_ENV_FILE" ]; then
        echo "  .env file already exists at $PI_ENV_FILE"
    elif [ -n "$SHARED_ENDPOINT" ]; then
        echo "DEEPSEEK_OCR_BASE_URL=\"$SHARED_ENDPOINT\"" > "$PI_ENV_FILE"
        echo "  created $PI_ENV_FILE (endpoint: $SHARED_ENDPOINT)"
    else
        ensure_env_file "$PI_ENV_FILE" "pi"
    fi

    # pi routing config: seed from opencode's if present (convenience on the
    # same host), else a minimal default. User edits this file afterward.
    PI_ROUTING_CONFIG="$PI_TOOL_DIR/ocr_routing.json"
    if [ ! -f "$PI_ROUTING_CONFIG" ]; then
        if [ -f "$TOOL_DIR/ocr_routing.json" ]; then
            cp "$TOOL_DIR/ocr_routing.json" "$PI_ROUTING_CONFIG"
            echo "  seeded ocr_routing.json from opencode's config"
        else
            cat > "$PI_ROUTING_CONFIG" << 'EOF'
{
  "_comment": "OCR Routing Configuration - Maps loaded model sets to preferred OCR method. Comma-separated keys match ALL loaded models (order-independent).",
  "ocr_routing": {},
  "default": "current_model"
}
EOF
            echo "  created default ocr_routing.json (edit to enable dedicated OCR models)"
        fi
    fi

    # Install/update Python dependencies for the pi backend.
    echo "Installing pi Python dependencies..."
    cd "$PI_TOOL_DIR"
    uv sync

    # Verify pi installation.
    echo "Verifying pi installation..."
    if [ -f "$PI_EXT_DIR/index.ts" ] && [ -f "$PI_EXT_DIR/package.json" ] && [ -f "$PI_TOOL_DIR/pdf_ocr_backend.py" ] && [ -f "$PI_TOOL_DIR/pyproject.toml" ]; then
        echo "All pi files installed successfully"
    else
        echo "Warning: Some pi files may be missing"
    fi
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
