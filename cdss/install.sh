#!/usr/bin/env bash
# install.sh — Setup script for the breast-cancer prediction environment
# Works around the pip certifi path-with-spaces bug on Linux.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SYSTEM_CERT="/etc/ssl/certs/ca-certificates.crt"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Breast Cancer Prediction — Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Detect Python 3.12 (most mature ML wheel support) or Python 3.13
if command -v python3.12 &>/dev/null; then
    PYTHON=python3.12
elif command -v python3.13 &>/dev/null; then
    PYTHON=python3.13
else
    echo "ERROR: Python 3.12 or 3.13 required (found: $(python3 --version 2>&1))"
    exit 1
fi
echo "  Using: $PYTHON ($($PYTHON --version))"

# 2. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# 3. Fix certifi CA bundle (path-with-spaces workaround)
CERTIFI_PEM="$VENV_DIR/lib/$($PYTHON -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pip/_vendor/certifi/cacert.pem"
if [ -f "$SYSTEM_CERT" ] && [ -f "$CERTIFI_PEM" ]; then
    cp "$SYSTEM_CERT" "$CERTIFI_PEM"
    echo "  Patched certifi with system CA bundle."
fi

# 4. Install dependencies with explicit cert paths
export SSL_CERT_FILE="$SYSTEM_CERT"
export REQUESTS_CA_BUNDLE="$SYSTEM_CERT"

if command -v uv &>/dev/null; then
    echo "  Installing requirements using uv (fast mode)..."
    uv pip install --python "$VENV_DIR/bin/python" -r "$SCRIPT_DIR/requirements.txt"
else
    echo "  Upgrading pip..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    echo "  Installing requirements..."
    "$VENV_DIR/bin/pip" install --timeout 120 -r "$SCRIPT_DIR/requirements.txt"
fi

# 5. Also patch the installed certifi package (not just pip's vendor copy)
INSTALLED_CERTIFI="$VENV_DIR/lib/$($PYTHON -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/certifi/cacert.pem"
if [ -f "$SYSTEM_CERT" ] && [ -f "$INSTALLED_CERTIFI" ]; then
    cp "$SYSTEM_CERT" "$INSTALLED_CERTIFI"
    echo "  Patched installed certifi with system CA bundle."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  Setup complete!"
echo ""
echo "  Activate:  source .venv/bin/activate"
echo "  Train:     SSL_CERT_FILE=$SYSTEM_CERT .venv/bin/python train_pipeline.py"
echo "  Dashboard: .venv/bin/streamlit run dashboard/app.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
