#!/usr/bin/env bash
set -euo pipefail
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$INFERENCE_DIR/../.." && pwd)"
source "$REPO_DIR/1 Environment/environment.sh"
flybrain_environment "$REPO_DIR"
export FLYBRAIN_TEMP_DIR="${FLYBRAIN_TEMP_DIR:-$REPO_DIR/10 temp/ki67}"
if ! "$PYTHON_BIN" -B -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("ki67") is None)'; then
    echo "Install the project from its root: python -m pip install -e '.[vision]'" >&2
    exit 2
fi
exec "$PYTHON_BIN" -B -m ki67.brap_inference "$@"
