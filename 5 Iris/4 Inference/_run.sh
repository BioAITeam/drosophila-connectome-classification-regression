#!/usr/bin/env bash
set -euo pipefail
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$INFERENCE_DIR/../.." && pwd)"
source "$REPO_DIR/1 Environment/environment.sh"
flybrain_environment "$REPO_DIR"
export FLYBRAIN_TEMP_DIR="${FLYBRAIN_TEMP_DIR:-$REPO_DIR/10 temp/iris}"
exec "$PYTHON_BIN" -B -m flybrain.iris_inference "$@"
