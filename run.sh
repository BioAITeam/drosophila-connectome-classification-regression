#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_DIR/1 Environment/environment.sh"
flybrain_environment "$REPO_DIR"
exec "$PYTHON_BIN" -B "$REPO_DIR/1 Environment/launch.py" "$@"
