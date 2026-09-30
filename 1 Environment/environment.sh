#!/usr/bin/env bash

flybrain_environment() {
    local repository="$1"
    if [[ -n "${FLYBRAIN_PYTHON:-}" ]]; then
        PYTHON_BIN="$FLYBRAIN_PYTHON"
    elif [[ -n "${FLYBRAIN_ENV:-}" ]]; then
        PYTHON_BIN="$FLYBRAIN_ENV/bin/python"
    else
        PYTHON_BIN="$(command -v python || command -v python3 || true)"
    fi
    if [[ "$PYTHON_BIN" != */* ]]; then
        PYTHON_BIN="$(command -v "$PYTHON_BIN" || true)"
    fi
    if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
        echo "Python was not found. Activate an environment or set FLYBRAIN_PYTHON or FLYBRAIN_ENV." >&2
        return 2
    fi
    PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd)/$(basename -- "$PYTHON_BIN")"

    export FLYBRAIN_REPO="$repository"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$repository/3 Connectome${PYTHONPATH:+:$PYTHONPATH}"
    export FLYBRAIN_GPU_LOCK_DIR="${FLYBRAIN_GPU_LOCK_DIR:-$repository/10 temp/gpu-locks}"
    local environment_root library_dir
    environment_root="$(cd -- "$(dirname -- "$PYTHON_BIN")/.." && pwd)"
    for library_dir in "$environment_root"/lib/python*/site-packages/nvidia/nvjitlink/lib; do
        if [[ -d "$library_dir" ]]; then
            export LD_LIBRARY_PATH="$library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            break
        fi
    done
}
