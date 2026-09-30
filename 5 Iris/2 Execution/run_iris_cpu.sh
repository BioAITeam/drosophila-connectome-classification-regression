#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
source "$REPO_DIR/1 Environment/environment.sh"
flybrain_environment "$REPO_DIR"
export FLYBRAIN_THREADS="${FLYBRAIN_THREADS:-1}"
TASK_WORKERS="${FLYBRAIN_WORKERS:-18}"
RETRAINED_BASE="$(realpath -m -- "$REPO_DIR/5 Iris/6 Workspace/retrained")"
CAMPAIGN_ROOT="${FLYBRAIN_CAMPAIGN_ROOT:-$RETRAINED_BASE/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[[ "$CAMPAIGN_ROOT" == /* ]] || CAMPAIGN_ROOT="$REPO_DIR/$CAMPAIGN_ROOT"
CAMPAIGN_ROOT="$(realpath -m -- "$CAMPAIGN_ROOT")"
if [[ "$CAMPAIGN_ROOT" != "$RETRAINED_BASE/"* ]]; then
    echo "FLYBRAIN_CAMPAIGN_ROOT must name a directory under 5 Iris/6 Workspace/retrained." >&2
    exit 2
fi
"$PYTHON_BIN" -B - "$CAMPAIGN_ROOT" <<'PYTHON'
import sys
from flybrain.paths import allowed_derived_write
allowed_derived_write(sys.argv[1]).mkdir(parents=True, exist_ok=True)
PYTHON
echo "Iris retraining destination: $CAMPAIGN_ROOT"
for TASK_CAMPAIGN in core controls lif local transfer null; do
  ./run.sh --device cpu campaign \
    --plan "5 Iris/1 Configuration/plan_${TASK_CAMPAIGN}.json" \
    --output "$CAMPAIGN_ROOT/round4-${TASK_CAMPAIGN}" \
    --workers "$TASK_WORKERS" \
    >> "$CAMPAIGN_ROOT/round4-${TASK_CAMPAIGN}.log" 2>&1
done
