#!/bin/bash
# Auto-trade lifecycle manager
# Usage: ./scripts/run_auto_trade.sh [check|start|stop|status]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

ACTION="${1:-check}"

echo "=== Auto-Trade: $ACTION ==="
python scripts/auto_trade.py "$ACTION"
