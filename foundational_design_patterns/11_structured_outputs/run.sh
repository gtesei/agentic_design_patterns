#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

uv run python src/structured_outputs_basic.py
uv run python src/structured_outputs_advanced.py
