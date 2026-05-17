#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WS_DIR="$ROOT_DIR/typescript_base"

if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$ROOT_DIR/.env"
    set +a
fi

if [ ! -d "$WS_DIR/node_modules" ]; then
    echo -e "${BLUE}>>> First run: bootstrapping bun workspace <<<${NC}"
    (cd "$WS_DIR" && bun install)
fi

cd "$SCRIPT_DIR"

echo "========================================="
echo "  Human-in-the-Loop (HITL) Pattern Examples"
echo "========================================="
echo ""

if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic HITL with Console Approval"
echo "2) Advanced HITL with Risk-Based Checkpoints"
echo "3) LangGraph HITL Integration"
echo "4) Run All Examples"
echo ""
read -r -p "Enter your choice (1-4): " choice

run_example() {
    local label="$1"
    local file="$2"
    echo ""
    echo "$label"
    echo "----------------------------------------"
    bun run "$file"
}

case "$choice" in
    1)
        run_example "Running Basic HITL with Console Approval..." src/hitl_basic.ts
        ;;
    2)
        run_example "Running Advanced HITL with Risk-Based Checkpoints..." src/hitl_advanced.ts
        ;;
    3)
        run_example "Running LangGraph HITL Integration..." src/hitl_langgraph.ts
        ;;
    4)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        run_example "1. Basic HITL with Console Approval" src/hitl_basic.ts
        echo ""
        echo "========================================="
        echo ""
        run_example "2. Advanced HITL with Risk-Based Checkpoints" src/hitl_advanced.ts
        echo ""
        echo "========================================="
        echo ""
        run_example "3. LangGraph HITL Integration" src/hitl_langgraph.ts
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "  Examples completed!"
echo "========================================="
