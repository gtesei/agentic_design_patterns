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
echo "  ReAct Pattern Examples"
echo "========================================="
echo ""

if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic ReAct Agent (Research Assistant)"
echo "2) Advanced ReAct with Explicit Reasoning Traces"
echo "3) Run All Examples"
echo ""
read -r -p "Enter your choice (1-3): " choice

case "$choice" in
    1)
        echo ""
        echo "Running Basic ReAct Agent..."
        echo "----------------------------------------"
        bun run src/react_agent.ts
        ;;
    2)
        echo ""
        echo "Running Advanced ReAct Agent..."
        echo "----------------------------------------"
        bun run src/react_agent_advanced.ts
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic ReAct Agent"
        echo "----------------------------------------"
        bun run src/react_agent.ts
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced ReAct Agent"
        echo "----------------------------------------"
        bun run src/react_agent_advanced.ts
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
