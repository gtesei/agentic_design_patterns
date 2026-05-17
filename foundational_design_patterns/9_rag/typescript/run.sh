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
echo "  RAG Pattern Examples"
echo "========================================="
echo ""

if [ ! -f "$ROOT_DIR/.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic Hybrid RAG (lexical + dense + RRF + rerank)"
echo "2) Advanced RAG (hosted file_search + local fallback)"
echo "3) Agentic RAG loop (rewrite/retrieve/grade/fallback/self-check)"
echo "4) RAG evaluation and failure modes"
echo "5) Run All Examples"
echo ""
read -r -p "Enter your choice (1-5): " choice

run_example() {
    local file="$1"
    echo "----------------------------------------"
    bun run "$file"
    echo ""
}

case "$choice" in
    1)
        echo "\nRunning Basic Hybrid RAG..."
        run_example src/rag_basic.ts
        ;;
    2)
        echo "\nRunning Advanced RAG..."
        run_example src/rag_advanced.ts
        ;;
    3)
        echo "\nRunning Agentic RAG..."
        run_example src/rag_agentic.ts
        ;;
    4)
        echo "\nRunning RAG Eval + Failure Modes..."
        run_example src/rag_eval_and_failure_modes.ts
        ;;
    5)
        echo "\nRunning All Examples..."
        run_example src/rag_basic.ts
        run_example src/rag_advanced.ts
        run_example src/rag_agentic.ts
        run_example src/rag_eval_and_failure_modes.ts
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "========================================="
echo "  Examples completed!"
echo "========================================="
