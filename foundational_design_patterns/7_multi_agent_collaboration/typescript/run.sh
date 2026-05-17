#!/bin/bash
set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

if [ -z "${TAVILY_API_KEY:-}" ]; then
    echo -e "${YELLOW}Note: TAVILY_API_KEY not set. Legacy research_report_agent may fail.${NC}"
fi

cd "$SCRIPT_DIR"

echo "Select an example to run:"
echo "1) Orchestrator-Worker topology (recommended)"
echo "2) Peer/Swarm topology"
echo "3) Legacy monolithic supervisor"
read -r -p "Enter your choice (1-3): " choice

run_example() {
    local file="$1"
    echo -e "\n${BLUE}>>> Running: $file <<<${NC}"
    bun run "$file"
    echo -e "${GREEN}>>> Completed: $file <<<${NC}"
}

case "$choice" in
    1)
        run_example src/orchestrator_worker.ts
        ;;
    2)
        run_example src/peer_swarm.ts
        ;;
    3)
        run_example src/research_report_agent.ts
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
