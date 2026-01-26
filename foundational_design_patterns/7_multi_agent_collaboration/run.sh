#!/bin/bash
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

ENV_FILE=".env"

load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        echo -e "${BLUE}>>> Loading environment from .env <<<${NC}"
        set -a
        source "$ENV_FILE"
        set +a
    else
        echo -e "${RED}ERROR: .env file not found.${NC}"
        echo
        echo "Create a .env file in this directory with:"
        echo
        echo '  TAVILY_API_KEY="your_key_here"'
        echo
        exit 1
    fi
}

require_tavily_key() {
    if [[ -z "${TAVILY_API_KEY:-}" ]]; then
        echo -e "${RED}ERROR: TAVILY_API_KEY is not set in .env.${NC}"
        echo
        echo "Open .env and add:"
        echo
        echo '  TAVILY_API_KEY="your_key_here"'
        echo
        exit 1
    fi
}

run() {
    echo -e "\n${BLUE}>>> Running: $1 <<<${NC}"
    uv run python "$1"
    echo -e "${GREEN}>>> Completed: $1 <<<${NC}"
}

# --- Pre-flight checks ---
load_env
require_tavily_key

# --- Run agents ---
run "src/research_report_agent.py"