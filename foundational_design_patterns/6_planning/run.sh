#!/bin/bash
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

run() {
    echo -e "\n${BLUE}>>> $1 <<<${NC}"
    uv run python "$1"
    echo -e "${GREEN}>>> Completed <<<${NC}"
}

run "src/planning_cust_agent.py"