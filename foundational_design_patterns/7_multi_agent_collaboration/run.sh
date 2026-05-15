#!/bin/bash
set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

LOCAL_ENV_FILE=".env"
ROOT_ENV_FILE="../../.env"

load_env_file() {
    local env_path="$1"
    if [[ -f "$env_path" ]]; then
        echo -e "${BLUE}>>> Loading environment from ${env_path} <<<${NC}"
        set -a
        # shellcheck disable=SC1090
        source "$env_path"
        set +a
        return 0
    fi
    return 1
}

load_envs() {
    load_env_file "$LOCAL_ENV_FILE" || true
    load_env_file "$ROOT_ENV_FILE" || true

    if [[ -n "${AGENT_DISABLE_SSL:-}" && -z "${AGENTIC_DISABLE_SSL:-}" ]]; then
        echo -e "${YELLOW}Warning: AGENT_DISABLE_SSL is set, but the correct variable is AGENTIC_DISABLE_SSL.${NC}"
        echo -e "${YELLOW}         Use: AGENTIC_DISABLE_SSL=1 bash run.sh${NC}"
    fi
}

require_tavily_key() {
    if [[ -z "${TAVILY_API_KEY:-}" ]]; then
        echo -e "${YELLOW}Note: TAVILY_API_KEY not set. Legacy research_report_agent may fail.${NC}"
    fi
}

run() {
    echo -e "\n${BLUE}>>> Running: $1 <<<${NC}"
    SSL_CERT_FILE="${SSL_CERT_FILE:-}" \
    REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}" \
    CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}" \
    AGENTIC_DISABLE_SSL="${AGENTIC_DISABLE_SSL:-}" \
    uv run python "$1"
    echo -e "${GREEN}>>> Completed: $1 <<<${NC}"
}

load_envs
require_tavily_key

echo "Select an example to run:"
echo "1) Orchestrator-Worker topology (recommended)"
echo "2) Peer/Swarm topology"
echo "3) Legacy monolithic supervisor"
read -r -p "Enter your choice (1-3): " choice

case "$choice" in
    1)
        run "src/orchestrator_worker.py"
        ;;
    2)
        run "src/peer_swarm.py"
        ;;
    3)
        run "src/research_report_agent.py"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
