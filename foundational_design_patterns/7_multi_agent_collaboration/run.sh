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
    # Priority: existing environment > local .env > repo-root .env
    load_env_file "$LOCAL_ENV_FILE" || true
    load_env_file "$ROOT_ENV_FILE" || true

    if [[ -n "${AGENT_DISABLE_SSL:-}" && -z "${AGENTIC_DISABLE_SSL:-}" ]]; then
        echo -e "${YELLOW}Warning: AGENT_DISABLE_SSL is set, but the correct variable is AGENTIC_DISABLE_SSL.${NC}"
        echo -e "${YELLOW}         Use: AGENTIC_DISABLE_SSL=1 bash run.sh${NC}"
    fi
}

require_tavily_key() {
    if [[ -z "${TAVILY_API_KEY:-}" ]]; then
        echo -e "${RED}ERROR: TAVILY_API_KEY is not set.${NC}"
        echo
        echo "Set it in one of these places:"
        echo "  1) Current shell environment"
        echo "  2) $LOCAL_ENV_FILE"
        echo "  3) $ROOT_ENV_FILE"
        echo
        echo "Example:"
        echo '  export TAVILY_API_KEY="your_key_here"'
        echo
        exit 1
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

# --- Pre-flight checks ---
load_envs
require_tavily_key

# --- Run agents ---
run "src/research_report_agent.py"