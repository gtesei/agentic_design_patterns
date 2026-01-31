#!/bin/bash
set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Build SSL environment variables if they exist
SSL_ENV_VARS=""
if [ -n "${SSL_CERT_FILE:-}" ]; then
    SSL_ENV_VARS="$SSL_ENV_VARS SSL_CERT_FILE=$SSL_CERT_FILE"
fi
if [ -n "${REQUESTS_CA_BUNDLE:-}" ]; then
    SSL_ENV_VARS="$SSL_ENV_VARS REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE"
fi
if [ -n "${CURL_CA_BUNDLE:-}" ]; then
    SSL_ENV_VARS="$SSL_ENV_VARS CURL_CA_BUNDLE=$CURL_CA_BUNDLE"
fi

run() {
    echo -e "\n${BLUE}>>> $1 <<<${NC}"
    if [ -n "$SSL_ENV_VARS" ]; then
        env $SSL_ENV_VARS uv run python "$1"
    else
        uv run python "$1"
    fi
    echo -e "${GREEN}>>> Completed <<<${NC}"
}

run "src/chain_prompt.py"