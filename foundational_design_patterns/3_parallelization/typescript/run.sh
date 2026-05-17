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

SSL_ENV_VARS=""
if [ -n "${NODE_EXTRA_CA_CERTS:-}" ]; then
    SSL_ENV_VARS="$SSL_ENV_VARS NODE_EXTRA_CA_CERTS=$NODE_EXTRA_CA_CERTS"
fi
if [ -n "${SSL_CERT_FILE:-}" ]; then
    SSL_ENV_VARS="$SSL_ENV_VARS SSL_CERT_FILE=$SSL_CERT_FILE"
fi

run() {
    echo -e "\n${BLUE}>>> $1 <<<${NC}"
    if [ -n "$SSL_ENV_VARS" ]; then
        env $SSL_ENV_VARS bun run "$1"
    else
        bun run "$1"
    fi
    echo -e "${GREEN}>>> Completed <<<${NC}"
}

cd "$SCRIPT_DIR"
run "src/parallelization.ts"
