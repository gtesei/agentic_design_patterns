#!/bin/bash

# Prioritization Pattern Examples Runner
# This script runs the different Prioritization pattern examples

set -e  # Exit on error

echo "========================================="
echo "  Prioritization Pattern Examples"
echo "========================================="
echo ""

if [ -n "${AGENT_DISABLE_SSL:-}" ] && [ -z "${AGENTIC_DISABLE_SSL:-}" ]; then
    echo "Warning: AGENT_DISABLE_SSL is set, but the supported variable is AGENTIC_DISABLE_SSL."
    echo "         Use: AGENTIC_DISABLE_SSL=1 bash run.sh"
    echo ""
fi

if [ -z "${AGENTIC_DISABLE_SSL:-}" ]; then
    echo "Note: if you are on a corporate SSL-inspecting network and see"
    echo "certificate verification failures, rerun with:"
    echo "  AGENTIC_DISABLE_SSL=1 bash run.sh"
    echo ""
fi

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

is_ssl_error() {
    local log_file="$1"
    grep -Eqi "CERTIFICATE_VERIFY_FAILED|SSL: CERTIFICATE_VERIFY_FAILED|httpx\.ConnectError|openai\.APIConnectionError" "$log_file"
}

run_example() {
    local script_path="$1"
    local tmp_log
    local retry_log

    tmp_log="$(mktemp)"

    set +e
    SSL_CERT_FILE="${SSL_CERT_FILE:-}" \
    REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}" \
    CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}" \
    AGENTIC_DISABLE_SSL="${AGENTIC_DISABLE_SSL:-}" \
    uv run python "$script_path" < <(yes "") 2>&1 | tee "$tmp_log"
    local rc=${PIPESTATUS[0]}
    set -e

    if [ "$rc" -eq 0 ]; then
        rm -f "$tmp_log"
        return 0
    fi

    if is_ssl_error "$tmp_log" && [ -z "${AGENTIC_DISABLE_SSL:-}" ]; then
        echo ""
        echo "Detected SSL certificate verification failure."
        echo "Retrying once with AGENTIC_DISABLE_SSL=1 ..."

        retry_log="$(mktemp)"
        set +e
        SSL_CERT_FILE="${SSL_CERT_FILE:-}" \
        REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}" \
        CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}" \
        AGENTIC_DISABLE_SSL=1 \
        uv run python "$script_path" < <(yes "") 2>&1 | tee "$retry_log"
        local retry_rc=${PIPESTATUS[0]}
        set -e

        rm -f "$tmp_log" "$retry_log"

        if [ "$retry_rc" -eq 0 ]; then
            return 0
        fi

        echo ""
        echo "Retry with AGENTIC_DISABLE_SSL=1 also failed."
        exit "$retry_rc"
    fi

    rm -f "$tmp_log"
    exit "$rc"
}

echo "Select an example to run:"
echo "1) Basic Prioritization (Multi-Criteria Scoring)"
echo "2) Advanced Prioritization (Dynamic Rebalancing & Deadline-Aware)"
echo "3) Run All Examples"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Prioritization..."
        echo "----------------------------------------"
        run_example src/prioritization_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced Prioritization..."
        echo "----------------------------------------"
        run_example src/prioritization_advanced.py
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic Prioritization"
        echo "----------------------------------------"
        run_example src/prioritization_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced Prioritization"
        echo "----------------------------------------"
        run_example src/prioritization_advanced.py
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
