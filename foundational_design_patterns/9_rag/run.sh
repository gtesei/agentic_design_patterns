#!/bin/bash

# RAG Pattern Examples Runner

set -e  # Exit on error

echo "========================================="
echo "  RAG Pattern Examples"
echo "========================================="
echo ""

if [ ! -f "../../.env" ]; then
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
read -p "Enter your choice (1-5): " choice

run_example() {
    local file="$1"
    echo "----------------------------------------"
    uv run python "$file"
    echo ""
}

case $choice in
    1)
        echo "\nRunning Basic Hybrid RAG..."
        run_example src/rag_basic.py
        ;;
    2)
        echo "\nRunning Advanced RAG..."
        run_example src/rag_advanced.py
        ;;
    3)
        echo "\nRunning Agentic RAG..."
        run_example src/rag_agentic.py
        ;;
    4)
        echo "\nRunning RAG Eval + Failure Modes..."
        run_example src/rag_eval_and_failure_modes.py
        ;;
    5)
        echo "\nRunning All Examples..."
        run_example src/rag_basic.py
        run_example src/rag_advanced.py
        run_example src/rag_agentic.py
        run_example src/rag_eval_and_failure_modes.py
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "========================================="
echo "  Examples completed!"
echo "========================================="
