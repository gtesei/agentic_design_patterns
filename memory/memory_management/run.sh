#!/bin/bash

# Memory Management Pattern - Run Script

set -e

echo "=========================================="
echo "   Memory Management Pattern Demo"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f "../../.env" ]; then
    echo "Error: ../../.env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

# Default to running both if no argument provided
MODE=${1:-both}

case $MODE in
    basic)
        echo "Running Basic Memory Demo (Buffer + Summarization)..."
        echo ""
        python src/memory_basic.py
        ;;
    advanced)
        echo "Running Advanced Memory Demo (Semantic + Entity)..."
        echo ""
        python src/memory_advanced.py
        ;;
    both)
        echo "Running Basic Memory Demo..."
        echo "=========================================="
        echo ""
        python src/memory_basic.py
        echo ""
        echo ""
        echo "=========================================="
        echo "Running Advanced Memory Demo..."
        echo "=========================================="
        echo ""
        python src/memory_advanced.py
        ;;
    *)
        echo "Usage: ./run.sh [basic|advanced|both]"
        echo ""
        echo "Options:"
        echo "  basic    - Run basic memory demo (buffer + summarization)"
        echo "  advanced - Run advanced memory demo (semantic + entity)"
        echo "  both     - Run both demos (default)"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "   Demo Complete!"
echo "=========================================="
