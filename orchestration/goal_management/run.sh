#!/bin/bash

# Goal Management Pattern Examples Runner
# This script runs the different Goal Management pattern examples

set -e  # Exit on error

echo "========================================="
echo "  Goal Management Pattern Examples"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic Goal Management (Hierarchical Decomposition)"
echo "2) Advanced Goal Management (Dynamic Replanning & Parallel Execution)"
echo "3) Run All Examples"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Goal Management..."
        echo "----------------------------------------"
        uv run python src/goal_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced Goal Management..."
        echo "----------------------------------------"
        uv run python src/goal_advanced.py
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic Goal Management"
        echo "----------------------------------------"
        uv run python src/goal_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced Goal Management"
        echo "----------------------------------------"
        uv run python src/goal_advanced.py
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
