#!/bin/bash

# Tree of Thoughts Pattern Examples Runner
# This script runs the different Tree of Thoughts pattern examples

set -e  # Exit on error

echo "========================================="
echo "  Tree of Thoughts Pattern Examples"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic ToT with BFS (Game of 24 Puzzle)"
echo "2) Advanced ToT with Beam Search (Creative Writing)"
echo "3) Run All Examples"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Tree of Thoughts with BFS..."
        echo "----------------------------------------"
        uv run python src/tot_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced Tree of Thoughts with Beam Search..."
        echo "----------------------------------------"
        uv run python src/tot_advanced.py
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic ToT with BFS"
        echo "----------------------------------------"
        uv run python src/tot_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced ToT with Beam Search"
        echo "----------------------------------------"
        uv run python src/tot_advanced.py
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
