#!/bin/bash

# Exploration and Discovery Pattern Examples Runner
# This script runs the different exploration pattern examples

set -e  # Exit on error

echo "========================================="
echo "  Exploration and Discovery Examples"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic Epsilon-Greedy Exploration"
echo "2) Advanced UCB (Upper Confidence Bound) Exploration"
echo "3) Run All Examples"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Epsilon-Greedy Exploration..."
        echo "----------------------------------------"
        uv run python src/exploration_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced UCB Exploration..."
        echo "----------------------------------------"
        uv run python src/exploration_advanced.py
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic Epsilon-Greedy Exploration"
        echo "----------------------------------------"
        uv run python src/exploration_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced UCB Exploration"
        echo "----------------------------------------"
        uv run python src/exploration_advanced.py
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
