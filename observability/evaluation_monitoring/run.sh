#!/bin/bash

# Evaluation and Monitoring Pattern Examples Runner
# This script runs different monitoring and evaluation examples

set -e  # Exit on error

echo "========================================="
echo "  Evaluation & Monitoring Examples"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic Monitoring (Metrics & Logs)"
echo "2) Advanced Evaluation (Quality Assessment with LLM-as-Judge)"
echo "3) Run All Examples"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic Monitoring..."
        echo "----------------------------------------"
        uv run python src/monitoring_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced Evaluation..."
        echo "----------------------------------------"
        uv run python src/monitoring_advanced.py
        ;;
    3)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic Monitoring"
        echo "----------------------------------------"
        uv run python src/monitoring_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced Evaluation"
        echo "----------------------------------------"
        uv run python src/monitoring_advanced.py
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
