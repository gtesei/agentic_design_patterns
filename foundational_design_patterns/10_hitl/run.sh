#!/bin/bash

# HITL Pattern Examples Runner
# This script runs the different Human-in-the-Loop pattern examples

set -e  # Exit on error

echo "========================================="
echo "  Human-in-the-Loop (HITL) Pattern Examples"
echo "========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo "Error: .env file not found in project root"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

echo "Select an example to run:"
echo "1) Basic HITL with Console Approval"
echo "2) Advanced HITL with Risk-Based Checkpoints"
echo "3) LangGraph HITL Integration"
echo "4) Run All Examples"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running Basic HITL with Console Approval..."
        echo "----------------------------------------"
        uv run python src/hitl_basic.py
        ;;
    2)
        echo ""
        echo "Running Advanced HITL with Risk-Based Checkpoints..."
        echo "----------------------------------------"
        uv run python src/hitl_advanced.py
        ;;
    3)
        echo ""
        echo "Running LangGraph HITL Integration..."
        echo "----------------------------------------"
        uv run python src/hitl_langgraph.py
        ;;
    4)
        echo ""
        echo "Running All Examples..."
        echo "========================================="
        echo ""
        echo "1. Basic HITL with Console Approval"
        echo "----------------------------------------"
        uv run python src/hitl_basic.py
        echo ""
        echo "========================================="
        echo ""
        echo "2. Advanced HITL with Risk-Based Checkpoints"
        echo "----------------------------------------"
        uv run python src/hitl_advanced.py
        echo ""
        echo "========================================="
        echo ""
        echo "3. LangGraph HITL Integration"
        echo "----------------------------------------"
        uv run python src/hitl_langgraph.py
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
