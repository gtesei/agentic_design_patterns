#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Graph of Thoughts Pattern - Demo Runner${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if .env exists in parent directory
if [ ! -f "../../.env" ]; then
    echo -e "${RED}Error: .env file not found in parent directory${NC}"
    echo -e "${YELLOW}Please create ../../.env with your OPENAI_API_KEY${NC}"
    exit 1
fi

# Function to check if Python packages are installed
check_dependencies() {
    python3 -c "import langchain; import networkx; import dotenv" 2>/dev/null
    return $?
}

# Function to install dependencies
install_deps() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q langchain langchain-openai python-dotenv networkx
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Dependencies installed successfully!${NC}"
    else
        echo -e "${RED}Failed to install dependencies${NC}"
        exit 1
    fi
}

# Check dependencies
if ! check_dependencies; then
    echo -e "${YELLOW}Dependencies not found.${NC}"
    read -p "Install dependencies now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_deps
    else
        echo -e "${RED}Dependencies required. Exiting.${NC}"
        exit 1
    fi
fi

# Main menu
show_menu() {
    echo ""
    echo -e "${GREEN}Select an example to run:${NC}"
    echo ""
    echo "  1) Basic GoT (DAG with single round)"
    echo "     - Multi-perspective analysis"
    echo "     - Simple graph structure"
    echo "     - Quick demonstration"
    echo ""
    echo "  2) Advanced GoT (Multi-round consensus)"
    echo "     - Iterative refinement"
    echo "     - Agent-based perspectives"
    echo "     - Consensus building"
    echo ""
    echo "  3) Run all examples"
    echo ""
    echo "  4) Install/Update dependencies"
    echo ""
    echo "  q) Quit"
    echo ""
    echo -ne "${BLUE}Enter your choice: ${NC}"
}

# Run basic example
run_basic() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}   Running: Basic Graph of Thoughts${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    python3 src/got_basic.py
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Basic example completed successfully!${NC}"
    else
        echo -e "${RED}✗ Basic example failed${NC}"
    fi
}

# Run advanced example
run_advanced() {
    echo ""
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}   Running: Advanced Graph of Thoughts${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
    python3 src/got_advanced.py
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Advanced example completed successfully!${NC}"
    else
        echo -e "${RED}✗ Advanced example failed${NC}"
    fi
}

# Main loop
while true; do
    show_menu
    read -r choice

    case $choice in
        1)
            run_basic
            ;;
        2)
            run_advanced
            ;;
        3)
            run_basic
            echo ""
            echo -e "${YELLOW}Press Enter to continue to Advanced example...${NC}"
            read
            run_advanced
            ;;
        4)
            install_deps
            ;;
        q|Q)
            echo ""
            echo -e "${GREEN}Thank you for exploring Graph of Thoughts!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            ;;
    esac

    echo ""
    echo -e "${YELLOW}Press Enter to return to menu...${NC}"
    read
done
