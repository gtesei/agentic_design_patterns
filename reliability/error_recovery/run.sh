#!/bin/bash

# Error Recovery Pattern - Run Script

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Error Recovery Pattern - Execution Script            ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import tenacity" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -e .
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Check for .env file
if [ ! -f "../../.env" ]; then
    echo -e "${YELLOW}Warning: .env file not found at ../../.env${NC}"
    echo -e "${YELLOW}Please create it with: OPENAI_API_KEY=your-key-here${NC}"
    echo ""
fi

echo ""

# Run based on argument
case "${1:-all}" in
    basic)
        echo -e "${GREEN}Running Basic Error Recovery Example...${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        python src/recovery_basic.py
        ;;
    advanced)
        echo -e "${GREEN}Running Advanced Self-Correction Example...${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        python src/recovery_advanced.py
        ;;
    all|*)
        echo -e "${GREEN}Running All Examples...${NC}"
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}Example 1: Basic Error Recovery${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        python src/recovery_basic.py
        echo ""
        echo ""
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}Example 2: Advanced Self-Correction${NC}"
        echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
        python src/recovery_advanced.py
        ;;
esac

echo ""
echo -e "${GREEN}✓ Execution complete!${NC}"
echo ""
echo -e "${BLUE}For more information:${NC}"
echo -e "  • Full documentation: ${YELLOW}README.md${NC}"
echo -e "  • Quick start guide: ${YELLOW}QUICK_START.md${NC}"
echo -e "  • Source code: ${YELLOW}src/${NC}"
