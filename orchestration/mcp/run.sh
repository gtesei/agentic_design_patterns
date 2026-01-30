#!/bin/bash

# Model Context Protocol (MCP) Pattern Runner

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Model Context Protocol (MCP) Pattern                     ║"
echo "║     Standardized tool integration for LLMs                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo -e "${YELLOW}Warning: ../../.env file not found${NC}"
    echo "Please create it with: OPENAI_API_KEY=sk-your-key-here"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import langchain_openai" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q langchain langchain-openai langgraph python-dotenv pydantic rich 2>/dev/null || {
        echo -e "${YELLOW}Note: Some dependencies may not be installed. Run: pip install langchain langchain-openai langgraph python-dotenv pydantic rich${NC}"
    }
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${YELLOW}Warning: Python 3.11+ required, found $PYTHON_VERSION${NC}"
fi

# Determine which example to run
EXAMPLE=${1:-basic}

case $EXAMPLE in
    basic)
        echo -e "${GREEN}Running: Basic MCP Server${NC}"
        echo "Demonstrates:"
        echo "  - Simple MCP server with tools"
        echo "  - Tool discovery and invocation"
        echo "  - File operations and calculations"
        echo "  - Basic resource exposure"
        echo ""
        python3 src/mcp_basic.py
        ;;

    advanced)
        echo -e "${GREEN}Running: Advanced Multi-Server MCP${NC}"
        echo "Demonstrates:"
        echo "  - Multiple specialized MCP servers"
        echo "  - Server composition and orchestration"
        echo "  - Resource subscriptions"
        echo "  - LangChain integration"
        echo "  - Rich monitoring dashboard"
        echo ""
        python3 src/mcp_advanced.py
        ;;

    *)
        echo -e "${YELLOW}Usage: ./run.sh [basic|advanced]${NC}"
        echo ""
        echo "Examples:"
        echo "  ./run.sh basic     - Run basic MCP server example"
        echo "  ./run.sh advanced  - Run advanced multi-server example"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Example completed successfully${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
