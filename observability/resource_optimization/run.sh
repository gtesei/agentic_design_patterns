#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       Resource Optimization Pattern - Example Runner          ║"
echo "║                                                                ║"
echo "║  Demonstrates cost reduction and performance optimization      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if .env file exists
if [ ! -f "../../.env" ]; then
    echo -e "${YELLOW}Warning: ../../.env file not found${NC}"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo "Example: OPENAI_API_KEY=sk-..."
    exit 1
fi

# Function to run a Python script
run_script() {
    local script=$1
    echo -e "\n${GREEN}Running: ${script}${NC}\n"
    python "src/${script}"
    echo -e "\n${GREEN}Completed: ${script}${NC}\n"
}

# Main menu
echo "Select an example to run:"
echo "1. Basic Optimization (caching + prompt optimization + model routing)"
echo "2. Advanced Optimization (batching + predictive caching + cost-aware routing)"
echo "3. Run all examples"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        run_script "optimization_basic.py"
        ;;
    2)
        run_script "optimization_advanced.py"
        ;;
    3)
        echo -e "\n${BLUE}Running all examples...${NC}\n"
        run_script "optimization_basic.py"
        echo -e "\n${YELLOW}═══════════════════════════════════════════════════════════${NC}\n"
        run_script "optimization_advanced.py"
        ;;
    4)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${YELLOW}Invalid choice. Please run again and select 1-4.${NC}"
        exit 1
        ;;
esac

echo -e "\n${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                   Examples Complete!                           ║"
echo "║                                                                ║"
echo "║  Key Takeaways:                                                ║"
echo "║  • Caching provides 60-80% cost reduction                      ║"
echo "║  • Model routing balances cost and quality                     ║"
echo "║  • Prompt optimization reduces token usage 20-40%              ║"
echo "║  • Batching increases throughput 2-5x                          ║"
echo "║  • Always monitor quality alongside cost/speed                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
