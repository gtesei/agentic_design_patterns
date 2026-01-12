"""
Tool Use Pattern: Stock Market Agent
This example demonstrates using LangChain tools to create an agent that can 
access real-time stock data, perform calculations, and provide market analysis.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4")

# --- Define Stock Market Tools ---

@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price and basic information for a given ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
        
    Returns:
        Dictionary with current price, change, and volume information
        
    Examples:
        get_stock_price("AAPL") -> Apple stock information
        get_stock_price("MSFT") -> Microsoft stock information
    """
    # In production, use actual API like yfinance, Alpha Vantage, or Yahoo Finance
    # This is a mock implementation
    mock_data = {
        "AAPL": {"price": 178.52, "change": 2.34, "change_pct": 1.33, "volume": 52_340_000},
        "GOOGL": {"price": 142.87, "change": -0.89, "change_pct": -0.62, "volume": 28_450_000},
        "MSFT": {"price": 384.79, "change": 5.12, "change_pct": 1.35, "volume": 31_220_000},
        "TSLA": {"price": 248.42, "change": -3.21, "change_pct": -1.28, "volume": 89_340_000},
        "AMZN": {"price": 175.33, "change": 1.87, "change_pct": 1.08, "volume": 42_150_000},
        "NVDA": {"price": 495.22, "change": 8.45, "change_pct": 1.74, "volume": 45_780_000},
    }
    
    symbol = symbol.upper()
    
    if symbol not in mock_data:
        return {
            "error": f"Stock symbol '{symbol}' not found. Available symbols: {', '.join(mock_data.keys())}"
        }
    
    data = mock_data[symbol]
    return {
        "symbol": symbol,
        "current_price": f"${data['price']:.2f}",
        "change": f"${data['change']:+.2f}",
        "change_percent": f"{data['change_pct']:+.2f}%",
        "volume": f"{data['volume']:,}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@tool
def calculate_portfolio_value(holdings: str) -> dict:
    """Calculate the total value of a stock portfolio.
    
    Args:
        holdings: Comma-separated list of "SYMBOL:SHARES" pairs
                 Example: "AAPL:10,GOOGL:5,TSLA:3"
        
    Returns:
        Dictionary with portfolio breakdown and total value
    """
    # Parse holdings
    portfolio = {}
    total_value = 0.0
    
    try:
        for holding in holdings.split(","):
            symbol, shares = holding.strip().split(":")
            symbol = symbol.upper()
            shares = int(shares)
            
            # Get current price
            price_data = get_stock_price.invoke({"symbol": symbol})
            if "error" in price_data:
                return price_data
            
            # Extract numeric price
            price = float(price_data["current_price"].replace("$", "").replace(",", ""))
            value = price * shares
            
            portfolio[symbol] = {
                "shares": shares,
                "price_per_share": f"${price:.2f}",
                "total_value": f"${value:,.2f}"
            }
            total_value += value
            
    except Exception as e:
        return {"error": f"Invalid holdings format. Use 'SYMBOL:SHARES,SYMBOL:SHARES'. Error: {str(e)}"}
    
    return {
        "portfolio": portfolio,
        "total_portfolio_value": f"${total_value:,.2f}",
        "number_of_positions": len(portfolio)
    }

@tool
def calculate_return_on_investment(symbol: str, purchase_price: float, current_shares: int) -> dict:
    """Calculate ROI for a stock investment.
    
    Args:
        symbol: Stock ticker symbol
        purchase_price: Original purchase price per share
        current_shares: Number of shares owned
        
    Returns:
        Dictionary with ROI calculations
    """
    # Get current price
    price_data = get_stock_price.invoke({"symbol": symbol})
    if "error" in price_data:
        return price_data
    
    current_price = float(price_data["current_price"].replace("$", "").replace(",", ""))
    
    # Calculate ROI
    original_value = purchase_price * current_shares
    current_value = current_price * current_shares
    profit_loss = current_value - original_value
    roi_percent = (profit_loss / original_value) * 100
    
    return {
        "symbol": symbol.upper(),
        "shares": current_shares,
        "purchase_price": f"${purchase_price:.2f}",
        "current_price": f"${current_price:.2f}",
        "original_investment": f"${original_value:,.2f}",
        "current_value": f"${current_value:,.2f}",
        "profit_loss": f"${profit_loss:+,.2f}",
        "roi_percentage": f"{roi_percent:+.2f}%"
    }

@tool
def compare_stocks(symbols: str) -> dict:
    """Compare multiple stocks side-by-side.
    
    Args:
        symbols: Comma-separated list of stock symbols
                Example: "AAPL,GOOGL,MSFT"
        
    Returns:
        Dictionary with comparison data
    """
    comparison = {}
    
    for symbol in symbols.split(","):
        symbol = symbol.strip().upper()
        data = get_stock_price.invoke({"symbol": symbol})
        
        if "error" not in data:
            comparison[symbol] = {
                "price": data["current_price"],
                "change": data["change"],
                "change_pct": data["change_percent"],
                "volume": data["volume"]
            }
    
    if not comparison:
        return {"error": "No valid stock symbols provided"}
    
    # Find best/worst performers
    stocks_with_pct = [(s, float(d["change_pct"].replace("%", ""))) 
                       for s, d in comparison.items()]
    best = max(stocks_with_pct, key=lambda x: x[1])
    worst = min(stocks_with_pct, key=lambda x: x[1])
    
    return {
        "stocks": comparison,
        "best_performer": {"symbol": best[0], "change": f"{best[1]:+.2f}%"},
        "worst_performer": {"symbol": worst[0], "change": f"{worst[1]:+.2f}%"}
    }

@tool
def calculate_percentage(expression: str) -> str:
    """Calculate percentage changes or values.
    
    Args:
        expression: Math expression to evaluate (e.g., "150 * 0.15" for 15% of 150)
        
    Returns:
        Result of the calculation
    """
    try:
        # Simple eval for demo - in production, use safer alternatives
        result = eval(expression, {"__builtins__": {}})
        return f"${result:,.2f}" if result > 1 else f"{result:.4f}"
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_market_summary() -> dict:
    """Get a summary of overall market performance.
    
    Returns:
        Dictionary with major indices and market sentiment
    """
    return {
        "major_indices": {
            "S&P 500": {"value": "4,783.45", "change": "+0.89%"},
            "Dow Jones": {"value": "37,545.33", "change": "+0.67%"},
            "NASDAQ": {"value": "15,074.57", "change": "+1.24%"}
        },
        "market_sentiment": "Bullish",
        "top_gainers": ["NVDA (+1.74%)", "MSFT (+1.35%)", "AAPL (+1.33%)"],
        "top_losers": ["TSLA (-1.28%)", "GOOGL (-0.62%)"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- Create Stock Market Agent ---

# Map tool names to functions
TOOLS_MAP = {
    "get_stock_price": get_stock_price,
    "calculate_portfolio_value": calculate_portfolio_value,
    "calculate_return_on_investment": calculate_return_on_investment,
    "compare_stocks": compare_stocks,
    "calculate_percentage": calculate_percentage,
    "get_market_summary": get_market_summary
}

def create_stock_agent():
    """Create a stock market agent with all tools."""
    
    tools = list(TOOLS_MAP.values())
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools

def run_agent_with_tools(query: str, llm_with_tools):
    """Run agent with tool calling loop."""
    
    messages = [
        {"role": "system", "content": """You are a helpful stock market analyst assistant. You can:

1. Get real-time stock prices and information
2. Calculate portfolio values
3. Compute ROI (Return on Investment)
4. Compare multiple stocks
5. Perform financial calculations
6. Provide market summaries

When users ask about stocks:
- Always use the appropriate tools to get current data
- Provide clear, actionable insights
- Explain your calculations
- Format numbers clearly with $ and % symbols

Be conversational and helpful, but always back up your statements with data from the tools."""},
        {"role": "user", "content": query}
    ]
    
    # Tool calling loop (max 10 iterations)
    for _ in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if tool calls are present
        if not response.tool_calls:
            # No more tool calls, return final answer
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Execute the tool
            tool_func = TOOLS_MAP[tool_name]
            tool_result = tool_func.invoke(tool_args)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call["id"]
            })
    
    return "Max iterations reached"

# --- Example Usage ---

def run_stock_agent():
    """Run interactive examples with the stock market agent."""
    
    agent = create_stock_agent()
    
    print("="*80)
    print("STOCK MARKET AGENT WITH TOOL USE")
    print("="*80)
    
    # Example 1: Simple price query
    print("\n" + "="*80)
    print("EXAMPLE 1: Get Stock Price")
    print("="*80)
    
    query1 = "What's the current price of Apple stock?"
    print(f"\nUser: {query1}")
    
    response1 = run_agent_with_tools(query1, agent)
    print(f"\nAgent: {response1}")
    
    # Example 2: Portfolio calculation
    print("\n" + "="*80)
    print("EXAMPLE 2: Calculate Portfolio Value")
    print("="*80)
    
    query2 = "I own 10 shares of AAPL, 5 shares of GOOGL, and 3 shares of TSLA. What's my total portfolio worth?"
    print(f"\nUser: {query2}")
    
    response2 = run_agent_with_tools(query2, agent)
    print(f"\nAgent: {response2}")
    
    # Example 3: ROI calculation
    print("\n" + "="*80)
    print("EXAMPLE 3: Calculate ROI")
    print("="*80)
    
    query3 = "I bought 50 shares of Microsoft at $350 per share. What's my return on investment?"
    print(f"\nUser: {query3}")
    
    response3 = run_agent_with_tools(query3, agent)
    print(f"\nAgent: {response3}")
    
    # Example 4: Stock comparison
    print("\n" + "="*80)
    print("EXAMPLE 4: Compare Multiple Stocks")
    print("="*80)
    
    query4 = "Compare the performance of AAPL, MSFT, and NVDA today. Which one performed best?"
    print(f"\nUser: {query4}")
    
    response4 = run_agent_with_tools(query4, agent)
    print(f"\nAgent: {response4}")
    
    # Example 5: Market summary
    print("\n" + "="*80)
    print("EXAMPLE 5: Market Summary")
    print("="*80)
    
    query5 = "Give me a summary of how the market is doing today"
    print(f"\nUser: {query5}")
    
    response5 = run_agent_with_tools(query5, agent)
    print(f"\nAgent: {response5}")
    
    # Example 6: Complex multi-tool query
    print("\n" + "="*80)
    print("EXAMPLE 6: Complex Multi-Tool Query")
    print("="*80)
    
    query6 = """I'm thinking about investing $10,000. Should I split it equally between 
AAPL, MSFT, and NVDA based on their current performance? How many shares of each 
would I get, and what would be the total value?"""
    print(f"\nUser: {query6}")
    
    response6 = run_agent_with_tools(query6, agent)
    print(f"\nAgent: {response6}")

if __name__ == "__main__":
    run_stock_agent()