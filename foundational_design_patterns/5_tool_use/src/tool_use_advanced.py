"""
Tool Use Pattern: Advanced Code Generation Agent
This example demonstrates using tools to generate, execute, and refine Python code,
with a focus on data visualization using seaborn and matplotlib.
"""

import os
import sys
import json
import traceback
import subprocess
from datetime import datetime
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

# Initialize the Language Model
llm = ChatOpenAI(temperature=0, model="gpt-4")

# --- Define Code Generation Tools ---

@tool
def generate_python_code(description: str, requirements: str = None) -> dict:
    """Generate Python code based on a description.
    
    Args:
        description: What the code should do
        requirements: Additional requirements or constraints
        
    Returns:
        Dictionary with generated code and explanation
    """
    # This is a placeholder - the LLM will actually generate the code
    # In practice, you might use a code-specific model or API
    return {
        "message": "Code generation request received",
        "description": description,
        "requirements": requirements or "None specified",
        "note": "The LLM will generate actual code based on this request"
    }

@tool
def execute_python_code(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a safe environment and return the results.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
        
    Returns:
        Dictionary with execution results, output, and any errors
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture stdout and stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    result = {
        "success": False,
        "output": "",
        "error": "",
        "files_created": []
    }
    
    try:
        # Prepare execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'output_dir': output_dir
        }
        
        # Redirect stdout and stderr
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code
            exec(code, exec_globals)
        
        result["success"] = True
        result["output"] = stdout_capture.getvalue()
        
        # Check for created files
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            result["files_created"] = files
            
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result["output"] = stdout_capture.getvalue()
    
    return result

@tool
def install_python_package(package_name: str) -> dict:
    """Install a Python package using pip.
    
    Args:
        package_name: Name of the package to install (e.g., 'seaborn', 'pandas')
        
    Returns:
        Dictionary with installation results
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return {
            "success": result.returncode == 0,
            "package": package_name,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else ""
        }
    except Exception as e:
        return {
            "success": False,
            "package": package_name,
            "error": str(e)
        }

@tool
def check_code_syntax(code: str) -> dict:
    """Check if Python code has valid syntax without executing it.
    
    Args:
        code: Python code to check
        
    Returns:
        Dictionary with syntax validation results
    """
    try:
        compile(code, '<string>', 'exec')
        return {
            "valid": True,
            "message": "Code syntax is valid"
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e),
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

@tool
def list_available_datasets() -> dict:
    """List commonly available datasets for visualization examples.
    
    Returns:
        Dictionary with available datasets and their descriptions
    """
    datasets = {
        "seaborn_builtin": [
            {"name": "iris", "description": "Iris flower dataset with measurements"},
            {"name": "tips", "description": "Restaurant tipping data"},
            {"name": "titanic", "description": "Titanic passenger survival data"},
            {"name": "diamonds", "description": "Diamond prices and attributes"},
            {"name": "flights", "description": "Airline passenger data"},
            {"name": "penguins", "description": "Palmer penguins measurements"}
        ],
        "sklearn_builtin": [
            {"name": "load_iris", "description": "Iris dataset from sklearn"},
            {"name": "load_wine", "description": "Wine classification dataset"},
            {"name": "load_breast_cancer", "description": "Breast cancer dataset"}
        ]
    }
    
    return {
        "available_datasets": datasets,
        "note": "These datasets can be loaded directly in your code"
    }

@tool
def get_plot_types() -> dict:
    """Get information about available plot types in seaborn.
    
    Returns:
        Dictionary with plot types and their use cases
    """
    plot_types = {
        "distribution": {
            "histplot": "Histogram for univariate distributions",
            "kdeplot": "Kernel density estimate plot",
            "ecdfplot": "Empirical cumulative distribution function",
            "rugplot": "Marginal rug plot"
        },
        "categorical": {
            "stripplot": "Scatter plot for categorical data",
            "swarmplot": "Non-overlapping categorical scatter plot",
            "boxplot": "Box plot for distributions",
            "violinplot": "Violin plot combining box and KDE plots",
            "barplot": "Bar plot with error bars",
            "countplot": "Count plot for categorical frequencies"
        },
        "relational": {
            "scatterplot": "Scatter plot for two variables",
            "lineplot": "Line plot for trends over time"
        },
        "regression": {
            "regplot": "Scatter plot with regression line",
            "residplot": "Residual plot for regression"
        },
        "matrix": {
            "heatmap": "Heatmap for matrix data",
            "clustermap": "Hierarchically clustered heatmap"
        }
    }
    
    return {
        "plot_types": plot_types,
        "note": "Choose plot type based on data structure and analysis goals"
    }

@tool
def validate_visualization_requirements(description: str) -> dict:
    """Analyze visualization requirements and suggest appropriate plot types.
    
    Args:
        description: Description of what to visualize
        
    Returns:
        Dictionary with analysis and recommendations
    """
    description_lower = description.lower()
    
    suggestions = []
    
    # Analyze keywords
    if any(word in description_lower for word in ["distribution", "histogram", "density"]):
        suggestions.append({
            "plot": "histplot or kdeplot",
            "reason": "Good for showing distributions"
        })
    
    if any(word in description_lower for word in ["relationship", "correlation", "scatter"]):
        suggestions.append({
            "plot": "scatterplot or regplot",
            "reason": "Shows relationships between variables"
        })
    
    if any(word in description_lower for word in ["category", "categorical", "group"]):
        suggestions.append({
            "plot": "boxplot, violinplot, or barplot",
            "reason": "Effective for categorical comparisons"
        })
    
    if any(word in description_lower for word in ["trend", "time", "line"]):
        suggestions.append({
            "plot": "lineplot",
            "reason": "Shows trends over time or continuous variables"
        })
    
    if any(word in description_lower for word in ["heatmap", "matrix", "correlation matrix"]):
        suggestions.append({
            "plot": "heatmap",
            "reason": "Visualizes matrix data effectively"
        })
    
    return {
        "analysis": f"Analyzed: '{description}'",
        "suggestions": suggestions if suggestions else [{"note": "No specific plot type identified. Consider using scatterplot for general exploration."}],
        "recommendation": "Start with exploratory plots, then refine based on insights"
    }

# --- Create Code Generation Agent ---

TOOLS_MAP = {
    "generate_python_code": generate_python_code,
    "execute_python_code": execute_python_code,
    "install_python_package": install_python_package,
    "check_code_syntax": check_code_syntax,
    "list_available_datasets": list_available_datasets,
    "get_plot_types": get_plot_types,
    "validate_visualization_requirements": validate_visualization_requirements
}

def create_code_agent():
    """Create a code generation agent with all tools."""
    tools = list(TOOLS_MAP.values())
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

def run_agent_with_tools(query: str, llm_with_tools):
    """Run agent with tool calling loop."""
    
    messages = [
        {"role": "system", "content": """You are an expert Python programmer and data visualization specialist. You can:

1. Generate Python code for data visualization using seaborn, matplotlib, pandas
2. Execute code safely and return results
3. Check code syntax before execution
4. Install required packages
5. Recommend appropriate plot types
6. Work with various datasets

When generating code:
- Always import required libraries at the top
- Use seaborn's built-in datasets when appropriate
- Save plots to 'output_dir' variable (already defined in execution environment)
- Use descriptive variable names
- Add comments to explain complex operations
- Set appropriate figure sizes (e.g., plt.figure(figsize=(10, 6)))
- Use informative titles and labels
- Save plots as PNG files with clear names

CRITICAL: When writing code, write the COMPLETE, EXECUTABLE code directly. Don't use placeholders.
Always save plots using: plt.savefig(os.path.join(output_dir, 'filename.png'), dpi=300, bbox_inches='tight')

After generating code:
1. Check syntax with check_code_syntax
2. Execute with execute_python_code
3. Report results to user

Be helpful, accurate, and always test the code you generate."""},
        {"role": "user", "content": query}
    ]
    
    # Tool calling loop
    for iteration in range(15):  # Increased iterations for complex workflows
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if tool calls are present
        if not response.tool_calls:
            return response.content
        
        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\n[Iteration {iteration + 1}] Using tool: {tool_name}")
            
            # Execute the tool
            tool_func = TOOLS_MAP[tool_name]
            tool_result = tool_func.invoke(tool_args)
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call["id"]
            })
    
    return "Max iterations reached. Please simplify your request."

# --- Example Usage ---

def run_code_generation_examples():
    """Run examples demonstrating code generation capabilities."""
    
    agent = create_code_agent()
    
    print("="*80)
    print("ADVANCED CODE GENERATION AGENT WITH TOOL USE")
    print("="*80)
    
    examples = [
        {
            "title": "Simple Distribution Plot",
            "query": "Create a histogram showing the distribution of sepal length in the iris dataset. Use seaborn and make it look professional."
        },
        {
            "title": "Correlation Heatmap",
            "query": "Generate a correlation heatmap for the iris dataset showing relationships between all numeric features. Use a good color palette."
        },
        {
            "title": "Multi-Plot Comparison",
            "query": "Create a figure with 2x2 subplots comparing different features of the penguins dataset. Show species distributions."
        },
        {
            "title": "Complex Violin Plot",
            "query": "Make a violin plot comparing tip amounts by day of week from the tips dataset. Split by time (lunch/dinner) and add individual data points."
        },
        {
            "title": "Regression Analysis",
            "query": "Create a regression plot showing the relationship between total bill and tip in the tips dataset. Include confidence intervals."
        }
    ]
    
    # Run first 2 examples for demonstration
    for i, example in enumerate(examples[:2], 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}: {example['title']}")
        print("="*80)
        print(f"\nUser: {example['query']}")
        print("\n" + "-"*80)
        
        response = run_agent_with_tools(example['query'], agent)
        print(f"\nAgent Response:\n{response}")

if __name__ == "__main__":
    run_code_generation_examples()