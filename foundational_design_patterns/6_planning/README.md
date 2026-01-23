# Planning Pattern

## Overview

The **Planning Pattern** enables agentic systems to decompose complex, multi-step goals into structured sequences of actionable sub-tasks, creating a strategic roadmap before execution. This transforms reactive agents into proactive, strategic executors capable of handling sophisticated workflows.

## Why Use This Pattern?

Many real-world tasks are too complex to be solved with a single action or prompt. They require:
- Multiple interdependent steps executed in a specific order
- Coordination across different tools and resources
- Adaptive decision-making based on intermediate results
- Management of dependencies and prerequisites

The Planning pattern solves this by:
- **Decomposing complex goals** into manageable sub-tasks
- **Creating structured execution plans** with clear sequences
- **Identifying dependencies** between steps
- **Enabling strategic thinking** rather than reactive responses
- **Facilitating adaptation** when plans need to change

### Example: Without vs. With Planning
```
Without Planning (Reactive):
User: "Create a comprehensive market analysis report"
Agent: [Attempts to generate entire report in one shot]
Result: Incomplete, lacks depth, misses key aspects

With Planning (Strategic):
User: "Create a comprehensive market analysis report"
Agent: 
  Step 1: Identify key competitors and market segments
  Step 2: Gather data on each competitor (tools: web search, databases)
  Step 3: Analyze market trends and growth patterns
  Step 4: Collect customer sentiment data
  Step 5: Synthesize findings into structured report
  Step 6: Generate visualizations for key metrics
  Step 7: Write executive summary
→ Executes each step systematically
Result: Comprehensive, well-structured, data-driven report
```

## How It Works

1. **Goal Analysis**: Agent analyzes the high-level objective and requirements
2. **Task Decomposition**: Break down the goal into smaller, actionable sub-tasks
3. **Dependency Mapping**: Identify which tasks depend on others (sequencing)
4. **Resource Allocation**: Determine which tools/APIs are needed for each step
5. **Plan Generation**: Create structured execution plan with ordered steps
6. **Execution**: Execute steps sequentially (or in parallel when possible)
7. **Monitoring & Adaptation**: Track progress and adjust plan if needed

### Typical Architecture
```
User Goal/Objective
    ↓
Planning Agent
    ↓
[Analyze Goal]
    ↓
[Decompose into Sub-tasks]
    ↓
Generate Execution Plan:
  - Step 1: Research competitors
  - Step 2: Analyze market data  
  - Step 3: Gather customer feedback
  - Step 4: Synthesize findings
  - Step 5: Generate report
    ↓
Execute Step 1
    ↓
Execute Step 2
    ↓
Execute Step 3
    ↓
[Monitor Progress & Adapt if Needed]
    ↓
Execute Step 4
    ↓
Execute Step 5
    ↓
Final Output
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Research & Analysis**: Competitive analysis, market research, literature reviews
- **Content Creation**: Multi-section reports, comprehensive guides, documentation
- **Workflow Automation**: Employee onboarding, procurement processes, project setup
- **Data Pipeline**: Extract → Transform → Analyze → Visualize → Report
- **Complex Problem Solving**: Debugging systems, optimizing processes, strategic planning
- **Multi-tool Orchestration**: Tasks requiring coordinated use of multiple APIs/tools
- **Project Management**: Breaking down projects into phases and deliverables

### ❌ When NOT to Use

- **Simple, single-step tasks**: Direct questions, basic lookups, simple calculations
- **Time-critical responses**: Real-time chat, immediate answers
- **Well-defined, atomic operations**: Tasks that are already clearly scoped
- **Exploratory queries**: Open-ended browsing without clear objectives
- **When existing patterns suffice**: Use Prompt Chaining for pre-defined sequences

## Rule of Thumb

**Use Planning when:**
1. User's request is **too complex for a single action or tool**
2. Task requires **multiple interdependent operations** in sequence
3. Success depends on **executing steps in the right order**
4. You need to **orchestrate multiple tools/resources** strategically
5. The goal requires **synthesizing information from multiple sources**

**Don't use Planning when:**
1. Task is simple and direct (single tool call or prompt)
2. Sequence of steps is already pre-defined (use Prompt Chaining)
3. Real-time response is critical (planning adds latency)
4. Goal is exploratory without clear success criteria

## Framework Support

### LangChain with Plan-and-Execute
```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0, model="gpt-4")

# Planning prompt
planning_prompt = ChatPromptTemplate.from_template(
    """Given this objective: {objective}
    
    Create a detailed step-by-step plan to accomplish it.
    
    For each step:
    1. Describe what needs to be done
    2. Identify required tools or resources
    3. Note any dependencies on previous steps
    
    Plan:"""
)

# Execution prompt for each step
execution_prompt = ChatPromptTemplate.from_template(
    """Execute this step: {step}
    
    Context from previous steps: {context}
    
    Available tools: {tools}
    
    Result:"""
)

# Create chains
planner = planning_prompt | llm
executor = execution_prompt | llm

# Generate plan
plan = planner.invoke({"objective": "Create a market analysis report"})

# Execute each step
results = []
for step in plan.split('\n'):
    if step.strip():
        result = executor.invoke({
            "step": step,
            "context": results,
            "tools": available_tools
        })
        results.append(result)
```

### LangGraph (Stateful Planning)
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class PlanningState(TypedDict):
    objective: str
    plan: List[str]
    current_step: int
    results: List[dict]
    complete: bool

def plan_node(state: PlanningState):
    """Generate execution plan"""
    plan = generate_plan(state["objective"])
    return {"plan": plan, "current_step": 0}

def execute_step_node(state: PlanningState):
    """Execute current step in plan"""
    step = state["plan"][state["current_step"]]
    result = execute_step(step, state["results"])
    
    return {
        "results": state["results"] + [result],
        "current_step": state["current_step"] + 1
    }

def should_continue(state: PlanningState):
    if state["current_step"] >= len(state["plan"]):
        return "end"
    return "execute"

# Build graph
workflow = StateGraph(PlanningState)
workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_step_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_conditional_edges("executor", should_continue, {
    "execute": "executor",
    "end": END
})

app = workflow.compile()
```

### Google ADK (Deep Research)

Google's Deep Research demonstrates planning in action:
```python
# Google Deep Research uses Planning pattern internally:
# 1. Analyzes research question
# 2. Generates search strategy (plan)
# 3. Executes searches sequentially
# 4. Reflects on findings
# 5. Adapts plan based on results
# 6. Synthesizes comprehensive report
```

### ReAct Pattern Integration

Combine Planning with ReAct (Reasoning + Acting):
```python
# ReAct loop within planned steps
for step in plan:
    # Reason: What should I do for this step?
    reasoning = reason_about_step(step)
    
    # Act: Execute the action
    action = determine_action(reasoning)
    result = execute_action(action)
    
    # Observe: What happened?
    observation = observe_result(result)
    
    # Update plan if needed
    if observation.requires_replan:
        plan = replan(objective, completed_steps, observation)
```

## Key Benefits

### 🎯 Strategic Execution
- **Structured approach**: Clear roadmap from start to finish
- **Logical sequencing**: Tasks executed in optimal order
- **Dependency management**: Prerequisites handled automatically
- **Resource optimization**: Right tool for each step

### 🧠 Complex Problem Solving
- **Decomposition**: Break overwhelming tasks into manageable pieces
- **Systematic coverage**: Ensure all aspects are addressed
- **Quality improvement**: Each step can be optimized independently
- **Transparency**: Clear visibility into what's being done and why

### 🔄 Adaptability
- **Dynamic replanning**: Adjust strategy based on intermediate results
- **Error recovery**: Identify and retry failed steps
- **Incremental progress**: Partial completion still provides value
- **Learning**: Improve future plans based on execution outcomes

## Important Considerations

### ⚠️ Planning Overhead

**Initial Latency:**
- Planning phase adds upfront delay (5-15 seconds)
- More complex than direct execution
- Requires additional LLM calls for plan generation

**Token Costs:**
- Plan generation: 200-500 tokens
- Plan storage in context: Carried through execution
- Total: +20-40% token usage vs. direct execution

**When Overhead is Worth It:**
- Complex tasks where planning saves overall time
- Multi-step workflows that would fail without structure
- Tasks requiring tool coordination

### 🔍 Plan Quality Matters

**Good Plans:**
- Clear, specific steps with measurable outcomes
- Realistic resource requirements
- Proper dependency ordering
- Adaptable to changing conditions

**Poor Plans:**
- Vague steps ("do research", "analyze data")
- Missing dependencies (step 3 needs output from step 2)
- Unrealistic assumptions about available tools
- Too rigid (can't adapt to failures)

### 🛠️ Implementation Complexity

**Simple Planning (LangChain):**
- Generate plan → Execute steps sequentially
- Good for predictable workflows
- Limited adaptability

**Advanced Planning (LangGraph):**
- Stateful plan execution
- Dynamic replanning
- Parallel step execution when possible
- More complex but more powerful

## Best Practices

1. **Make plans concrete and actionable**: Each step should be executable
2. **Include success criteria**: How to know if a step succeeded
3. **Anticipate failure modes**: What to do if steps fail
4. **Enable plan visibility**: Show users the plan before execution
5. **Support human-in-the-loop**: Allow plan approval/modification
6. **Log plan execution**: Track which steps completed, which failed
7. **Implement replanning**: Adapt when circumstances change
8. **Balance detail and flexibility**: Detailed enough to execute, flexible enough to adapt

## Performance Metrics

Track these metrics for planning effectiveness:

- **Plan quality**: Completeness, accuracy, feasibility
- **Execution success rate**: % of plans that complete successfully
- **Time to completion**: Total time from goal to result
- **Step failure rate**: % of individual steps that fail
- **Replanning frequency**: How often plans need adjustment
- **Resource efficiency**: Tools/API calls used vs. optimal
- **User satisfaction**: Did the result meet the objective?

## Example Scenarios

### Scenario 1: Competitive Analysis Report
```
Objective: "Create a competitive analysis comparing our product to top 3 competitors"

Generated Plan:
1. Identify top 3 competitors (Tool: Web Search)
2. For each competitor:
   a. Gather product features (Tool: Web Scraping)
   b. Collect pricing information (Tool: Web Search)
   c. Analyze customer reviews (Tool: Review API)
3. Compare features side-by-side (Tool: Spreadsheet)
4. Analyze pricing strategies (Tool: Data Analysis)
5. Synthesize strengths/weaknesses (Tool: LLM)
6. Generate visual comparisons (Tool: Charts)
7. Write executive summary (Tool: LLM)
8. Compile into report (Tool: Document Generator)

Execution: 8 steps, 45 seconds, 12 tool calls
Result: Comprehensive 10-page report with data, visuals, insights
```

### Scenario 2: Employee Onboarding Automation
```
Objective: "Onboard new software engineer Sarah Chen"

Generated Plan:
1. Create email account (Tool: Google Workspace API)
2. Add to Slack channels (Tool: Slack API)
3. Grant GitHub access (Tool: GitHub API)
4. Assign onboarding buddy (Tool: HR Database)
5. Schedule orientation meetings (Tool: Calendar API)
6. Send welcome email with credentials (Tool: Email)
7. Create Jira account (Tool: Jira API)
8. Add to team distribution lists (Tool: Email API)
9. Order equipment (Tool: Procurement System)
10. Generate welcome packet (Tool: Document Generator)

Execution: 10 steps, 30 seconds, 10 tool calls
Result: Fully onboarded employee with all access and resources
```

### Scenario 3: Data Pipeline
```
Objective: "Analyze Q4 sales data and create executive dashboard"

Generated Plan:
1. Extract sales data from database (Tool: SQL Query)
2. Clean and validate data (Tool: Pandas)
3. Calculate key metrics (Tool: Data Analysis)
   - Revenue by region
   - Growth rates
   - Top products
4. Generate trend visualizations (Tool: Matplotlib/Seaborn)
5. Create comparison charts (Tool: Visualization)
6. Build interactive dashboard (Tool: Dashboard Framework)
7. Write insights summary (Tool: LLM)
8. Deploy dashboard (Tool: Deployment)

Execution: 8 steps, 60 seconds, 8 tool calls
Result: Live dashboard with visualizations and insights
```

## Planning Strategies

### Top-Down Decomposition
```
Goal: Create marketing campaign
├─ Research target audience
├─ Develop messaging strategy
├─ Create content assets
│  ├─ Write copy
│  ├─ Design visuals
│  └─ Produce videos
├─ Set up distribution channels
└─ Launch and monitor
```

### Dependency-Aware Planning
```
Step 1: Gather requirements (no dependencies)
Step 2: Design architecture (depends on Step 1)
Step 3a: Build frontend (depends on Step 2)
Step 3b: Build backend (depends on Step 2)  [Can run in parallel]
Step 4: Integration testing (depends on 3a, 3b)
Step 5: Deployment (depends on Step 4)
```

### Iterative Planning (Agile)
```
Sprint 1 Plan:
- User research
- MVP design
- Core features

[Execute Sprint 1]

Sprint 2 Plan (adapted based on Sprint 1 results):
- Implement feedback
- Add requested features
- Performance optimization
```

## Related Patterns

- **Prompt Chaining**: Fixed sequence vs. dynamic planning
- **Tool Use**: Plans orchestrate tool usage
- **Reflection**: Can reflect on plan quality before execution
- **Routing**: Different plans for different goal types
- **Parallelization**: Execute independent plan steps concurrently

## Conclusion

The Planning pattern is essential for building agents that can handle complex, multi-step objectives with strategic thinking. By decomposing goals into actionable plans before execution, agents become proactive problem-solvers rather than reactive responders.

**Use Planning when:**
- Tasks are too complex for single-step execution
- Multiple tools need orchestration
- Steps have dependencies requiring specific ordering
- You need transparency into the execution strategy
- Adaptation based on intermediate results is valuable

**Implementation guidance:**
- Start with **simple linear plans** for predictable workflows
- Add **replanning capabilities** for dynamic environments
- Enable **human oversight** for high-stakes plans
- **Log execution thoroughly** for debugging and improvement
- **Measure plan quality** through execution success rates

**Key Takeaways:**
- 🎯 Planning transforms complex goals into structured, executable steps
- 🧠 Essential for multi-step tasks, workflow automation, and tool orchestration
- 📊 LLMs excel at generating contextually appropriate plans
- ⚡ Adds overhead but enables solving otherwise intractable problems
- 🔄 Dynamic replanning enables adaptation to changing conditions
- 🛠️ Well-supported by frameworks like LangGraph and AutoGen

---

*Planning elevates agents from reactive executors to strategic problem-solvers capable of tackling complex, real-world objectives.*



=======

add thse observations 
====
M5 Agentic AI - Customer Service Agent
1. Introduction
As Andrew explained in the lecture, planning with code execution means letting the LLM write code that becomes the plan itself. Compared to plain-text or JSON-based plans, this approach is more expressive and flexible: the code not only documents the steps but can also execute them directly.

In this lab, you will implement this design pattern in practice.
Instead of asking the LLM to output a plan in JSON format and then manually executing each step, we will allow it to write Python code that directly captures multiple steps of a plan. By executing this code, we can carry out complex queries automatically.

To make things concrete, we simulate a sunglasses store with an inventory of products and a set of transactions (sales, returns, balance updates). This example shows how the LLM can generate code to query or update records, demonstrating the flexibility of this pattern.

1.1 Lab Overview
We will:

Create simple inventory and transaction datasets.
Build a schema block describing the data.
Prompt the LLM to write a plan as Python code (with comments explaining each step).
Execute the code in a sandbox to obtain the answer.
1.2 Learning Outcomes
By the end of this lab, you will be able to:

Explain why letting the model write code (instead of JSON or plain text plans) enables richer, more flexible planning.
Prompt an LLM to produce Python code with step-by-step comments that both documents and executes the plan.
Run the generated code safely in a sandbox and interpret the results.
This illustrates how Code as Action can outperform brittle tool chains and JSON-based planning approaches.

2. Setup
# ==== Imports ====
from __future__ import annotations
import json
from dotenv import load_dotenv
from openai import OpenAI
import re, io, sys, traceback, json
from typing import Any, Dict, Optional
from tinydb import Query, where
​
# Utility modules
import utils      # helper functions for prompting/printing
import inv_utils  # functions for inventory, transactions, schema building, and TinyDB seeding
​
load_dotenv()
client = OpenAI()
In the inv_utils module, we have functions like:

create_inventory() – builds the sunglasses inventory.
create_transactions() – builds the initial transaction log.
seed_db() – loads both inventory and transactions into a JSON-backed store.
build_schema_block() – generates a schema description used in the prompt.
Helpers like get_current_balance() and next_transaction_id() – let the LLM handle consistent updates across inventory and transactions.
2.1 Create Example Tables
We will now create two small tables for the sunglasses store simulation, using TinyDB — a lightweight document-oriented database written in pure Python.
TinyDB stores data as JSON documents and is well-suited for small applications or prototypes, since it requires no server setup and allows you to query and update data easily.

The two tables are:

inventory_tbl: contains product details such as name, item ID, description, quantity in stock, and price.
transactions_tbl: starts with an opening balance and will later track purchases, returns, and adjustments.
You will generate these tables using helper functions in inv_utils, and then preview the first few rows below.

db, inventory_tbl, transactions_tbl = inv_utils.seed_db()
Now, you can inspect the records in each table by printing them as formatted JSON:

utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table")
utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table")
As you can see above, the schemas of each table are as follows:

Inventory Table (inventory_tbl)
item_id (string): Unique product identifier (e.g., SG001).
name (string): Style of sunglasses (e.g., Aviator, Round).
description (string): Text description of the product.
quantity_in_stock (int): Current stock available.
price (float): Price in USD.
Transactions Table (transactions_tbl)
transaction_id (string): Unique identifier (e.g., TXN001).
customer_name (string): Name of the customer, or OPENING_BALANCE for initial entry.
transaction_summary (string): Short description of the transaction.
transaction_amount (float): Amount of money for this transaction.
balance_after_transaction (float): Running balance after applying the transaction.
timestamp (string): ISO-8601 formatted date/time of the transaction.
Planning with Code Execution
2.1. The plan
Once the schema is clear, you’ll build the prompt that instructs the model to plan by writing code and then execute that code. As Andrew emphasized, the code is the plan: the model explains each step in comments, then carries it out. Your prompt below also makes the model self-decide whether the request is read-only or a state change, and it enforces safe execution (no I/O, no network, TinyDB Query only, consistent mutations).

PROMPT = """You are a senior data assistant. PLAN BY WRITING PYTHON CODE USING TINYDB.
​
Database Schema & Samples (read-only):
{schema_block}
​
Execution Environment (already imported/provided):
- Variables: db, inventory_tbl, transactions_tbl  # TinyDB Table objects
- Helpers: get_current_balance(tbl) -> float, next_transaction_id(tbl, prefix="TXN") -> str
- Natural language: user_request: str  # the original user message
​
PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request (shape/keywords, price ranges "under/over/between", stock mentions,
  quantities, buy/return intent). Do NOT hard-code values.
- Build TinyDB queries dynamically with Query(). If a constraint isn't in user_request, don't apply it.
- Be conservative: if intent is ambiguous, do read-only (DRY RUN).
​
TRANSACTION POLICY (hard):
- Do NOT create aggregated multi-item transactions.
- If the request contains multiple items, create a separate transaction row PER ITEM.
- For each item:
  - compute its own line total (unit_price * qty),
  - insert ONE transaction with that amount,
  - update balance sequentially (balance += line_total),
  - update the item’s stock.
- If any requested item lacks sufficient stock, do NOT mutate anything; reply with STATUS="insufficient_stock".
​
HUMAN RESPONSE REQUIREMENT (hard):
- You MUST set a variable named `answer_text` (type str) with a short, customer-friendly sentence (1–2 lines).
- This sentence is the only user-facing message. No dataframes/JSON, no boilerplate disclaimers.
- If nothing matches, politely say so and offer a nearby alternative (closest style/price) or a next step.
​
ACTION POLICY:
- If the request clearly asks to change state (buy/purchase/return/restock/adjust):
    ACTION="mutate"; SHOULD_MUTATE=True; perform the change and write a matching transaction row.
  Otherwise:
    ACTION="read"; SHOULD_MUTATE=False; simulate and explain briefly as a dry run (in logs only).
​
FAILURE & EDGE-CASE HANDLING (must implement):
- Do not capture outer variables in Query.test. Pass them as explicit args.
- Always set a short `answer_text`. Also set a string `STATUS` to one of:
  "success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent".
- no_match: No items satisfy the filters → suggest the closest in style/price, or invite a different range.
- insufficient_stock: Item found but stock < requested qty → state available qty and offer the max you can fulfill.
- invalid_request: Unable to parse essential info (e.g., quantity for a purchase/return) → ask for the missing piece succinctly.
- unsupported_intent: The action is outside the store’s capabilities → provide the nearest supported alternative.
- In all cases, keep the tone helpful and concise (1–2 sentences). Put technical details (e.g., ACTION/DRY RUN) only in stdout logs.
​
OUTPUT CONTRACT:
- Return ONLY executable Python between these tags (no extra text):
  <execute_python>
  # your python
  </execute_python>
​
CODE CHECKLIST (follow in code):
1) Parse intent & constraints from user_request (regex ok).
2) Build TinyDB condition incrementally; query inventory_tbl.
3) If mutate: validate stock, update inventory, insert a transaction (new id, amount, balance, timestamp).
4) ALWAYS set:
   - `answer_text` (human sentence, required),
   - `STATUS` (see list above).
   Also print a brief log to stdout, e.g., "LOG: ACTION=read DRY_RUN=True STATUS=no_match".
5) Optional: set `answer_rows` or `answer_json` if useful, but `answer_text` is mandatory.
​
TONE EXAMPLES (for `answer_text`):
- success: "Yes, we have our Classic sunglasses, a round frame, for $60."
- no_match: "We don’t have round frames under $100 in stock right now, but our Moon round frame is available at $120."
- insufficient_stock: "We only have 1 pair of Classic left; I can reserve that for you."
- invalid_request: "I can help with that—how many pairs would you like to purchase?"
- unsupported_intent: "We can’t refurbish frames, but I can suggest similar new models."
​
Constraints:
- Use TinyDB Query for filtering. Standard library imports only if needed.
- Keep code clear and commented with numbered steps.
​
User request:
{question}
"""
​
2.2 From Prompt to Code (Planning in Code)
Let’s generate code that is the plan.

Instead of asking the model to output a plan in JSON and running it step-by-step with many tiny tools, let’s have it write Python that encodes the whole plan (e.g., “filter this, then compute that, then update this row”). The function generate_llm_code:

Builds a live schema from inventory_tbl and transactions_tbl so the model sees real fields, types, and examples.
Formats the prompt with that schema plus the user’s question.
Calls the model to produce a plan-with-code response — typically an <execute_python>...</execute_python> block whose body contains the step-by-step logic.
Returns the full response (including the plan and the code).
We don’t execute anything in this step.
Why this pattern? Let’s leverage Python/TinyDB as a rich toolbox the model already “knows,” so it can compose multi-step solutions directly in code instead of relying on a growing set of bespoke tools. We’ll extract and run the code in a later step.

# ---------- 1) Code generation ----------
def generate_llm_code(
    prompt: str,
    *,
    inventory_tbl,
    transactions_tbl,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> str:
    """
    Ask the LLM to produce a plan-with-code response.
    Returns the FULL assistant content (including surrounding text and tags).
    The actual code extraction happens later in execute_generated_code.
    """
    schema_block = inv_utils.build_schema_block(inventory_tbl, transactions_tbl)
    prompt = PROMPT.format(schema_block=schema_block, question=prompt)
​
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You write safe, well-commented TinyDB code to handle data questions and updates."
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    
    return content  
2.3 Try a Sample Prompt (Planning-in-Code)
We’ll use the same prompt Andrew used in the lecture:

Prompt: “Do you have any round sunglasses in stock that are under $100?”

Before generating any code, let’s manually inspect the TinyDB tables to see if there are truly round frames (word-only match) and what their prices look like. Run the next cell to preview the inventory and highlight items that match the word-only “round” filter.

Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)
​
# Search the inventory table for documents where either the description OR the name
# contains the word "round" (case-insensitive). The check is done inline:
# - (v or "") ensures we handle None by converting it to an empty string
# - .lower() normalizes case
# - " round " enforces a crude word boundary (won't match "wraparound")
round_sunglasses = inventory_tbl.search(
    (Item.description.test(lambda v: " round " in ((v or "").lower()))) |
    (Item.name.test(        lambda v: " round " in ((v or "").lower())))
)
​
# Render the results as formatted JSON in the notebook UI
utils.print_html(json.dumps(round_sunglasses, indent=2), title="Inventory Status: Round Sunglasses")
Great — we do have round frames available. From our manual inspection, there are two round styles in stock, but only one is under $100. Therefore, the item that satisfies the requirement is:

{
  "item_id": "SG005",
  "name": "Classic",
  "description": "Classic round profile with minimalist metal frames, offering a timeless and versatile style that fits both casual and formal wear.",
  "quantity_in_stock": 10,
  "price": 60
}
Now let’s ask the model to generate a plan in code that answers Andrew’s prompt (no execution yet).

# Andrew's prompt from the lecture
prompt_round = "Do you have any round sunglasses in stock that are under $100?"
​
# Generate the plan-as-code (FULL content; may include <execute_python> tags)
full_content_round = generate_llm_code(
    prompt_round,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1.0,
)
​
# Inspect the LLM’s plan + code (no execution here)
utils.print_html(full_content_round, title="Plan with Code (Full Response)")
2.4. Define the executor function (run a given plan)
Now we’ll define the function that takes a plan produced by the model and runs it safely:

It accepts either the full LLM response (with <execute_python>…</execute_python>) or raw Python code.
It extracts the executable block when needed.
It runs the code in a controlled namespace (TinyDB tables + safe helpers only).
It captures stdout, errors, and the model-set answer variables (answer_text, answer_rows, answer_json).
It renders before/after table snapshots to make side effects explicit.
This is the “executor” that turns a plan-as-code into actions and a concise user-facing answer.

# --- Helper: extract code between <execute_python>...</execute_python> ---
def _extract_execute_block(text: str) -> str:
    """
    Returns the Python code inside <execute_python>...</execute_python>.
    If no tags are found, assumes 'text' is already raw Python code.
    """
    if not text:
        raise RuntimeError("Empty content passed to code executor.")
    m = re.search(r"<execute_python>(.*?)</execute_python>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()
​
​
# ---------- 2) Code execution ----------
def execute_generated_code(
    code_or_content: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute code in a controlled namespace.
    Accepts either raw Python code OR full content with <execute_python> tags.
    Returns minimal artifacts: stdout, error, and extracted answer.
    """
    # Extract code here (now centralized)
    code = _extract_execute_block(code_or_content)
​
    SAFE_GLOBALS = {
        "Query": Query,
        "get_current_balance": inv_utils.get_current_balance,
        "next_transaction_id": inv_utils.next_transaction_id,
        "user_request": user_request or "",
    }
    SAFE_LOCALS = {
        "db": db,
        "inventory_tbl": inventory_tbl,
        "transactions_tbl": transactions_tbl,
    }
​
    # Capture stdout from the executed code
    _stdout_buf, _old_stdout = io.StringIO(), sys.stdout
    sys.stdout = _stdout_buf
    err_text = None
    try:
        exec(code, SAFE_GLOBALS, SAFE_LOCALS)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = _old_stdout
    printed = _stdout_buf.getvalue().strip()
​
    # Extract possible answers set by the generated code
    answer = (
        SAFE_LOCALS.get("answer_text")
        or SAFE_LOCALS.get("answer_rows")
        or SAFE_LOCALS.get("answer_json")
    )
​
​
    return {
        "code": code,            # <- ya sin etiquetas
        "stdout": printed,
        "error": err_text,
        "answer": answer,
        "transactions_tbl": transactions_tbl.all(),  # For inspection
        "inventory_tbl": inventory_tbl.all(),  # For inspection
    }
You’ve checked the shelves and confirmed there’s exactly one round style under $100. Now the fun part: let’s hand the model’s plan-as-code to our executor and watch it do the work. The executor will peel out the ... block, run it in a locked-down sandbox, and then show you everything that matters—what changed in the tables (before/after), any logs the plan printed, and the final, customer-friendly answer_text.

# Execute the generated plan for the round-sunglasses question
result = execute_generated_code(
    full_content_round,          # the full LLM response you generated earlier
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    user_request=prompt_round, # e.g., "Do you have any round sunglasses in stock that are under $100?"
)
​
# Peek at exactly what Python the plan executed
utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
As you can see, this is the expected result based on our previous manual analysis.

2.4 Return Two Aviator Sunglasses
In the previous step we only queried the data, so inventory and transactions were unchanged.
Now let’s handle a return scenario using the planning-in-code pattern:

Request: “Return 2 Aviator sunglasses I bought last week.”

Before generating the plan, let’s inspect the current inventory for the Aviator model.

Item = Query()                    # Create a Query object to reference fields (e.g., Item.name, Item.description)
​
# Query: fetch all inventory rows whose 'name' is exactly "Aviator".
# Notes:
# - This is a case-sensitive equality check. "aviator" won't match.
# - If you need case-insensitive matching, consider a .test(...) or .matches(...) with re.I.
aviators = inventory_tbl.search(
    (Item.name == "Aviator")
)
​
# Display the matched documents in a readable JSON panel
utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses before return")
Inventory confirms one Aviator SKU in stock — SG001 (Aviator): 23 units at $80 each. Now let's generate a plan to answer the prompt:

prompt_aviator = "Return 2 Aviator sunglasses I bought last week."
​
# Generate the plan-as-code (FULL content; may include <execute_python> tags)
full_content_aviator = generate_llm_code(
    prompt_aviator,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1,
)
​
# Inspect the LLM’s plan + code (no execution here)
utils.print_html(full_content_aviator, title="Plan with Code (Full Response)")
Before we execute the plan, let’s check the current status of the transactions.

utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table Before Return")
The transaction log currently shows a single entry — the opening balance (TXN001) for $500.00 recorded at 2025-10-03T09:16:59.628898.

Ready to go—execute the plan by running the cell below.

# Execute the generated plan for the round-sunglasses question
result = execute_generated_code(
    full_content_aviator,          # the full LLM response you generated earlier
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    user_request=prompt_aviator, # e.g., "Return 2 aviator sunglasses I bought last week."
)
​
# Peek at exactly what Python the plan executed
utils.print_html(result["answer"], title="Plan Execution · Extracted Answer")
You can see below that a new transaction has been inserted for the Aviator sunglasses return.

utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table After Return")
And by running the cell below, you’ll see the Aviator stock increase to 25 (quantity_in_stock).

Item = Query()                  
​
aviators = inventory_tbl.search(
    (Item.name == "Aviator")
)
​
utils.print_html(json.dumps(aviators, indent=2), title="Inventory status: Aviator sunglasses after return")
3. Putting It All Together: Customer Service Agent
You’ve built the pieces—schema, prompt, code generator, and executor. Now let’s wire them up into a single helper that takes a natural-language request, generates a plan-as-code, executes it safely, and shows the result (plus before/after tables).

What this agent does

Optionally reseeds the demo data for a clean run.
Generates the plan (Python inside <execute_python>…</execute_python>).
Executes the plan in a controlled namespace (TinyDB + helpers).
Surfaces a concise answer_text and renders before/after snapshots.
def customer_service_agent(
    question: str,
    *,
    db,
    inventory_tbl,
    transactions_tbl,
    model: str = "o4-mini",
    temperature: float = 1.0,
    reseed: bool = False,
) -> dict:
    """
    End-to-end helper:
      1) (Optional) reseed inventory & transactions
      2) Generate plan-as-code from `question`
      3) Execute in a controlled namespace
      4) Render before/after snapshots and return artifacts
​
    Returns:
      {
        "full_content": <raw LLM response (may include <execute_python> tags)>,
        "exec": {
            "code": <extracted python>,
            "stdout": <plan logs>,
            "error": <traceback or None>,
            "answer": <answer_text/rows/json>,
            "inventory_after": [...],
            "transactions_after": [...]
        }
      }
    """
    # 0) Optional reseed
    if reseed:
        inv_utils.create_inventory()
        inv_utils.create_transactions()
​
    # 1) Show the question
    utils.print_html(question, title="User Question")
​
    # 2) Generate plan-as-code (FULL content)
    full_content = generate_llm_code(
        question,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        model=model,
        temperature=temperature,
    )
    utils.print_html(full_content, title="Plan with Code (Full Response)")
​
    # 3) Before snapshots
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · Before")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · Before")
​
    # 4) Execute
    exec_res = execute_generated_code(
        full_content,
        db=db,
        inventory_tbl=inventory_tbl,
        transactions_tbl=transactions_tbl,
        user_request=question,
    )
​
    # 5) After snapshots + final answer
    utils.print_html(exec_res["answer"], title="Plan Execution · Extracted Answer")
    utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory Table · After")
    utils.print_html(json.dumps(transactions_tbl.all(), indent=2), title="Transactions Table · After")
​
    # 6) Return artifacts
    return {
        "full_content": full_content,
        "exec": {
            "code": exec_res["code"],
            "stdout": exec_res["stdout"],
            "error": exec_res["error"],
            "answer": exec_res["answer"],
            "inventory_after": inventory_tbl.all(),
            "transactions_after": transactions_tbl.all(),
        },
    }
​
4. Try It Out (with the Customer Service Agent)
Use the customer_service_agent(...) helper to go from a natural-language request → plan-as-code → safe execution → before/after snapshots.

Try these prompts:

1) Read-only (Andrew’s example):
“Do you have any round sunglasses in stock that are under $100?” 2) Mutation — return:
“Return 2 Aviator sunglasses.” 3) Mutation — purchase:
“Purchase 3 Wayfarer sunglasses for customer Alice.” 4) Mutation - purchase multiple items: "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator."

🔎 What does reseed=True do?

When you call customer_service_agent(..., reseed=True), the agent re-initializes the demo data before running your prompt:
Resets the inventory_tbl to the default product set.
Resets the transactions_tbl to a single opening-balance entry.
Ensures a clean, reproducible run so results aren’t affected by previous tests.
Set reseed=False if you want to preserve the current state and continue from prior operations.
prompt = "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator sunglasses."
​
out = customer_service_agent(
    prompt,
    db=db,
    inventory_tbl=inventory_tbl,
    transactions_tbl=transactions_tbl,
    model="o4-mini",
    temperature=1.0,
    reseed=True,   # set False to keep current state of the inventory and the transactions
)
5. Takeaways
You let code be the plan. Following Andrew’s “code-as-action” idea, you had the model write Python that chains the steps (filter → compute → update) and then you just ran it.

You skipped the brittle tool soup. Instead of piling on tiny tools or JSON plans, you used Python/TinyDB—giving the model a big, familiar toolbox that handles many query shapes with one prompt.

You kept runs safe and visible. You executed in a controlled namespace, captured logs/errors, and reviewed before/after tables—so you always know what changed and why.

🎉 Congratulations!

You just finished the lab and built an agentic customer service workflow. You let the model write code as the plan, ran it safely, and used simple validations to keep updates reliable. When things failed, you surfaced clear, human-readable reasons; when things worked, you saw exactly what changed via before/after snapshots.

With this pattern—planning in code, plus transparent execution—you’re ready to design your own workflows that feel automatic, safe, and easy to extend. 🚀