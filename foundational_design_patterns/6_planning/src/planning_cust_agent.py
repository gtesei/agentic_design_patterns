from __future__ import annotations

import os
import re
import traceback
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import utils_cust_agent  # your helper module

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))

LLM_MODEL = "o4-mini"          # match what you used
LLM_TEMPERATURE = 1.0

llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

PROMPT = """You are a senior data assistant. PLAN BY WRITING PYTHON CODE USING TINYDB.

Database Schema & Samples (read-only):
{schema_block}

Execution Environment (already imported/provided):
- Variables: db, inventory_tbl, transactions_tbl  # TinyDB Table objects
- Helpers: get_current_balance(tbl) -> float, next_transaction_id(tbl, prefix="TXN") -> str
- Natural language: user_request: str  # the original user message

PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request (shape/keywords, price ranges "under/over/between", stock mentions,
  quantities, buy/return intent). Do NOT hard-code values.
- Build TinyDB queries dynamically with Query(). If a constraint isn't in user_request, don't apply it.
- Be conservative: if intent is ambiguous, do read-only (DRY RUN).

TRANSACTION POLICY (hard):
- Do NOT create aggregated multi-item transactions.
- If the request contains multiple items, create a separate transaction row PER ITEM.
- For each item:
  - compute its own line total (unit_price * qty),
  - insert ONE transaction with that amount,
  - update balance sequentially (balance += line_total),
  - update the item’s stock.
- If any requested item lacks sufficient stock, do NOT mutate anything; reply with STATUS="insufficient_stock".

HUMAN RESPONSE REQUIREMENT (hard):
- You MUST set a variable named `answer_text` (type str) with a short, customer-friendly sentence (1–2 lines).
- This sentence is the only user-facing message. No dataframes/JSON, no boilerplate disclaimers.
- If nothing matches, politely say so and offer a nearby alternative (closest style/price) or a next step.

ACTION POLICY:
- If the request clearly asks to change state (buy/purchase/return/restock/adjust):
    ACTION="mutate"; SHOULD_MUTATE=True; perform the change and write a matching transaction row.
  Otherwise:
    ACTION="read"; SHOULD_MUTATE=False; simulate and explain briefly as a dry run (in logs only).

FAILURE & EDGE-CASE HANDLING (must implement):
- Do not capture outer variables in Query.test. Pass them as explicit args.
- Always set a short `answer_text`. Also set a string `STATUS` to one of:
  "success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent".
- no_match: No items satisfy the filters → suggest the closest in style/price, or invite a different range.
- insufficient_stock: Item found but stock < requested qty → state available qty and offer the max you can fulfill.
- invalid_request: Unable to parse essential info (e.g., quantity for a purchase/return) → ask for the missing piece succinctly.
- unsupported_intent: The action is outside the store’s capabilities → provide the nearest supported alternative.
- In all cases, keep the tone helpful and concise (1–2 sentences). Put technical details (e.g., ACTION/DRY RUN) only in stdout logs.

OUTPUT CONTRACT:
- Return ONLY executable Python between these tags (no extra text):
  <execute_python>
  # your python
  </execute_python>

CODE CHECKLIST (follow in code):
1) Parse intent & constraints from user_request (regex ok).
2) Build TinyDB condition incrementally; query inventory_tbl.
3) If mutate: validate stock, update inventory, insert a transaction (new id, amount, balance, timestamp).
4) ALWAYS set:
   - `answer_text` (human sentence, required),
   - `STATUS` (see list above).
   Also print a brief log to stdout, e.g., "LOG: ACTION=read DRY_RUN=True STATUS=no_match".
5) Optional: set `answer_rows` or `answer_json` if useful, but `answer_text` is mandatory.

User request:
{question}
"""

# -----------------------------------------------------------------------------
# Prompt templates + chains
# -----------------------------------------------------------------------------
def build_generation_chain() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You write safe, well-commented TinyDB code to handle data questions and updates."
            ),
            (
                "user",
                PROMPT
            ),
        ]
    )

gen_chain = build_generation_chain() | llm | StrOutputParser()

# Optional “repair” chain if tags are missing.
repair_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You fix formatting. Output ONLY <execute_python>...</execute_python> with valid Python inside."),
        ("user", "Rewrite the following so it strictly matches the required tags and contains only Python code:\n\n{raw}"),
    ]
)
repair_chain = repair_prompt | llm | StrOutputParser()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
EXEC_RE = re.compile(r"<execute_python>\s*(.*?)\s*</execute_python>", re.DOTALL)

def extract_execute_python(raw: str) -> Optional[str]:
    m = EXEC_RE.search(raw or "")
    return m.group(1) if m else None

def generate_llm_code(question: str, inventory_tbl, transactions_tbl) -> str:
    schema_block = utils_cust_agent.build_schema_block(inventory_tbl, transactions_tbl)
    return gen_chain.invoke({"schema_block": schema_block, "question": question})

def generate_and_extract_code(question: str, inventory_tbl, transactions_tbl) -> str:
    raw = generate_llm_code(question, inventory_tbl, transactions_tbl)
    code = extract_execute_python(raw)
    if code:
        return code

    # If model ignored tags, try a repair pass (optional but practical)
    repaired = repair_chain.invoke({"raw": raw})
    code = extract_execute_python(repaired)
    if not code:
        raise ValueError("Model did not return <execute_python>...</execute_python> block.")
    return code

def execute_generated_code(code: str, *, db, inventory_tbl, transactions_tbl, user_request: str) -> dict:
    """
    Executes the generated code in a controlled locals() scope.
    NOTE: This is still arbitrary code execution. Consider sandboxing for production.
    """
    local_vars = {
        "db": db,
        "inventory_tbl": inventory_tbl,
        "transactions_tbl": transactions_tbl,
        "user_request": user_request,
        # helpers provided by your env / utils:
        "get_current_balance": utils_cust_agent.get_current_balance,
        "next_transaction_id": utils_cust_agent.next_transaction_id,
    }

    try:
        exec(code, {}, local_vars)  # noqa: S102 (explicitly acknowledging exec)
    except Exception as e:
        return {
            "STATUS": "invalid_request",
            "answer_text": "Something went wrong while processing that request—can you rephrase it?",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    # Enforce contract
    answer_text = local_vars.get("answer_text")
    status = local_vars.get("STATUS")

    if not isinstance(answer_text, str) or not answer_text.strip():
        answer_text = "I can help with that—could you clarify what you’re looking for?"
    if status not in {"success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent"}:
        status = "invalid_request"

    return {"STATUS": status, "answer_text": answer_text}

# -----------------------------------------------------------------------------
# Example run
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    db, inventory_tbl, transactions_tbl = utils_cust_agent.seed_db()

    utils_cust_agent.print_json_pretty(inventory_tbl.all(), indent=2, title="Inventory Table")
    utils_cust_agent.print_json_pretty(transactions_tbl.all(), indent=2, title="Transactions Table")

    for i in os.environ:
        print(i,os.environ[i])

    question = "Do you have any round sunglasses in stock that are under $100?"

    raw = generate_llm_code(question, inventory_tbl, transactions_tbl)
    print("\n=== Plan with Code (Full Response) ===\n")
    print(raw)

    code = generate_and_extract_code(question, inventory_tbl, transactions_tbl)
    print("\n=== Extracted Code ===\n")
    print(code)

    result = execute_generated_code(code, db=db, inventory_tbl=inventory_tbl, transactions_tbl=transactions_tbl, user_request=question)
    print("\n=== User-Facing Answer ===\n")
    print(result["answer_text"])
