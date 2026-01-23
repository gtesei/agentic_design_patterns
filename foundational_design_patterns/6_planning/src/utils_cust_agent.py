# =========================
# Imports
# =========================
from typing import Any
from datetime import datetime
import random
import json
import copy

import pandas as pd
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


# =========================
# TinyDB Initialization
# =========================
DB_PATH = "store_db.json"

db = TinyDB(DB_PATH)
inventory_table = db.table("inventory")
transactions_table = db.table("transactions")


# =========================
# Inventory & Transactions (TinyDB)
# =========================
def create_inventory():
    """
    Create and store the initial sunglasses inventory in TinyDB.
    """
    random.seed(42)

    inventory = [
        {
            "item_id": "SG001",
            "name": "Aviator",
            "description": (
                "Originally designed for pilots, these teardrop-shaped lenses with thin metal frames "
                "offer timeless appeal."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 80,
        },
        {
            "item_id": "SG002",
            "name": "Wayfarer",
            "description": (
                "Featuring thick, angular frames that combine retro charm with modern edge."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 95,
        },
        {
            "item_id": "SG003",
            "name": "Mystique",
            "description": (
                "Inspired by 1950s glamour, with elegant upward-sweeping frames."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 70,
        },
        {
            "item_id": "SG004",
            "name": "Sport",
            "description": (
                "Wraparound sunglasses designed for active lifestyles."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 110,
        },
        {
            "item_id": "SG005",
            "name": "Classic",
            "description": (
                "Classic round profile with minimalist metal frames."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 60,
        },
        {
            "item_id": "SG006",
            "name": "Moon",
            "description": (
                "Oversized round style with bold plastic frames."
            ),
            "quantity_in_stock": random.randint(3, 25),
            "price": 120,
        },
    ]

    inventory_table.truncate()
    inventory_table.insert_multiple(inventory)
    return inventory


def create_transactions(opening_balance: float = 500.00):
    """
    Create initial transaction log with opening balance.
    """
    opening_txn = {
        "transaction_id": "TXN001",
        "customer_name": "OPENING_BALANCE",
        "transaction_summary": "Daily opening register balance",
        "transaction_amount": opening_balance,
        "balance_after_transaction": opening_balance,
        "timestamp": datetime.now().isoformat(),
    }

    transactions_table.truncate()
    transactions_table.insert(opening_txn)
    return opening_txn


def seed_db(db_path: str = DB_PATH):
    """
    Initialize and seed TinyDB with inventory and transactions.
    """
    db = TinyDB(db_path)
    inventory_tbl = db.table("inventory")
    transactions_tbl = db.table("transactions")

    create_inventory()
    create_transactions()

    return db, inventory_tbl, transactions_tbl


# =========================
# Schema Helpers
# =========================
def infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


def build_schema_for_table(tbl, table_name: str, sample_size: int = 3) -> str:
    rows = tbl.all()
    if not rows:
        return f"TABLE: {table_name} (empty)"

    schema = {}
    for row in rows:
        for key, value in row.items():
            schema.setdefault(key, {"type": type(value).__name__, "examples": []})
            if len(schema[key]["examples"]) < sample_size:
                schema[key]["examples"].append(str(value))

    lines = [f"TABLE: {table_name}", "COLUMNS:"]
    for col, info in schema.items():
        lines.append(f"  - {col}: {info['type']} | examples: {info['examples']}")

    lines.append(f"ROWS: {len(rows)}")
    lines.append(f"PREVIEW (first 3 rows): {rows[:3]}")
    return "\n".join(lines)


def build_schema_block(inventory_tbl, transactions_tbl) -> str:
    notes = (
        "NOTES:\n"
        "- Prices are in USD.\n"
        "- quantity_in_stock > 0 means available.\n"
        "- timestamps are ISO-8601.\n"
    )

    return "\n\n".join(
        [
            build_schema_for_table(inventory_tbl, "inventory_tbl"),
            build_schema_for_table(transactions_tbl, "transactions_tbl"),
            notes,
        ]
    )


# =========================
# Transaction Helpers
# =========================
def get_current_balance(transactions_tbl, default: float = 0.0) -> float:
    txns = transactions_tbl.all()
    return txns[-1]["balance_after_transaction"] if txns else default


def next_transaction_id(transactions_tbl, prefix: str = "TXN") -> str:
    return f"{prefix}{len(transactions_tbl) + 1:03d}"


# =========================
# Pandas Inventory & Ledger
# =========================
def create_inventory_dataframe() -> pd.DataFrame:
    random.seed(42)

    data = {
        "name": ["Aviator", "Wayfarer", "Mystique", "Sport", "Round"],
        "item_id": ["SG001", "SG002", "SG003", "SG004", "SG005"],
        "description": [
            "Teardrop-shaped lenses for pilots.",
            "Thick angular frames.",
            "Elegant vintage style.",
            "Wraparound sports frames.",
            "Minimalist round lenses.",
        ],
        "quantity_in_stock": [random.randint(3, 25) for _ in range(5)],
        "price": [random.randint(75, 150) for _ in range(5)],
    }

    return pd.DataFrame(data)


def create_transaction_dataframe(opening_balance: float = 500.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": ["TXN001"],
            "customer_name": ["OPENING_BALANCE"],
            "transaction_summary": ["Daily opening register balance"],
            "transaction_amount": [opening_balance],
            "balance_after_transaction": [opening_balance],
        }
    )


def create_ledger_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["transaction_date", "item_id", "quantity", "transaction_type"]
    )


# =========================
# Inventory Operations (Pandas)
# =========================
def get_formatted_item_names(df: pd.DataFrame) -> list[str]:
    return df["name"].tolist()


def check_inventory_by_name(df: pd.DataFrame, item_name: str) -> int:
    matches = df[df["name"].str.lower() == item_name.lower()]
    return -1 if matches.empty else int(matches.iloc[0]["quantity_in_stock"])


def update_stock(df: pd.DataFrame, item_name: str, transaction_type: str, quantity: int) -> bool:
    if quantity <= 0 or transaction_type.lower() not in {"sale", "return"}:
        return False

    mask = df["name"].str.lower() == item_name.lower()
    if not mask.any():
        return False

    delta = -quantity if transaction_type == "sale" else quantity
    df.loc[mask, "quantity_in_stock"] += delta
    df.loc[mask, "quantity_in_stock"] = df.loc[mask, "quantity_in_stock"].clip(lower=0)
    return True


# =========================
# Plan Execution
# =========================
def execute_step(step: dict, inventory_df: pd.DataFrame, functions: dict):
    args = copy.deepcopy(step["args"])
    if args.get("df") == "inventory_df":
        args["df"] = inventory_df
    return functions[step["task"]](**args)


def execute_plan(plan: list, inventory_df: pd.DataFrame, functions: dict):
    results = []
    for step in plan:
        result = execute_step(step, inventory_df, functions)
        results.append(result)
        print(f"Executed {step['task']}: {result}")
    return results

def print_json_pretty(obj, indent: int = 2, title: str | None = None):
    """Pretty-print a Python object as JSON to the terminal."""
    if title:
        print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=indent, ensure_ascii=False))