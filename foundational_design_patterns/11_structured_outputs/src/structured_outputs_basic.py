"""Structured Outputs (basic): naive extraction vs schema-enforced extraction."""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_default_model

configure_example(__file__)

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ExtractedInvoice(BaseModel):
    vendor: str = Field(..., description="Vendor name")
    invoice_id: str = Field(..., description="Invoice identifier")
    total_amount_usd: float = Field(..., ge=0)
    due_date: str = Field(..., description="ISO date string")


RAW_INVOICE = """
Invoice Notice
Vendor: Northwind Cloud Services
Invoice Number: INV-9042
Amount Due: USD 1,248.50
Payment Due Date: 2026-06-10
"""


def naive_extract(llm: ChatOpenAI, text: str) -> dict:
    prompt = f"""
Extract invoice fields from this text and return JSON with keys:
vendor, invoice_id, total_amount_usd, due_date

{text}
"""
    raw = llm.invoke(prompt).content

    # intentionally brittle parser to show failures
    vendor_match = re.search(r'"vendor"\s*:\s*"([^"]+)"', raw)
    invoice_match = re.search(r'"invoice_id"\s*:\s*"([^"]+)"', raw)
    amount_match = re.search(r'"total_amount_usd"\s*:\s*([0-9.]+)', raw)
    date_match = re.search(r'"due_date"\s*:\s*"([^"]+)"', raw)

    return {
        "raw_model_output": raw,
        "vendor": vendor_match.group(1) if vendor_match else None,
        "invoice_id": invoice_match.group(1) if invoice_match else None,
        "total_amount_usd": float(amount_match.group(1)) if amount_match else None,
        "due_date": date_match.group(1) if date_match else None,
    }


def strict_extract(llm: ChatOpenAI, text: str) -> Optional[ExtractedInvoice]:
    structured = llm.with_structured_output(ExtractedInvoice)
    return structured.invoke(f"Extract invoice data from:\n\n{text}")


def main() -> None:
    print("\n" + "=" * 80)
    print("STRUCTURED OUTPUTS — BASIC")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    llm = ChatOpenAI(model=get_default_model(), temperature=0)

    try:
        naive = naive_extract(llm, RAW_INVOICE)
        print("\nNaive extraction result:")
        print(naive)

        strict = strict_extract(llm, RAW_INVOICE)
        print("\nSchema-enforced extraction result:")
        print(strict.model_dump() if strict else None)
    except Exception as exc:
        print("\n⚠️ Extraction failed due to connectivity/provider issue:")
        print(exc)
        print("Tip: AGENTIC_DISABLE_SSL=1 bash run.sh")


if __name__ == "__main__":
    main()
