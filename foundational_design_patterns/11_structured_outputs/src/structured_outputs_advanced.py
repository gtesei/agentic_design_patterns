"""Structured Outputs (advanced): Instructor retry + malformed input failure mode."""

from __future__ import annotations

import os
import sys

from pathlib import Path

ROOT_DIR = next(parent for parent in Path(__file__).resolve().parents if (parent / "ssl_fix.py").exists())
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example, get_advanced_model

configure_example(__file__)

from pydantic import BaseModel, Field, ValidationError


class ExtractedInvoice(BaseModel):
    vendor: str
    invoice_id: str
    total_amount_usd: float = Field(..., ge=0)
    due_date: str


BROKEN_INVOICE = """
Supplier maybe: ???
Invoice id unclear maybe INV-X
Total: TBD
Due: soon-ish
"""


def instructor_extract(text: str) -> ExtractedInvoice:
    import instructor
    from openai import OpenAI

    client = instructor.from_openai(OpenAI())
    return client.chat.completions.create(
        model=get_advanced_model(),
        response_model=ExtractedInvoice,
        messages=[
            {"role": "system", "content": "Extract invoice fields. If missing, infer conservatively but validly."},
            {"role": "user", "content": text},
        ],
        max_retries=2,
    )


def main() -> None:
    print("\n" + "=" * 80)
    print("STRUCTURED OUTPUTS — ADVANCED (INSTRUCTOR + FAILURE MODE)")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY is required.")
        return

    try:
        extracted = instructor_extract(BROKEN_INVOICE)
        print("\nInstructor extraction output:")
        print(extracted.model_dump())
    except ValidationError as exc:
        print("\nValidation failed even after retries (expected failure mode):")
        print(exc)
    except Exception as exc:
        print("\nExtraction error:")
        print(exc)

    print("\nNote: For self-hosted models, grammar-constrained decoding (Outlines/XGrammar) is recommended.")


if __name__ == "__main__":
    main()
