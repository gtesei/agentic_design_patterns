import { describe, expect, test } from "bun:test";
import {
  ExtractedInvoiceSchema,
  RAW_INVOICE,
} from "../src/structured_outputs_basic.ts";
import { BROKEN_INVOICE } from "../src/structured_outputs_advanced.ts";

describe("structured_outputs (TS) — module shape", () => {
  test("exports schemas and sample inputs", () => {
    expect(ExtractedInvoiceSchema.shape.vendor).toBeDefined();
    expect(RAW_INVOICE).toContain("Northwind");
    expect(BROKEN_INVOICE).toContain("TBD");
  });

  test("schema validates a correct payload", () => {
    const parsed = ExtractedInvoiceSchema.parse({
      vendor: "Northwind Cloud Services",
      invoice_id: "INV-9042",
      total_amount_usd: 1248.5,
      due_date: "2026-06-10",
    });
    expect(parsed.vendor).toBe("Northwind Cloud Services");
  });
});
