/**
 * Structured Outputs (advanced) — TypeScript port of src/structured_outputs_advanced.py.
 */

import OpenAI from "openai";
import { generateObject } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";

export const ExtractedInvoiceSchema = z.object({
  vendor: z.string(),
  invoice_id: z.string(),
  total_amount_usd: z.number().min(0),
  due_date: z.string(),
});

export type ExtractedInvoice = z.infer<typeof ExtractedInvoiceSchema>;

export const BROKEN_INVOICE = `
Supplier maybe: ???
Invoice id unclear maybe INV-X
Total: TBD
Due: soon-ish
`;

export async function instructorExtract(
  text: string,
): Promise<ExtractedInvoice> {
  const model = process.env.OPENAI_ADVANCED_MODEL ?? process.env.OPENAI_MODEL ?? "gpt-5.2";
  let lastError: unknown;

  // Mirrors max_retries=2 in spirit: initial try + 2 retries.
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const result = await generateObject({
        model: openai(model),
        schema: ExtractedInvoiceSchema,
        system:
          "Extract invoice fields. If missing, infer conservatively but validly.",
        prompt: text,
      });
      return result.object;
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError;
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("STRUCTURED OUTPUTS — ADVANCED (INSTRUCTOR + FAILURE MODE)");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  try {
    const extracted = await instructorExtract(BROKEN_INVOICE);
    console.log("\nInstructor extraction output:");
    console.log(extracted);
  } catch (exc: unknown) {
    console.log("\nValidation failed even after retries (expected failure mode):");
    console.log(exc);
  }

  console.log(
    "\nNote: For self-hosted models, grammar-constrained decoding (Outlines/XGrammar) is recommended.",
  );
}

if (import.meta.main) {
  // Touch OpenAI client import path so the runtime mirrors the Python example's
  // explicit OpenAI/instructor dependency surface.
  void OpenAI;
  await main();
}
