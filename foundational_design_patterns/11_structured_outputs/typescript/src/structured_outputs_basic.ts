/**
 * Structured Outputs (basic) — TypeScript port of src/structured_outputs_basic.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

export const ExtractedInvoiceSchema = z.object({
  vendor: z.string().describe("Vendor name"),
  invoice_id: z.string().describe("Invoice identifier"),
  total_amount_usd: z.number().min(0),
  due_date: z.string().describe("ISO date string"),
});

export type ExtractedInvoice = z.infer<typeof ExtractedInvoiceSchema>;

export const RAW_INVOICE = `
Invoice Notice
Vendor: Northwind Cloud Services
Invoice Number: INV-9042
Amount Due: USD 1,248.50
Payment Due Date: 2026-06-10
`;

export async function naiveExtract(
  llm: ChatOpenAI,
  text: string,
): Promise<Record<string, unknown>> {
  const prompt = `
Extract invoice fields from this text and return JSON with keys:
vendor, invoice_id, total_amount_usd, due_date

${text}
`;
  const rawResponse = await llm.invoke(prompt);
  const raw = String(rawResponse.content);

  const vendorMatch = raw.match(/"vendor"\s*:\s*"([^"]+)"/);
  const invoiceMatch = raw.match(/"invoice_id"\s*:\s*"([^"]+)"/);
  const amountMatch = raw.match(/"total_amount_usd"\s*:\s*([0-9.]+)/);
  const dateMatch = raw.match(/"due_date"\s*:\s*"([^"]+)"/);

  return {
    raw_model_output: raw,
    vendor: vendorMatch?.[1] ?? null,
    invoice_id: invoiceMatch?.[1] ?? null,
    total_amount_usd: amountMatch?.[1] ? Number(amountMatch[1]) : null,
    due_date: dateMatch?.[1] ?? null,
  };
}

export async function strictExtract(
  llm: ChatOpenAI,
  text: string,
): Promise<ExtractedInvoice | null> {
  const structured = llm.withStructuredOutput(ExtractedInvoiceSchema);
  return await structured.invoke(`Extract invoice data from:\n\n${text}`);
}

async function main(): Promise<void> {
  console.log("\n" + "=".repeat(80));
  console.log("STRUCTURED OUTPUTS — BASIC");
  console.log("=".repeat(80));

  if (!process.env.OPENAI_API_KEY) {
    console.log("❌ OPENAI_API_KEY is required.");
    return;
  }

  const llm = new ChatOpenAI({
    model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
    temperature: 0,
  });

  try {
    const naive = await naiveExtract(llm, RAW_INVOICE);
    console.log("\nNaive extraction result:");
    console.log(naive);

    const strict = await strictExtract(llm, RAW_INVOICE);
    console.log("\nSchema-enforced extraction result:");
    console.log(strict ?? null);
  } catch (exc: unknown) {
    console.log("\n⚠️ Extraction failed due to connectivity/provider issue:");
    console.log(exc);
    console.log("Tip: AGENTIC_DISABLE_SSL=1 bash run.sh");
  }
}

if (import.meta.main) {
  await main();
}
