/**
 * Sequential Prompt Chaining Example — TypeScript port of src/chain_prompt.py.
 *
 * Demonstrates a two-step chain that:
 *   1. Extracts technical specifications from natural language
 *   2. Transforms extracted specs into structured JSON format
 *
 * This pattern is useful for converting unstructured text into structured data
 * through multiple LLM processing steps.
 *
 * Uses LangChain.js LCEL (`pipe(...)`) to mirror the Python pipe operator.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableMap, type Runnable } from "@langchain/core/runnables";

// ============================================================================
// CONFIGURATION
// ============================================================================

const LLM_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";
const LLM_TEMPERATURE = 0;

const llm = new ChatOpenAI({ model: LLM_MODEL, temperature: LLM_TEMPERATURE });

// ============================================================================
// PROMPT DEFINITIONS
// ============================================================================

/**
 * Create a prompt for extracting technical specifications from text.
 */
export function createExtractionPrompt(): ChatPromptTemplate {
  return ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a technical specification extraction expert. " +
        "Extract all hardware specifications from the provided text. " +
        "Focus on CPU, memory (RAM), and storage details. " +
        "Be precise and include units of measurement.",
    ],
    [
      "user",
      "Extract the technical specifications from the following text:\n\n{text_input}",
    ],
  ]);
}

/**
 * Create a prompt for transforming specifications into JSON format.
 */
export function createTransformationPrompt(): ChatPromptTemplate {
  return ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a data formatting expert. Convert technical specifications " +
        "into valid JSON format. Use these exact keys: 'cpu', 'memory', 'storage'. " +
        "Include units in the values (e.g., 'GHz', 'GB', 'TB'). " +
        "Return ONLY the JSON object, no additional text or markdown formatting.",
    ],
    [
      "user",
      "Transform the following specifications into a JSON object:\n\n{specifications}",
    ],
  ]);
}

// ============================================================================
// CHAIN CONSTRUCTION
// ============================================================================

/**
 * Build the specification extraction chain.
 */
export function buildExtractionChain(): Runnable<{ text_input: string }, string> {
  const prompt = createExtractionPrompt();
  return prompt.pipe(llm).pipe(new StringOutputParser());
}

/**
 * Build the complete two-step chain: extraction → transformation.
 *
 * The chain:
 *   1. Extracts technical specifications from input text
 *   2. Transforms extracted specs into structured JSON
 */
export function buildFullChain(): Runnable<{ text_input: string }, string> {
  const extractionChain = buildExtractionChain();
  const transformationPrompt = createTransformationPrompt();

  // Mirrors Python:  {"specifications": extraction_chain} | transformation_prompt | llm | StrOutputParser()
  return RunnableMap.from<{ text_input: string }>({
    specifications: extractionChain,
  })
    .pipe(transformationPrompt)
    .pipe(llm)
    .pipe(new StringOutputParser());
}

// ============================================================================
// EXECUTION FUNCTIONS
// ============================================================================

export type ProcessResult = {
  status: "success" | "error";
  input: string;
  output: string | null;
  error?: string;
};

/**
 * Process input text through the extraction and transformation chain.
 */
export async function processTextToJson(
  text: string,
  verbose = true,
): Promise<ProcessResult> {
  if (verbose) {
    console.log(`\n${"=".repeat(70)}`);
    console.log("Processing Input Text");
    console.log(`${"=".repeat(70)}`);
    console.log(`\nInput:\n${text}\n`);
  }

  try {
    const chain = buildFullChain();
    const result = await chain.invoke({ text_input: text });

    if (verbose) {
      console.log("--- Extracted & Transformed JSON ---");
      console.log(result);
      console.log(`\n${"=".repeat(70)}\n`);
    }

    return { status: "success", input: text, output: result };
  } catch (e: unknown) {
    const errorMsg = `Chain execution failed: ${
      e instanceof Error ? e.message : String(e)
    }`;
    if (verbose) {
      console.log(`\n❌ ERROR: ${errorMsg}\n`);
    }
    return { status: "error", input: text, output: null, error: errorMsg };
  }
}

/**
 * Process multiple input texts in sequence.
 */
export async function processMultipleTexts(
  texts: string[],
): Promise<ProcessResult[]> {
  const chain = buildFullChain();
  const results: ProcessResult[] = [];

  for (let i = 0; i < texts.length; i++) {
    console.log(`\n--- Processing Text ${i + 1}/${texts.length} ---`);
    const text = texts[i]!;
    try {
      const output = await chain.invoke({ text_input: text });
      results.push({ status: "success", input: text, output });
    } catch (e: unknown) {
      results.push({
        status: "error",
        input: text,
        output: null,
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }

  return results;
}

// ============================================================================
// EXAMPLE DATA
// ============================================================================

export const EXAMPLE_TEXTS: string[] = [
  "The new laptop model features a 3.5 GHz octa-core processor, 16GB of RAM, and a 1TB NVMe SSD.",
  "This workstation includes an Intel Core i9-13900K running at 5.8 GHz, 64GB DDR5 memory, and 2TB PCIe 4.0 storage.",
  "Budget desktop: AMD Ryzen 5 5600G at 3.9GHz, 8 gigabytes RAM, 512GB solid state drive.",
];

// ============================================================================
// MAIN EXECUTION
// ============================================================================

async function main(): Promise<void> {
  console.log("\n🚀 Sequential Prompt Chaining Example");
  console.log("=".repeat(70));
  console.log("Extracting specs → Transforming to JSON");
  console.log("=".repeat(70));

  // Single example
  console.log("\n📝 Single Text Example:");
  await processTextToJson(EXAMPLE_TEXTS[0]!);

  // Multiple examples (commented out by default — matches Python)
  // console.log("\n📚 Multiple Texts Example:");
  // const results = await processMultipleTexts(EXAMPLE_TEXTS);
  //
  // const successful = results.filter((r) => r.status === "success").length;
  // console.log(`\n✅ Successfully processed ${successful}/${results.length} texts`);
  //
  // results.forEach((result, i) => {
  //   console.log(`\nResult ${i + 1}:`);
  //   if (result.status === "success") {
  //     console.log(result.output);
  //   } else {
  //     console.log(`❌ Error: ${result.error}`);
  //   }
  // });
}

if (import.meta.main) {
  await main();
}
