/**
 * Routing Pattern — TypeScript port of src/routing.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableBranch, RunnableMap } from "@langchain/core/runnables";

const llm = new ChatOpenAI({
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
  temperature: 0,
});

// ============================================================================
// HANDLER FUNCTIONS
// ============================================================================

export function bookingHandler(request: string): string {
  console.log("\n--- DELEGATING TO BOOKING HANDLER ---");
  return (
    `Booking Handler processed request: '${request}'.\n` +
    "Result: Simulated booking action."
  );
}

export function infoHandler(request: string): string {
  console.log("\n--- DELEGATING TO INFO HANDLER ---");
  return (
    `Info Handler processed request: '${request}'.\n` +
    "Result: Simulated information retrieval."
  );
}

export function unclearHandler(request: string): string {
  console.log("\n--- HANDLING UNCLEAR REQUEST ---");
  return (
    `Coordinator could not delegate request: '${request}'.\n` +
    "Please clarify your request."
  );
}

// ============================================================================
// COORDINATOR ROUTER CHAIN
// ============================================================================

export const coordinatorRouterPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `Analyze the user's request and determine which specialist handler should process it.

        - If the request is related to booking flights or hotels, output 'booker'.
        - For general information questions, output 'info'.
        - If the request is unclear or doesn't fit either category, output 'unclear'.

        ONLY output one word: 'booker', 'info', or 'unclear'.`,
  ],
  ["user", "{request}"],
]);

export const coordinatorRouterChain = coordinatorRouterPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

// ============================================================================
// DELEGATION LOGIC
// ============================================================================

export function extractRequest(x: { request?: string }): string {
  return x.request ?? "";
}

type RouteInput = {
  decision: string;
  request: string;
};

const branches = {
  booker: (x: RouteInput) => ({
    ...x,
    output: bookingHandler(extractRequest(x)),
  }),
  info: (x: RouteInput) => ({
    ...x,
    output: infoHandler(extractRequest(x)),
  }),
  unclear: (x: RouteInput) => ({
    ...x,
    output: unclearHandler(extractRequest(x)),
  }),
};

export const delegationBranch = RunnableBranch.from([
  [
    (x: RouteInput) => x.decision.trim().toLowerCase() === "booker",
    branches.booker,
  ],
  [
    (x: RouteInput) => x.decision.trim().toLowerCase() === "info",
    branches.info,
  ],
  branches.unclear,
]);

export const coordinatorAgent = RunnableMap.from<{
  request: string;
}>({
  decision: coordinatorRouterChain,
  request: (x: { request: string }) => x.request,
})
  .pipe(delegationBranch)
  .pipe((x: { output: string }) => x.output);

// ============================================================================
// MAIN EXECUTION
// ============================================================================

export async function runExample(
  description: string,
  request: string,
): Promise<void> {
  console.log(`\n${"=".repeat(70)}`);
  console.log(`--- ${description} ---`);
  console.log(`${"=".repeat(70)}`);
  const result = await coordinatorAgent.invoke({ request });
  console.log(`\nFinal Result: ${result}`);
}

async function main(): Promise<void> {
  await runExample("Running with a booking request", "Book me a flight to London.");
  await runExample("Running with an info request", "What is the capital of Italy?");
  await runExample("Running with an unclear request", "Tell me about quantum physics.");
}

if (import.meta.main) {
  await main();
}
