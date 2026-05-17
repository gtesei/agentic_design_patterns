/**
 * Reflection Pattern — TypeScript port of src/reflection.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";

const llm = new ChatOpenAI({
  temperature: 0,
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
});

export const producerPrompt = ChatPromptTemplate.fromTemplate(`You are an expert Python developer. Write a Python function that calculates the factorial of a number.

Requirements:
- Function name: factorial
- Input: positive integer n
- Output: factorial of n
- Include docstring

Write clean, working code.`);

export const producerChain = producerPrompt.pipe(llm).pipe(new StringOutputParser());

export const criticPrompt = ChatPromptTemplate.fromTemplate(`You are a senior code reviewer. Analyze this Python code and provide specific, actionable feedback.

Code to review:
{draft}

Evaluate based on:
1. **Correctness**: Does it handle all cases correctly?
2. **Edge cases**: Missing validation, error handling
3. **Performance**: Any efficiency concerns?
4. **Code quality**: Readability, documentation, style
5. **Best practices**: Type hints, robust error messages

Provide structured critique with specific suggestions for improvement.`);

export const criticChain = criticPrompt.pipe(llm).pipe(new StringOutputParser());

export const refinePrompt = ChatPromptTemplate.fromTemplate(`You are an expert Python developer. Improve the following code based on the critique provided.

Original Code:
{draft}

Critique:
{critique}

Generate an improved version that addresses all feedback. Maintain working functionality while incorporating suggested improvements.`);

export const refineChain = refinePrompt.pipe(llm).pipe(new StringOutputParser());

export const reflectionChain = RunnablePassthrough.assign({
  draft: async () => await producerChain.invoke({}),
})
  .assign({
    critique: async (x: { draft: string }) => await criticChain.invoke({ draft: x.draft }),
  })
  .pipe(refineChain);

async function main(): Promise<void> {
  console.log("=".repeat(80));
  console.log("REFLECTION PATTERN: Factorial Function Generation");
  console.log("=".repeat(80));

  console.log("\n--- Step 1: Producer - Initial Code Generation ---");
  const initialDraft = await producerChain.invoke({});
  console.log(initialDraft);

  console.log("\n--- Step 2: Critic - Code Review ---");
  const critique = await criticChain.invoke({ draft: initialDraft });
  console.log(critique);

  console.log("\n--- Step 3: Refinement - Improved Code ---");
  const refinedCode = await refineChain.invoke({
    draft: initialDraft,
    critique,
  });
  console.log(refinedCode);

  console.log("\n" + "=".repeat(80));
  console.log("SINGLE-CHAIN EXECUTION (Automated Reflection)");
  console.log("=".repeat(80));
  const finalResult = await reflectionChain.invoke({});
  console.log(finalResult);
}

if (import.meta.main) {
  await main();
}
