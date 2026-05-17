/**
 * ReAct Pattern: Basic Implementation — TypeScript port of src/react_agent.py.
 */

import { createAgent } from "langchain";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  temperature: 0,
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
});

export const search = tool(
  async ({ query }: { query: string }) => {
    const knowledgeBase: Record<string, string> = {
      python:
        "Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
      "tokyo population":
        "Tokyo is the capital of Japan with a population of approximately 14 million people in the city proper (as of 2024), and about 37.4 million in the Greater Tokyo Area, making it the most populous metropolitan area in the world.",
      "eiffel tower":
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was designed by Gustave Eiffel and completed in 1889. It stands 330 meters (1,083 feet) tall and was the world's tallest structure until 1930.",
      photosynthesis:
        "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. The process uses carbon dioxide and water to produce glucose and oxygen, with the chemical equation: 6CO2 + 6H2O + light → C6H12O6 + 6O2.",
      "marie curie":
        "Marie Curie (1867-1934) was a Polish-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win Nobel Prizes in two scientific fields (Physics in 1903 and Chemistry in 1911).",
      "great wall china":
        "The Great Wall of China is a series of fortifications built across northern China. Construction began in the 7th century BC, with major construction during the Ming Dynasty (1368-1644). The wall stretches approximately 21,196 kilometers (13,171 miles).",
      "black holes":
        "Black holes are regions of spacetime with gravitational fields so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycles. The boundary of a black hole is called the event horizon.",
      shakespeare:
        "William Shakespeare (1564-1616) was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language. He wrote approximately 39 plays, 154 sonnets, and several poems. Famous works include Hamlet, Romeo and Juliet, and Macbeth.",
    };

    const queryLower = query.toLowerCase();
    for (const [topic, info] of Object.entries(knowledgeBase)) {
      if (
        queryLower.includes(topic) ||
        topic.split(" ").some((word) => queryLower.includes(word))
      ) {
        return `Search results for '${query}':\n\n${info}`;
      }
    }
    return `Search results for '${query}':\n\nNo specific information found in the knowledge base. In a production system, this would query real search engines or APIs.`;
  },
  {
    name: "search",
    description:
      "Search for information on the web or in a knowledge base. Use this when you need to find facts, current information, or data that you don't already know.",
  },
);

export const calculator = tool(
  async ({ expression }: { expression: string }) => {
    try {
      const result = Function(`"use strict"; return (${expression});`)();
      return `Calculation result: ${expression} = ${result}`;
    } catch (e: unknown) {
      return `Error calculating '${expression}': ${
        e instanceof Error ? e.message : String(e)
      }`;
    }
  },
  {
    name: "calculator",
    description:
      "Perform mathematical calculations. Use this when you need to compute numerical results, solve equations, or perform any mathematical operations.",
  },
);

export const getWordCount = tool(
  async ({ text }: { text: string }) => {
    const words = text.split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const charCount = text.length;
    const sentenceCount =
      (text.match(/\./g)?.length ?? 0) +
      (text.match(/!/g)?.length ?? 0) +
      (text.match(/\?/g)?.length ?? 0);
    return `Text analysis:\n- Words: ${wordCount}\n- Characters: ${charCount}\n- Sentences: ~${sentenceCount}`;
  },
  {
    name: "get_word_count",
    description:
      "Count the number of words in a given text. Use this when you need to analyze text length or word frequency.",
  },
);

const tools = [search, calculator, getWordCount];
export const agent = createAgent({ model: llm, tools });

export async function runExample(query: string): Promise<void> {
  console.log(`\n${"=".repeat(80)}`);
  console.log(`Query: ${query}`);
  console.log(`${"=".repeat(80)}\n`);

  const result = await agent.invoke({ messages: [{ role: "user", content: query }] });
  const messages = (result as { messages: any[] }).messages ?? [];

  for (const message of messages) {
    if (message?.content) {
      const type = String(message?.type ?? "message").toUpperCase();
      console.log(`${type}: ${JSON.stringify(message.content)}`);
    }
    if (message?.tool_calls) {
      for (const toolCall of message.tool_calls) {
        console.log(`\nTOOL CALL: ${toolCall.name}`);
        console.log(`Arguments: ${JSON.stringify(toolCall.args)}`);
      }
    }
  }

  console.log(`\n${"=".repeat(80)}\n`);
}

async function main(): Promise<void> {
  console.log(`
    ╔═══════════════════════════════════════════════════════════════╗
    ║         ReAct Pattern - Basic Implementation                  ║
    ║                                                               ║
    ║  The agent will Reason, Act, and Observe to answer queries    ║
    ╚═══════════════════════════════════════════════════════════════╝
    `);

  await runExample(
    "What is photosynthesis and how many oxygen molecules does it produce per glucose molecule?",
  );
  await runExample(
    "If Tokyo has a population of 14 million and the Eiffel Tower is 330 meters tall, what is the sum of Tokyo's population plus the Eiffel Tower's height in meters?",
  );
  await runExample(
    "Tell me about Marie Curie and count how many words are in her description.",
  );

  console.log(`
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Examples Complete!                         ║
    ║                                                               ║
    ║  The ReAct agent demonstrated:                                ║
    ║  • Reasoning about what information was needed                ║
    ║  • Using tools to gather facts and perform calculations       ║
    ║  • Observing results and adapting its approach                ║
    ║  • Combining multiple tools to solve complex queries          ║
    ╚═══════════════════════════════════════════════════════════════╝
    `);
}

if (import.meta.main) {
  await main();
}
