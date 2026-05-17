/**
 * ReAct Pattern: Advanced Implementation — TypeScript port of src/react_agent_advanced.py.
 */

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, END, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

const llm = new ChatOpenAI({
  temperature: 0,
  model: process.env.OPENAI_MODEL ?? "gpt-4o-mini",
});

export const wikipediaSearch = tool(
  async ({ query }: { query: string }) => {
    const wikiArticles: Record<string, string> = {
      "albert einstein":
        "Albert Einstein (1879-1955) was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is known for its influence on the philosophy of science. He is best known to the general public for his mass–energy equivalence formula E = mc². He received the 1921 Nobel Prize in Physics for his services to theoretical physics, especially for his discovery of the law of the photoelectric effect.",
      "great barrier reef":
        "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and 900 islands stretching for over 2,300 kilometres (1,400 mi) over an area of approximately 344,400 square kilometres (133,000 sq mi). The reef is located in the Coral Sea, off the coast of Queensland, Australia. It was selected as a World Heritage Site in 1981.",
      "french revolution":
        "The French Revolution was a period of far-reaching social and political upheaval in France that lasted from 1789 until 1799. It led to the fall of the monarchy, the rise of Napoleon Bonaparte, and significant social and political changes. Major events include the Storming of the Bastille (July 14, 1789), the Declaration of the Rights of Man and of the Citizen, and the Reign of Terror.",
      "quantum mechanics":
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It differs from classical physics primarily at the quantum realm of atomic and subatomic length scales. Quantum mechanics provides a mathematical description of much of the dual particle-like and wave-like behavior and interactions of energy and matter.",
      "amazon rainforest":
        "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km² (2,700,000 sq mi), of which 5,500,000 km² (2,100,000 sq mi) are covered by the rainforest. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.",
    };

    const queryLower = query.toLowerCase();
    for (const [topic, article] of Object.entries(wikiArticles)) {
      if (
        queryLower.includes(topic) ||
        topic.split(" ").some((word) => queryLower.includes(word))
      ) {
        return `Wikipedia article on '${topic.replace(/\b\w/g, (x) => x.toUpperCase())}':\n\n${article}`;
      }
    }
    return `No Wikipedia article found for '${query}'. Try a different search term.`;
  },
  {
    name: "wikipedia_search",
    description:
      "Search Wikipedia for information on a topic. Use this when you need authoritative, factual information about historical figures, scientific concepts, places, or events.",
  },
);

export const scientificCalculator = tool(
  async ({
    operation,
    a,
    b,
  }: {
    operation: string;
    a: number;
    b?: number;
  }) => {
    try {
      if (operation === "add") return `${a} + ${b} = ${a + (b ?? 0)}`;
      if (operation === "subtract") return `${a} - ${b} = ${a - (b ?? 0)}`;
      if (operation === "multiply") return `${a} × ${b} = ${a * (b ?? 0)}`;
      if (operation === "divide") {
        if (b === 0) return "Error: Division by zero";
        return `${a} ÷ ${b} = ${a / (b ?? 1)}`;
      }
      if (operation === "power") return `${a}^${b} = ${a ** (b ?? 1)}`;
      if (operation === "sqrt") {
        if (a < 0) return "Error: Cannot take square root of negative number";
        return `√${a} = ${a ** 0.5}`;
      }
      if (operation === "percent") return `${a}% of ${b} = ${(a / 100) * (b ?? 0)}`;
      return `Unknown operation: ${operation}`;
    } catch (e: unknown) {
      return `Calculation error: ${e instanceof Error ? e.message : String(e)}`;
    }
  },
  {
    name: "scientific_calculator",
    description:
      "Perform scientific calculations. Use this for mathematical operations including basic arithmetic, exponents, square roots, and more.",
  },
);

export const textAnalyzer = tool(
  async ({
    text,
    analysis_type = "full",
  }: {
    text: string;
    analysis_type?: string;
  }) => {
    const words = text.split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const charCount = text.length;
    const charCountNoSpaces = text.replaceAll(" ", "").length;
    const sentenceCount = Math.max(
      1,
      (text.match(/\./g)?.length ?? 0) +
        (text.match(/!/g)?.length ?? 0) +
        (text.match(/\?/g)?.length ?? 0),
    );

    if (analysis_type === "words") return `Word count: ${wordCount}`;
    if (analysis_type === "chars") {
      return `Character count: ${charCount} (without spaces: ${charCountNoSpaces})`;
    }
    if (analysis_type === "sentences") return `Sentence count: ${sentenceCount}`;

    const avgWordLength = wordCount > 0 ? charCountNoSpaces / wordCount : 0;
    const avgSentenceLength = sentenceCount > 0 ? wordCount / sentenceCount : 0;
    return `Text Analysis:
        - Total words: ${wordCount}
        - Total characters: ${charCount}
        - Characters (no spaces): ${charCountNoSpaces}
        - Sentences: ${sentenceCount}
        - Average word length: ${avgWordLength.toFixed(1)} characters
        - Average sentence length: ${avgSentenceLength.toFixed(1)} words`;
  },
  {
    name: "text_analyzer",
    description: "Analyze text for various metrics. Use this to extract statistics and insights from text.",
  },
);

const tools = [wikipediaSearch, scientificCalculator, textAnalyzer];
const toolNode = new ToolNode(tools);
const llmWithTools = llm.bindTools(tools);

export const REACT_SYSTEM_PROMPT = `You are a helpful research assistant using the ReAct (Reasoning and Acting) framework.

For each task, follow this explicit pattern:

1. **Thought**: Clearly state what you're thinking about the task
   - What do I know so far?
   - What information do I still need?
   - What should I do next?

2. **Action**: If you need more information, use the appropriate tool
   - Choose the right tool for the job
   - Provide clear parameters

3. **Observation**: After receiving tool results, state what you learned
   - What did this tell me?
   - Is it sufficient to answer the question?
   - Do I need more information?

4. **Repeat**: Continue the thought-action-observation cycle until you have enough information

5. **Answer**: When you have all necessary information, provide a complete, accurate final answer

Be systematic, thorough, and show your reasoning process clearly.`;

export const ReActState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
  iteration: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0,
  }),
  max_iterations: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 10,
  }),
});

type State = typeof ReActState.State;

export async function agentNode(state: State): Promise<Partial<State>> {
  let messages = state.messages;
  if (state.iteration === 0) {
    messages = [new SystemMessage(REACT_SYSTEM_PROMPT), ...messages];
  }

  const remaining = state.max_iterations - state.iteration;
  if (remaining <= 0) {
    return {
      messages: [new AIMessage("Maximum iterations reached. Unable to complete task.")],
      iteration: state.iteration + 1,
    };
  }

  if (state.iteration > 0) {
    messages = messages.concat([
      new SystemMessage(
        `[Iteration ${state.iteration}/${state.max_iterations}] Continue reasoning and decide if you need more information.`,
      ),
    ]);
  }

  const response = await llmWithTools.invoke(messages);
  return {
    messages: [response],
    iteration: state.iteration + 1,
  };
}

export function shouldContinue(state: State): "tools" | typeof END {
  const lastMessage = state.messages[state.messages.length - 1];
  if (state.iteration >= state.max_iterations) return END;
  if (lastMessage instanceof AIMessage && lastMessage.tool_calls?.length) {
    return "tools";
  }
  return END;
}

export function buildGraph() {
  return new StateGraph(ReActState)
    .addNode("agent", agentNode)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue, {
      tools: "tools",
      [END]: END,
    })
    .addEdge("tools", "agent")
    .compile();
}

export function displayReActTrace(result: {
  messages: BaseMessage[];
  iteration?: number;
}): void {
  console.log("\n" + "=".repeat(100));
  console.log("REACT REASONING TRACE");
  console.log("=".repeat(100) + "\n");

  let iteration = 0;
  for (const message of result.messages) {
    if (message instanceof SystemMessage) continue;

    if (message instanceof HumanMessage) {
      console.log("┌─ USER QUERY");
      console.log(`│  ${message.content}`);
      console.log(`└─${"─".repeat(80)}\n`);
    } else if (message instanceof AIMessage) {
      if (message.tool_calls?.length) {
        iteration += 1;
        console.log(`┌─ ITERATION ${iteration}: THOUGHT + ACTION`);
        console.log("│");
        if (message.content) {
          console.log(`│  💭 Thought: ${String(message.content)}`);
          console.log("│");
        }
        console.log("│  🔧 Action: Using tools...");
        for (const toolCall of message.tool_calls) {
          console.log(`│     - Tool: ${toolCall.name}`);
          console.log(`│     - Args: ${JSON.stringify(toolCall.args)}`);
        }
        console.log(`└─${"─".repeat(80)}\n`);
      } else {
        console.log("┌─ FINAL ANSWER");
        console.log("│");
        console.log(`│  ✓ ${String(message.content)}`);
        console.log(`└─${"─".repeat(80)}\n`);
      }
    } else if (message instanceof ToolMessage) {
      console.log("┌─ OBSERVATION");
      console.log("│");
      console.log("│  📊 Result:");
      for (const line of String(message.content).split("\n")) {
        console.log(`│     ${line}`);
      }
      console.log(`└─${"─".repeat(80)}\n`);
    }
  }

  console.log("=".repeat(100));
  console.log(`Total Iterations: ${result.iteration ?? 0}`);
  console.log("=".repeat(100) + "\n");
}

export async function runAdvancedExample(
  query: string,
  maxIterations = 10,
): Promise<void> {
  console.log(`\n${"=".repeat(100)}`);
  console.log(`QUERY: ${query}`);
  console.log(`${"=".repeat(100)}\n`);

  const app = buildGraph();
  const result = await app.invoke({
    messages: [new HumanMessage(query)],
    iteration: 0,
    max_iterations: maxIterations,
  });
  displayReActTrace(result);
}

async function main(): Promise<void> {
  console.log(`
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║              ReAct Pattern - Advanced Implementation                           ║
    ║                                                                                ║
    ║  This version shows explicit Thought → Action → Observation traces             ║
    ║  with iteration tracking and enhanced observability                            ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    `);

  await runAdvancedExample(
    "Find information about Albert Einstein, then calculate what 20% of his Nobel Prize year would be.",
  );
  await runAdvancedExample(
    "Look up information about the Amazon rainforest and analyze how many words are in the description.",
  );
  await runAdvancedExample(
    "Search for the Great Barrier Reef, calculate the square root of its area in square kilometers, and tell me the result.",
  );

  console.log(`
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║                          Examples Complete!                                    ║
    ║                                                                                ║
    ║  The Advanced ReAct agent demonstrated:                                        ║
    ║  • Explicit reasoning traces at each step                                      ║
    ║  • Iteration tracking and limits                                               ║
    ║  • Clear Thought → Action → Observation cycles                                 ║
    ║  • Enhanced observability for debugging                                        ║
    ║  • Custom state management with LangGraph                                      ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    `);
}

if (import.meta.main) {
  await main();
}
