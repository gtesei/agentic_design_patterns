/**
 * Agentic Workflows — TypeScript port of src/research_report_agent.py.
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
import { z } from "zod";

import {
  arxivSearchTool,
  tavilySearchTool,
  wikipediaSearchTool,
  type ToolRecord,
} from "./utils.ts";

const DEFAULT_MODEL = "gpt-4o-mini";
const ADVANCED_MODEL = "gpt-5.2";

function getDefaultModel(): string {
  return process.env.OPENAI_MODEL ?? DEFAULT_MODEL;
}

function getAdvancedModel(): string {
  return (
    process.env.OPENAI_ADVANCED_MODEL ??
    process.env.OPENAI_MODEL ??
    ADVANCED_MODEL
  );
}

export function makeLlm(model: string, temperature = 0): ChatOpenAI {
  return new ChatOpenAI({ model, temperature });
}

export const arxivTool = tool(
  async ({
    query,
    max_results = 5,
  }: {
    query: string;
    max_results?: number;
  }) => await arxivSearchTool(query, max_results),
  {
    name: "arxiv_tool",
    description: "Search arXiv for papers by query string.",
    schema: z.object({
      query: z.string(),
      max_results: z.number().int().optional().default(5),
    }),
  },
);

export const tavilyTool = tool(
  async ({
    query,
    max_results = 5,
    include_images = false,
  }: {
    query: string;
    max_results?: number;
    include_images?: boolean;
  }) => await tavilySearchTool(query, max_results, include_images),
  {
    name: "tavily_tool",
    description: "Web search using the Tavily API.",
    schema: z.object({
      query: z.string(),
      max_results: z.number().int().optional().default(5),
      include_images: z.boolean().optional().default(false),
    }),
  },
);

export const wikipediaTool = tool(
  async ({
    query,
    sentences = 5,
  }: {
    query: string;
    sentences?: number;
  }) => await wikipediaSearchTool(query, sentences),
  {
    name: "wikipedia_tool",
    description: "Fetch a short Wikipedia summary and URL for a query.",
    schema: z.object({
      query: z.string(),
      sentences: z.number().int().optional().default(5),
    }),
  },
);

export const RESEARCH_TOOLS = [arxivTool, tavilyTool, wikipediaTool];
type ToolInvoker = {
  invoke(args: unknown): Promise<unknown>;
};

export const TOOLS_MAP: Record<string, ToolInvoker> = {
  [arxivTool.name]: arxivTool,
  [tavilyTool.name]: tavilyTool,
  [wikipediaTool.name]: wikipediaTool,
};

export const PLANNER_SYSTEM = `You are a planning agent responsible for organizing a research workflow with multiple intelligent agents.

🧠 Available agents:
- A research agent who can search the web, Wikipedia, and arXiv.
- A writer agent who can draft research summaries.
- An editor agent who can reflect and revise drafts.

🎯 Your job is to write a clear, step-by-step research plan as a valid Python list,
where each step is a string. Steps should be atomic and executable.

🚫 DO NOT include irrelevant tasks like "create CSV", "set up a repo", "install packages", etc.
✅ DO include real research-related tasks (e.g., search, summarize, draft, revise).
✅ DO NOT include explanation text — return ONLY the Python list.
✅ The final step should be to generate a Markdown document containing the complete research report.
`;

export const WRITER_SYSTEM = `You are a professional writing assistant specialized in producing clear, well-structured,
and rigorous academic and technical content.

Your role is to draft, expand, summarize, or refine text with high clarity and precision.
- Organize content logically using headings and coherent flow
- Avoid filler language and repetition
- Preserve technical correctness
- Do not invent facts; use only provided info
Produce polished, publication-quality text.
`;

export const EDITOR_SYSTEM = `You are a professional editor specialized in reviewing, critiquing, and improving
academic and technical writing.

Your role is NOT to rewrite from scratch, but to:
- Identify weaknesses in clarity, structure, logic, and flow
- Suggest concrete improvements and improved wording when helpful
- Preserve intent; do not introduce new facts
Provide actionable feedback and (when appropriate) revised versions of problematic sections.
`;

export const RESEARCH_SYSTEM = `You are an expert research assistant designed to execute complex research tasks using external tools.

You have access to:
- arxiv_tool: academic papers / technical research
- tavily_tool: current web context / recent info
- wikipedia_tool: definitions / background

Instructions:
- Use arxiv_tool for scientific/technical claims.
- Use tavily_tool for current events, stats, recent sources.
- Use wikipedia_tool for definitions and background.
- Cite sources by including the tool name and returned URLs for key claims.
- If tools are insufficient, say what's missing; never fabricate.
Return a structured research result.
`;

function cleanMarkdownCodeBlock(raw: string): string {
  const match = raw.match(/```(?:python)?\s*([\s\S]*?)```/);
  return match ? match[1]!.trim() : raw.trim();
}

function parsePythonListLiteral(raw: string): string[] {
  const cleaned = cleanMarkdownCodeBlock(raw).trim();
  if (!cleaned.startsWith("[") || !cleaned.endsWith("]")) {
    throw new Error(`planner_agent returned non-literal plan:\nRaw:\n${raw}`);
  }

  const matches = Array.from(
    cleaned.matchAll(/'([^'\\]*(?:\\.[^'\\]*)*)'|"([^"\\]*(?:\\.[^"\\]*)*)"/g),
  ).map((match) => (match[1] ?? match[2] ?? "").replace(/\\'/g, "'"));

  if (!matches.length) {
    throw new Error(`planner_agent returned non-literal plan:\nRaw:\n${raw}`);
  }

  return matches;
}

export async function plannerAgent(
  topic: string,
  model?: string,
): Promise<string[]> {
  const llm = makeLlm(model ?? getDefaultModel(), 1.0);
  const messages = [
    new SystemMessage(PLANNER_SYSTEM),
    new HumanMessage(`Topic: "${topic}"\n\nReturn ONLY the Python list.`),
  ];
  const response = await llm.invoke(messages);
  const steps = parsePythonListLiteral(String(response.content ?? ""));

  if (!steps.every((step) => typeof step === "string")) {
    throw new Error("planner_agent must return a Python list[str] (as text).");
  }

  console.log("\n🚀 --- Research Execution Plan ---");
  steps.forEach((step, index) => {
    console.log(`${index + 1}. ${step}`);
  });
  console.log("----------------------------------\n");

  return steps;
}

export async function runToolLoop(
  llmWithTools: ReturnType<ChatOpenAI["bindTools"]>,
  messages: BaseMessage[],
  maxTurns = 6,
): Promise<[string, BaseMessage[]]> {
  for (let i = 0; i < maxTurns; i += 1) {
    const response = await llmWithTools.invoke(messages);
    messages.push(response);

    const toolCalls = (response as AIMessage).tool_calls ?? [];
    if (!toolCalls.length) {
      return [String(response.content ?? ""), messages];
    }

    for (const call of toolCalls) {
      const toolObject = TOOLS_MAP[call.name];
      let toolOutput: unknown;

      if (!toolObject) {
        toolOutput = { error: `Unknown tool: ${call.name}` };
      } else {
        try {
          toolOutput = await toolObject.invoke(call.args);
        } catch (error: unknown) {
          toolOutput = {
            error: `${error instanceof Error ? error.name : "Error"}: ${
              error instanceof Error ? error.message : String(error)
            }`,
          };
        }
      }

      messages.push(
        new ToolMessage({
          content: JSON.stringify(toolOutput),
          tool_call_id: call.id ?? "",
        }),
      );
    }
  }

  const last = messages[messages.length - 1];
  if (last instanceof AIMessage && last.content) {
    return [String(last.content), messages];
  }

  return ["Max tool iterations reached. Please simplify the request.", messages];
}

export async function researchAgent(
  task: string,
  model?: string,
  returnMessages = false,
): Promise<string | [string, BaseMessage[]]> {
  console.log("==================================");
  console.log("🔍 Research Agent");
  console.log("==================================");

  const currentTime = new Date().toISOString().slice(0, 10);
  const llm = makeLlm(model ?? getAdvancedModel(), 0.2).bindTools(
    RESEARCH_TOOLS,
  );

  const messages: BaseMessage[] = [
    new SystemMessage(`${RESEARCH_SYSTEM}\n\nCurrent time: ${currentTime}`),
    new HumanMessage(`TASK:\n${task}\n\nBegin now.`),
  ];

  const [content, fullMessages] = await runToolLoop(llm, messages, 6);
  console.log("✅ Output:\n", content);
  return returnMessages ? [content, fullMessages] : content;
}

export async function writerAgent(task: string, model?: string): Promise<string> {
  console.log("==================================");
  console.log("✍️ Writer Agent");
  console.log("==================================");

  const llm = makeLlm(model ?? getAdvancedModel(), 1.0);
  const response = await llm.invoke([
    new SystemMessage(WRITER_SYSTEM),
    new HumanMessage(task),
  ]);
  return String(response.content ?? "");
}

export async function editorAgent(task: string, model?: string): Promise<string> {
  console.log("==================================");
  console.log("🧠 Editor Agent");
  console.log("==================================");

  const llm = makeLlm(model ?? getAdvancedModel(), 0.7);
  const response = await llm.invoke([
    new SystemMessage(EDITOR_SYSTEM),
    new HumanMessage(task),
  ]);
  return String(response.content ?? "");
}

export const agentRegistry: Record<
  AgentDecision["agent"],
  (task: string) => Promise<string>
> = {
  research_agent: async (task) => (await researchAgent(task)) as string,
  editor_agent: editorAgent,
  writer_agent: writerAgent,
};

export function cleanJsonBlock(raw: string): string {
  let cleaned = raw.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?\n?/, "");
    cleaned = cleaned.replace(/\n?```$/, "");
  }
  return cleaned.trim();
}

export interface AgentDecision {
  agent: "research_agent" | "editor_agent" | "writer_agent";
  task: string;
}

export async function decideAgent(
  step: string,
  model?: string,
): Promise<AgentDecision> {
  const llm = makeLlm(model ?? getAdvancedModel(), 0.0);
  const prompt = `
You are an execution manager for a multi-agent research team.

Given the following instruction, identify which agent should perform it and extract the clean task.

Return only a valid JSON object with two keys:
- "agent": one of ["research_agent", "editor_agent", "writer_agent"]
- "task": a string with the instruction that the agent should follow

Only respond with a valid JSON object. Do not include explanations or markdown formatting.

Instruction: "${step}"
`.trim();

  const response = await llm.invoke([new HumanMessage(prompt)]);
  const info = JSON.parse(cleanJsonBlock(String(response.content ?? ""))) as AgentDecision;
  return info;
}

export async function executorAgent(
  topic: string,
  model?: string,
  limitSteps = true,
  maxSteps = 10,
): Promise<Array<[string, string, string]>> {
  let planSteps = await plannerAgent(topic, model);
  if (limitSteps) {
    planSteps = planSteps.slice(0, Math.min(planSteps.length, maxSteps));
  }

  const history: Array<[string, string, string]> = [];

  console.log("==================================");
  console.log("🎯 Executor Agent");
  console.log("==================================");

  for (const [index, step] of planSteps.entries()) {
    const decision = await decideAgent(step, model);
    const agentName = decision.agent;
    const task = decision.task;
    const context = history
      .map(
        ([, previousAgent, result], historyIndex) =>
          `Step ${historyIndex + 1} executed by ${previousAgent}:\n${result}`,
      )
      .join("\n");

    const enrichedTask = `
You are ${agentName}.

Here is the context of what has been done so far:
${context}

Your next task is:
${task}
`.trim();

    console.log(`\n🛠️ Step ${index + 1}: agent=\`${agentName}\` task=${task}`);

    let output: string;
    if (!(agentName in agentRegistry)) {
      output = `⚠️ Unknown agent: ${agentName}`;
    } else {
      output = await agentRegistry[agentName](enrichedTask);
    }

    history.push([step, agentName, output]);
    console.log(`✅ Output:\n${output}`);
  }

  return history;
}

async function main(): Promise<void> {
  const topic = "The ensemble Kalman filter for time series forecasting";
  console.log();
  console.log("\n=== User topic ===\n");
  console.log(topic);

  const history = await executorAgent(topic, undefined, true);
  console.log("\n================ FINAL ================\n");
  console.log(history[history.length - 1]?.[2] ?? "");
}

if (import.meta.main) {
  await main();
}
