/**
 * Reflection Pattern with Stateful Loops — TypeScript port of src/reflection_stateful_loop.py.
 */

import { ChatOpenAI } from "@langchain/openai";
import { Annotation, END, StateGraph } from "@langchain/langgraph";
import { z } from "zod";

const DEFAULT_MODEL = process.env.OPENAI_MODEL ?? "gpt-4o-mini";
const REASONING_MODEL = process.env.OPENAI_ADVANCED_MODEL ?? DEFAULT_MODEL;

let reflectionModel = REASONING_MODEL;
let llm = new ChatOpenAI({ temperature: 0.7, model: reflectionModel });

function contentToString(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (typeof part === "object" && part !== null && "text" in part) {
          return String((part as { text: unknown }).text);
        }
        return "";
      })
      .join("");
  }
  return String(content ?? "");
}

export const BlogPostState = Annotation.Root({
  topic: Annotation<string>(),
  target_audience: Annotation<string>(),
  tone: Annotation<string>(),
  word_count_target: Annotation<number>(),
  draft: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  critique: Annotation<string>({
    reducer: (_left, right) => right,
    default: () => "",
  }),
  iteration: Annotation<number>({
    reducer: (left, right) => left + right,
    default: () => 0,
  }),
  issues_found: Annotation<string[]>({
    reducer: (_left, right) => right,
    default: () => [],
  }),
  quality_score: Annotation<number>({
    reducer: (_left, right) => right,
    default: () => 0,
  }),
  is_approved: Annotation<boolean>({
    reducer: (_left, right) => right,
    default: () => false,
  }),
  max_iterations: Annotation<number>(),
  improvement_history: Annotation<
    Array<{
      iteration: number;
      quality_score: number;
      issues_count: number;
      publication_ready: boolean;
    }>
  >({
    reducer: (left, right) => left.concat(right),
    default: () => [],
  }),
});

type State = typeof BlogPostState.State;

export const BlogCritiqueSchema = z.object({
  overall_quality: z.number().min(0).max(10),
  flow_score: z.number().min(0).max(10),
  flow_issues: z.array(z.string()).default([]),
  tone_score: z.number().min(0).max(10),
  tone_issues: z.array(z.string()).default([]),
  clarity_score: z.number().min(0).max(10),
  clarity_issues: z.array(z.string()).default([]),
  content_score: z.number().min(0).max(10),
  content_issues: z.array(z.string()).default([]),
  engagement_score: z.number().min(0).max(10),
  engagement_issues: z.array(z.string()).default([]),
  strengths: z.array(z.string()).default([]),
  priority_improvements: z.array(z.string()).default([]),
  suggested_additions: z.array(z.string()).default([]),
  suggested_removals: z.array(z.string()).default([]),
  is_publication_ready: z.boolean(),
});

type BlogCritique = z.infer<typeof BlogCritiqueSchema>;

export async function writerNode(state: State): Promise<Partial<State>> {
  console.log(`\n${"=".repeat(80)}`);
  console.log(`ITERATION ${state.iteration}: WRITER`);
  console.log(`${"=".repeat(80)}`);

  let context: string;
  let additionalInstructions: string;

  if (state.iteration === 0) {
    context = "This is your first draft. Write freely and comprehensively.";
    additionalInstructions = "Focus on getting ideas down. We'll refine later.";
  } else {
    context = `This is iteration ${state.iteration}.

Previous quality score: ${state.quality_score.toFixed(1)}/10

Previous critique summary:
${state.critique.slice(0, 500)}...

Issues to address:
${state.issues_found.slice(0, 5).map((issue) => `- ${issue}`).join("\n")}`;

    additionalInstructions = `CRITICAL: Address the specific feedback above.
Focus on the priority improvements identified by the editor.
Build on what's working; fix what's not.`;
  }

  let draft: string;
  if (state.iteration === 0) {
    const prompt = `You are an expert blog writer specializing in ${state.tone} content for ${state.target_audience}.

${context}

Topic: ${state.topic}
Target Word Count: ${state.word_count_target}
Tone: ${state.tone}
Audience: ${state.target_audience}

Requirements:
- Write engaging, well-structured content
- Use clear, accessible language
- Include relevant examples and insights
- Create compelling introduction and conclusion
- Use subheadings for better readability
- Maintain consistent voice throughout

${additionalInstructions}

Write the blog post:`;
    const response = await llm.invoke(prompt);
    draft = contentToString(response.content);
  } else {
    const priorityImprovements = state.issues_found.slice(0, 5).join("\n");
    const prompt = `You are an expert blog writer. Rewrite this blog post based on the critique.

Original Draft:
${state.draft}

Detailed Critique:
${state.critique}

Priority Improvements:
${priorityImprovements}

Instructions:
- Address ALL identified issues
- Maintain what works well
- Implement suggested improvements
- Keep the same topic and core message
- Match the ${state.tone} tone for ${state.target_audience}
- Aim for ${state.word_count_target} words

Write the improved version:`;
    const response = await llm.invoke(prompt);
    draft = contentToString(response.content);
  }

  const wordCount = draft.split(/\s+/).filter(Boolean).length;
  console.log(`\nDraft generated: ${wordCount} words`);
  console.log("\nFirst 300 characters:");
  console.log(`${draft.slice(0, 300)}...`);

  return {
    draft,
    iteration: 1,
  };
}

export async function criticNode(state: State): Promise<Partial<State>> {
  console.log(`\n${"=".repeat(80)}`);
  console.log(`ITERATION ${state.iteration}: CRITIC`);
  console.log(`${"=".repeat(80)}`);

  const llmStructured = new ChatOpenAI({
    temperature: 0,
    model: reflectionModel,
  }).withStructuredOutput(BlogCritiqueSchema);

  const critiqueResult = (await llmStructured.invoke(`You are an expert content editor and writing coach. Evaluate this blog post comprehensively.

Topic: ${state.topic}
Target Audience: ${state.target_audience}
Desired Tone: ${state.tone}
Target Word Count: ${state.word_count_target}

Blog Post Draft:
${state.draft}

Evaluate on these dimensions:
1. **Flow & Structure**: Is it logically organized? Do paragraphs transition smoothly?
2. **Tone & Voice**: Does it match the ${state.tone} tone? Is voice consistent?
3. **Clarity & Readability**: Is it easy to understand? Any jargon or unclear passages?
4. **Content Quality**: Is it informative, accurate, and valuable?
5. **Engagement**: Will it hook and retain readers?

Provide detailed, actionable feedback. Be specific about what to fix and how.`)) as BlogCritique;

  const critiqueText = `
OVERALL QUALITY: ${critiqueResult.overall_quality}/10

DETAILED SCORES:
- Flow & Structure: ${critiqueResult.flow_score}/10
- Tone & Voice: ${critiqueResult.tone_score}/10
- Clarity & Readability: ${critiqueResult.clarity_score}/10
- Content Quality: ${critiqueResult.content_score}/10
- Engagement: ${critiqueResult.engagement_score}/10

STRENGTHS:
${critiqueResult.strengths.map((s) => `✓ ${s}`).join("\n")}

PRIORITY IMPROVEMENTS:
${critiqueResult.priority_improvements.map((p) => `! ${p}`).join("\n")}

FLOW ISSUES:
${critiqueResult.flow_issues.map((i) => `- ${i}`).join("\n")}

TONE ISSUES:
${critiqueResult.tone_issues.map((i) => `- ${i}`).join("\n")}

CLARITY ISSUES:
${critiqueResult.clarity_issues.map((i) => `- ${i}`).join("\n")}

CONTENT ISSUES:
${critiqueResult.content_issues.map((i) => `- ${i}`).join("\n")}

ENGAGEMENT ISSUES:
${critiqueResult.engagement_issues.map((i) => `- ${i}`).join("\n")}

SUGGESTED ADDITIONS:
${critiqueResult.suggested_additions.map((a) => `+ ${a}`).join("\n")}

SUGGESTED REMOVALS:
${critiqueResult.suggested_removals.map((r) => `- ${r}`).join("\n")}

PUBLICATION READY: ${critiqueResult.is_publication_ready ? "YES" : "NO"}
`;

  console.log("\nCritique Summary:");
  console.log(`  Overall Quality: ${critiqueResult.overall_quality}/10`);
  console.log(`  Flow: ${critiqueResult.flow_score}/10`);
  console.log(`  Tone: ${critiqueResult.tone_score}/10`);
  console.log(`  Clarity: ${critiqueResult.clarity_score}/10`);
  console.log(`  Content: ${critiqueResult.content_score}/10`);
  console.log(`  Engagement: ${critiqueResult.engagement_score}/10`);
  console.log(`  Publication Ready: ${critiqueResult.is_publication_ready}`);

  const allIssues = [
    ...critiqueResult.flow_issues,
    ...critiqueResult.tone_issues,
    ...critiqueResult.clarity_issues,
    ...critiqueResult.content_issues,
    ...critiqueResult.engagement_issues,
  ];

  const improvementEntry = {
    iteration: state.iteration,
    quality_score: critiqueResult.overall_quality,
    issues_count: allIssues.length,
    publication_ready: critiqueResult.is_publication_ready,
  };

  return {
    critique: critiqueText,
    quality_score: critiqueResult.overall_quality,
    issues_found: allIssues,
    is_approved: critiqueResult.is_publication_ready,
    improvement_history: [improvementEntry],
  };
}

export function decisionNode(state: State): Partial<State> {
  console.log(`\n${"=".repeat(80)}`);
  console.log("DECISION NODE");
  console.log(`${"=".repeat(80)}`);
  console.log(`\nQuality Score: ${state.quality_score}/10`);
  console.log(`Approved: ${state.is_approved}`);
  console.log(`Issues Found: ${state.issues_found.length}`);
  console.log(`Current Iteration: ${state.iteration}/${state.max_iterations}`);

  const qualityThreshold = 8.0;
  const meetsThreshold = state.quality_score >= qualityThreshold;
  console.log(`Meets Quality Threshold (${qualityThreshold}): ${meetsThreshold}`);
  return state;
}

export function shouldContinue(state: State): "writer" | typeof END {
  if (state.is_approved && state.quality_score >= 8.0) {
    console.log("\n✓ Blog post approved for publication!");
    return END;
  }

  if (state.iteration >= state.max_iterations) {
    if (state.quality_score >= 7.0) {
      console.log(
        `\n⚠ Max iterations reached. Quality acceptable (${state.quality_score}/10)`,
      );
    } else {
      console.log(
        `\n✗ Max iterations reached. Quality below target (${state.quality_score}/10)`,
      );
    }
    return END;
  }

  console.log(`\n→ Continuing to iteration ${state.iteration + 1}`);
  console.log(`   Focus: ${JSON.stringify(state.issues_found.slice(0, 3))}`);
  return "writer";
}

export function createBlogWriterGraph() {
  return new StateGraph(BlogPostState)
    .addNode("writer", writerNode)
    .addNode("critic", criticNode)
    .addNode("decision", decisionNode)
    .addEdge("__start__", "writer")
    .addEdge("writer", "critic")
    .addEdge("critic", "decision")
    .addConditionalEdges("decision", shouldContinue, {
      writer: "writer",
      [END]: END,
    })
    .compile();
}

export async function writeBlogPost(
  topic: string,
  targetAudience = "general readers",
  tone = "professional yet accessible",
  wordCountTarget = 800,
  maxIterations = 3,
): Promise<State> {
  console.log("\n" + "=".repeat(80));
  console.log("BLOG POST WRITER WITH REFLECTION PATTERN");
  console.log("=".repeat(80));

  console.log(`\nTopic: ${topic}`);
  console.log(`Audience: ${targetAudience}`);
  console.log(`Tone: ${tone}`);
  console.log(`Target Length: ${wordCountTarget} words`);
  console.log(`Max Iterations: ${maxIterations}`);

  const initialState = {
    topic,
    target_audience: targetAudience,
    tone,
    word_count_target: wordCountTarget,
    draft: "",
    critique: "",
    iteration: 0,
    issues_found: [],
    quality_score: 0,
    is_approved: false,
    max_iterations: maxIterations,
    improvement_history: [],
  };

  const graph = createBlogWriterGraph();
  const finalState = await graph.invoke(initialState);

  console.log("\n" + "=".repeat(80));
  console.log("FINAL RESULTS");
  console.log("=".repeat(80));

  console.log(`\nTotal Iterations: ${finalState.iteration}`);
  console.log(`Final Quality Score: ${finalState.quality_score}/10`);
  console.log(`Status: ${finalState.is_approved ? "✓ APPROVED" : "⚠ NEEDS WORK"}`);
  console.log(`Word Count: ${finalState.draft.split(/\s+/).filter(Boolean).length} words`);

  console.log("\nImprovement Trajectory:");
  for (const entry of finalState.improvement_history) {
    const status = entry.publication_ready ? "✓" : "✗";
    console.log(
      `  Iteration ${entry.iteration}: ${entry.quality_score.toFixed(1)}/10 ${status} (${entry.issues_count} issues)`,
    );
  }

  if (finalState.issues_found.length > 0) {
    console.log(`\nRemaining Issues (${finalState.issues_found.length}):`);
    finalState.issues_found.slice(0, 5).forEach((issue, index) => {
      console.log(`  ${index + 1}. ${issue}`);
    });
  }

  console.log(`\n${"=".repeat(80)}`);
  console.log("FINAL BLOG POST");
  console.log("=".repeat(80));
  console.log(finalState.draft);

  console.log(`\n${"=".repeat(80)}`);
  console.log("FINAL CRITIQUE");
  console.log("=".repeat(80));
  console.log(finalState.critique);

  return finalState;
}

export const EXAMPLE_TOPICS = {
  technical: {
    topic: "Understanding Agentic AI Frameworks: A Practical Guide",
    target_audience: "software developers and ML engineers",
    tone: "technical but accessible",
    word_count_target: 1200,
  },
  business: {
    topic: "Why Remote Work is Transforming Company Culture",
    target_audience: "business leaders and HR professionals",
    tone: "professional and analytical",
    word_count_target: 1000,
  },
  casual: {
    topic: "5 Simple Habits That Changed My Morning Routine",
    target_audience: "lifestyle enthusiasts",
    tone: "casual and conversational",
    word_count_target: 800,
  },
  educational: {
    topic: "How Machine Learning is Revolutionizing Healthcare",
    target_audience: "healthcare professionals with limited tech background",
    tone: "educational and encouraging",
    word_count_target: 1000,
  },
} as const;

export function visualizeImprovement(state: State): void {
  console.log("\n" + "=".repeat(80));
  console.log("QUALITY IMPROVEMENT VISUALIZATION");
  console.log("=".repeat(80));

  const seenIterations = new Map<number, (typeof state.improvement_history)[number]>();
  for (const entry of state.improvement_history) {
    seenIterations.set(entry.iteration, entry);
  }

  for (const iteration of [...seenIterations.keys()].sort((a, b) => a - b)) {
    const entry = seenIterations.get(iteration)!;
    const score = entry.quality_score;
    const ready = entry.publication_ready;
    const barLength = Math.floor(score);
    const bar = "█".repeat(barLength) + "░".repeat(10 - barLength);
    const status = ready ? "✓" : "✗";
    console.log(`Iteration ${iteration}: ${bar} ${score.toFixed(1)}/10 ${status}`);
  }
}

async function main(): Promise<void> {
  const quick = process.argv.includes("--quick");

  if (quick) {
    reflectionModel = DEFAULT_MODEL;
    llm = new ChatOpenAI({ temperature: 0.7, model: reflectionModel });
    console.log("\n" + "#".repeat(80));
    console.log("# QUICK MODE: Compact Reflection Demo");
    console.log("#".repeat(80));
    console.log("Using one shorter example, fewer iterations, and the default model.\n");

    const result = await writeBlogPost(
      "Understanding Agentic AI Frameworks: A Practical Guide",
      "software developers and ML engineers",
      "technical but accessible",
      450,
      1,
    );
    visualizeImprovement(result);
    return;
  }

  console.log("\n" + "#".repeat(80));
  console.log("# EXAMPLE 1: Technical Blog Post");
  console.log("#".repeat(80));

  const result1 = await writeBlogPost(
    "Understanding Agentic AI Frameworks: A Practical Guide",
    "software developers and ML engineers",
    "technical but accessible",
    1200,
    3,
  );
  visualizeImprovement(result1);

  console.log("\n" + "#".repeat(80));
  console.log("# EXAMPLE 2: Business Blog Post (Uncomment to run)");
  console.log("#".repeat(80));

  const result2 = await writeBlogPost(
    EXAMPLE_TOPICS.business.topic,
    EXAMPLE_TOPICS.business.target_audience,
    EXAMPLE_TOPICS.business.tone,
    EXAMPLE_TOPICS.business.word_count_target,
    3,
  );
  visualizeImprovement(result2);
}

if (import.meta.main) {
  await main();
}
