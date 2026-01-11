# Reflection Pattern

## Overview

The **Reflection Pattern** enables agentic systems to iteratively self-correct and refine their outputs through a structured feedback loop of generation, evaluation, and improvement, leading to significantly higher quality results.

## Why Use This Pattern?

LLMs can produce outputs that lack accuracy, miss nuances, or fail to fully satisfy complex requirements on the first attempt. While a single generation might be "good enough" for simple tasks, critical applications require outputs that are polished, accurate, and thoroughly vetted.

The Reflection pattern solves this by:
- **Establishing a feedback loop** where outputs are critically evaluated
- **Identifying gaps and errors** through systematic critique
- **Iteratively refining** based on structured feedback
- **Progressively improving quality** with each reflection cycle

### Example: Single-Shot vs. Reflection
```
Single-Shot (Fast but potentially flawed):
Input → Generate Output → Done

Reflection (Higher quality):
Input → Generate v1 → Critique v1 → Generate v2 → Critique v2 → Final Output
```

## How It Works

1. **Producer generates initial output** based on the task requirements
2. **Critic evaluates the output** against predefined criteria (accuracy, completeness, coherence)
3. **Feedback is provided** highlighting strengths, weaknesses, and specific improvements needed
4. **Producer refines the output** incorporating the critique
5. **Cycle repeats** until quality threshold is met or iteration limit reached

### Typical Architecture

#### Single-Agent Reflection (Self-Critique)
```
Input
  ↓
Generate Initial Output
  ↓
Self-Critique
  ↓
Refine Output
  ↓
[Repeat if needed]
  ↓
Final Output
```

#### Producer-Critic Model (Dual-Agent)
```
Input
  ↓
Producer Agent: Generate v1
  ↓
Critic Agent: Evaluate & Provide Feedback
  ↓
Producer Agent: Generate v2 (with critique context)
  ↓
Critic Agent: Re-evaluate
  ↓
[Continue until criteria met]
  ↓
Final Output
```

## When to Use This Pattern

### ✅ Ideal Use Cases

- **Long-form content creation**: Articles, reports, documentation requiring polish and coherence
- **Code generation and debugging**: Writing correct, well-structured, maintainable code
- **Complex planning**: Strategic plans, architectural designs, detailed roadmaps
- **High-stakes communication**: Legal documents, customer-facing content, technical specifications
- **Creative refinement**: Marketing copy, storytelling, design descriptions
- **Technical accuracy**: Scientific writing, data analysis reports, research summaries
- **Compliance-sensitive outputs**: Content requiring adherence to strict guidelines or regulations

### ❌ When NOT to Use

- **Time-sensitive responses**: Chat applications, real-time interactions where speed matters most
- **Cost-constrained scenarios**: Multiple LLM calls significantly increase token usage
- **Simple, straightforward tasks**: Basic classification, extraction, or formatting
- **When "good enough" suffices**: Internal drafts, exploratory work, rapid prototyping
- **Token budget limitations**: Risk of exceeding context windows with lengthy critique cycles
- **Rate-limited environments**: Multiple API calls may trigger throttling

## Rule of Thumb

**Use Reflection when:**
1. **Quality, accuracy, and detail** are more important than speed and cost
2. Outputs require **specialized evaluation** (technical correctness, stylistic nuance)
3. Tasks involve **complex instructions** that benefit from iterative refinement
4. The **cost of errors** is high (production code, published content, critical decisions)

**Use a separate Critic Agent when:**
1. Tasks require **high objectivity** (avoiding producer bias)
2. **Specialized expertise** is needed for evaluation (e.g., security review, compliance check)
3. Critique requires **different context or knowledge** than generation
4. You want **structured, consistent feedback** across evaluations

**Don't use Reflection when:**
1. Simple tasks complete correctly on first attempt
2. Speed and cost constraints outweigh quality benefits
3. Iterative refinement shows diminishing returns
4. Context window limitations make feedback incorporation difficult

## Framework Support

### LangChain (LCEL)

LangChain supports single-step reflection using sequential chains:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI()

# Producer chain
producer_prompt = ChatPromptTemplate.from_template(
    "Generate a technical blog post about {topic}"
)
producer_chain = producer_prompt | llm | StrOutputParser()

# Critic chain
critic_prompt = ChatPromptTemplate.from_template(
    "Critique this blog post and suggest improvements:\n\n{draft}"
)
critic_chain = critic_prompt | llm | StrOutputParser()

# Refinement chain
refine_prompt = ChatPromptTemplate.from_template(
    "Improve this blog post based on the critique:\n\nOriginal:\n{draft}\n\nCritique:\n{critique}"
)
refine_chain = refine_prompt | llm | StrOutputParser()

# Single reflection step
reflection_chain = (
    {"draft": producer_chain}
    | RunnablePassthrough.assign(critique=lambda x: critic_chain.invoke({"draft": x["draft"]}))
    | refine_chain
)
```

For **iterative reflection**, use LangGraph for stateful loops.

### LangGraph

LangGraph enables full iterative reflection with state management:
```python
from langgraph.graph import StateGraph

# Define state with draft, critique, iteration tracking
# Create producer node, critic node, decision node
# Loop until quality threshold met or max iterations reached
```

### Google ADK

Google's Agent Development Kit supports reflection through:
- **Sequential workflows**: One agent produces, another critiques, producer refines
- **Iterative delegation**: Coordinator manages reflection cycles
- **Specialized agents**: Dedicated critic agents with domain expertise

### Other Frameworks

- **AutoGen**: Multi-agent reflection through conversational feedback
- **CrewAI**: Role-based agents where critic reviews producer output
- **Semantic Kernel**: Function chaining for critique and refinement

## Key Benefits

### 🎯 Quality & Accuracy
- **Higher correctness**: Iterative error correction catches mistakes
- **Better adherence to requirements**: Critique ensures all criteria are met
- **Enhanced coherence**: Multiple passes improve logical flow and consistency
- **Reduced hallucinations**: Critical evaluation filters out unsupported claims

### 🧠 Specialized Evaluation
- **Objectivity**: Separate critic avoids producer bias
- **Domain expertise**: Specialized critics (security, compliance, style)
- **Structured feedback**: Consistent evaluation criteria across iterations

### 📈 Continuous Improvement
- **Progressive refinement**: Each cycle builds on previous improvements
- **Learning from mistakes**: Explicit feedback guides better outputs
- **Adaptive quality**: Can iterate until specific thresholds are met

## Important Considerations

### ⚠️ Cost & Latency Trade-offs

**Increased Token Usage:**
- Multiple LLM calls (producer + critic + refined producer)
- Longer context windows (including critique in refinement prompt)
- Potential for 2-5x token costs compared to single-shot generation

**Higher Latency:**
- Sequential evaluation and refinement steps
- Each iteration adds cumulative delay
- Can take 3-10x longer than single-shot approaches

**Context Window Risks:**
- Long drafts + detailed critiques can exceed token limits
- Requires careful prompt engineering to stay within bounds
- May need to truncate or summarize earlier iterations

### 🔄 Diminishing Returns

- **First reflection**: Often provides 60-80% of total quality improvement
- **Subsequent iterations**: Marginal gains decrease with each cycle
- **Practical limit**: 2-3 reflection cycles usually optimal

### 🛠️ Implementation Complexity

**Single-Step Reflection (Simple):**
- Can be implemented with LCEL in LangChain
- One critique, one refinement
- Suitable for most use cases

**Iterative Reflection (Complex):**
- Requires stateful workflows (LangGraph, AutoGen)
- Loop management, convergence criteria
- More complex debugging and monitoring

### 🎯 Critique Quality Matters

- **Vague feedback**: "This could be better" provides little actionable guidance
- **Specific, actionable critique**: "Add quantitative metrics in section 2; clarify technical terms for non-expert audience"
- **Structured evaluation**: Use rubrics or checklists for consistent critique

## Best Practices

1. **Define clear evaluation criteria**: Specify what "good" looks like (accuracy, completeness, style)
2. **Use structured critique prompts**: Ask for specific feedback categories (technical accuracy, clarity, completeness)
3. **Set iteration limits**: Prevent infinite loops (typically 2-3 iterations max)
4. **Monitor token usage**: Track cumulative costs per reflection cycle
5. **Provide examples in critique prompts**: Show the critic what good feedback looks like
6. **Consider hybrid approaches**: Single reflection for most cases, iterative for critical outputs
7. **Measure quality improvements**: Track metrics to validate reflection effectiveness
8. **Include critique in context**: Ensure refined producer sees original critique for targeted improvements

## Performance Metrics

Track these metrics to evaluate reflection effectiveness:

- **Quality improvement per iteration**: Measure output quality at each step
- **Token usage**: Total tokens across all reflection cycles
- **Latency**: Time from input to final refined output
- **Convergence rate**: How many iterations typically needed
- **Critique quality**: Actionability and specificity of feedback
- **Cost per request**: Total API costs including all reflection steps
- **Error reduction**: Decrease in factual errors, logical inconsistencies

## Example Scenarios

### Scenario 1: Code Generation with Debugging
```
Iteration 0 (Producer): Generate Python function (5 seconds, 500 tokens)
Iteration 1 (Critic): Identify bug in edge case handling (3 seconds, 200 tokens)
Iteration 2 (Refine): Fix bug, add error handling (5 seconds, 600 tokens)
Total: 13 seconds, 1,300 tokens

Result: Working, robust code vs. buggy initial version
```

### Scenario 2: Technical Blog Post
```
Iteration 0: Draft article (8 seconds, 1,200 tokens)
Iteration 1 (Critic): Suggest adding concrete examples, clarify jargon (4 seconds, 300 tokens)
Iteration 2 (Refine): Add examples, simplify language (8 seconds, 1,400 tokens)
Iteration 3 (Critic): Verify improvements meet standards (4 seconds, 200 tokens)
Total: 24 seconds, 3,100 tokens

Result: Polished, accessible article vs. technical-but-unclear draft
```

### Scenario 3: Strategic Plan
```
Single-shot: High-level plan, missing critical details (10 seconds, 800 tokens)

With Reflection:
- Iteration 0: Initial plan (10 seconds, 800 tokens)
- Critique: Identify missing risk analysis, timeline gaps (5 seconds, 300 tokens)
- Refine: Add risk mitigation, detailed timeline (10 seconds, 1,000 tokens)
Total: 25 seconds, 2,100 tokens

Result: Comprehensive, actionable plan
```

## Related Patterns

- **Prompt Chaining**: Reflection is a specialized form of chaining with feedback
- **Tool Use**: Critic may use external tools (linters, fact-checkers) for evaluation
- **Routing**: Route to reflection loop only for complex/high-stakes tasks
- **Parallelization**: Can parallelize multiple independent critiques (style, accuracy, completeness)

## Conclusion

Reflection is an essential pattern for agentic systems requiring high-quality, accurate, or nuanced outputs. By introducing a structured feedback loop of generation, critique, and refinement, it enables progressive quality improvement at the cost of increased latency and token usage.

**Use Reflection when:**
- Quality and accuracy are paramount
- Outputs have high stakes (production code, published content, critical decisions)
- Complex requirements benefit from iterative refinement
- You can afford the latency and cost trade-offs

**Implementation guidance:**
- Start with **single-step reflection** (producer → critic → refine) for most cases
- Use **iterative reflection** (2-3 cycles) only for critical outputs
- Employ **separate critic agents** when objectivity or specialized evaluation is needed
- Consider **hybrid approaches**: reflection for complex tasks, single-shot for simple ones

**Key Takeaways:**
- 🎯 Iterative self-correction significantly improves output quality
- 🔄 2-3 reflection cycles typically provide optimal quality/cost balance
- 🤖 Producer-Critic separation enhances objectivity and specialization
- ⚠️ Costs 2-5x more tokens and time than single-shot generation
- 📊 Measure quality gains to ensure benefits justify costs

---

*Reflection transforms good-enough outputs into polished, production-ready results—but reserve it for tasks where quality truly matters.*