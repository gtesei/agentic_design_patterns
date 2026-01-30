# Prompt Chaining - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Directory
```bash
cd foundational_design_patterns/1_prompt_chain
```

### Step 2: Run the Example
```bash
bash run.sh
```

---

## 📖 Understanding Prompt Chaining in 30 Seconds

**Prompt Chaining** breaks complex tasks into a sequence of simpler steps:

```
Input → Step 1 → Step 2 → Step 3 → Final Output
```

Each step focuses on ONE thing, making the process:
- More reliable (fewer errors)
- More transparent (see each step)
- Easier to debug (find problems faster)

---

## 🎯 What This Example Does

The example demonstrates a **content generation pipeline**:

1. **Research** - Gather key facts about a topic
2. **Outline** - Create a structured outline
3. **Draft** - Write the initial content
4. **Edit** - Refine and improve the draft

---

## 💡 Example Output Structure

```
Topic: "Climate Change"
    ↓
Research Step: "Key facts: greenhouse gases, temperature rise..."
    ↓
Outline Step: "I. Introduction II. Causes III. Effects..."
    ↓
Draft Step: "Climate change refers to long-term shifts..."
    ↓
Edit Step: "Climate change represents one of the most..."
```

---

## 🔧 Key Concepts

### Sequential Processing
Each step waits for the previous step to complete before starting.

### Focused Prompts
Each step has a specific, narrow task instead of trying to do everything at once.

### Intermediate Outputs
You can inspect, validate, or modify outputs between steps.

### Error Isolation
If something fails, you know exactly which step had the problem.

---

## 🎨 When to Use Prompt Chaining

✅ **Good For:**
- Multi-step document processing
- Content generation workflows
- Data transformation pipelines
- Sequential reasoning tasks

❌ **Not Ideal For:**
- Simple single-step tasks
- Real-time applications (adds latency)
- Tasks needing all context at once

---

## 🛠️ Customization Tips

### Modify the Pipeline

Edit `src/chain_prompt.py` to change steps:

```python
# Add a new step
def new_step(input_text):
    prompt = f"Process this: {input_text}"
    return llm.invoke(prompt)

# Add to chain
result = step1(input) | step2 | new_step | step3
```

### Change the Topic

Modify the input in the main function:

```python
topic = "Your Custom Topic Here"
```

### Add Validation

Insert validation between steps:

```python
research = research_step(topic)
if len(research) < 100:
    research = research_step(topic)  # Retry
outline = outline_step(research)
```

---

## 📊 Performance Notes

- **Latency**: Each step adds ~1-3 seconds (sequential)
- **Tokens**: Each step uses separate tokens
- **Quality**: Higher quality than single-step approach
- **Cost**: 3-5x more expensive than single prompt

**Trade-off**: Better quality and reliability vs. higher cost and latency

---

## 🔍 Debugging Tips

1. **Print intermediate outputs** to see what each step produces
2. **Test each step independently** before chaining
3. **Check token usage** for each step
4. **Validate outputs** between steps
5. **Use smaller models** for non-critical steps

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)
- **Related Patterns**:
  - Pattern 4 (Reflection) - Quality improvement
  - Pattern 6 (Planning) - Complex orchestration

---

## 🎓 Next Steps

1. ✅ Run the basic example
2. ✅ Modify the topic and see results
3. ✅ Add a custom step to the chain
4. ✅ Try different validation strategies
5. ✅ Combine with other patterns

---

**Pattern Type**: Sequential Processing

**Complexity**: ⭐⭐ (Beginner-Friendly)

**Best For**: Multi-step transformations
