# Error Recovery Pattern - Quick Start Guide

Get started with error recovery in less than 5 minutes!

## Prerequisites

- Python 3.11+
- OpenAI API key

## Installation

1. **Navigate to the directory:**
```bash
cd reliability/error_recovery
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -e .
```

4. **Set up environment variables:**
Create a `.env` file in the project root (two levels up):
```bash
echo "OPENAI_API_KEY=your-api-key-here" > ../../.env
```

## Running Examples

### Basic Error Recovery

Simple retry mechanism with exponential backoff:

```bash
chmod +x run.sh
./run.sh basic
```

**What it demonstrates:**
- Automatic retry on transient failures
- Exponential backoff with jitter
- Multiple recovery strategies (retry, fallback, alternative)
- Error classification and logging

### Advanced Self-Correction

Complex recovery with validation and learning:

```bash
./run.sh advanced
```

**What it demonstrates:**
- Self-correction loops for LLM output validation
- Circuit breaker pattern implementation
- Multi-level fallback strategies
- Error pattern learning
- Comprehensive recovery history

### Run Both Examples

```bash
./run.sh
```

## Quick Example Code

### Basic Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_api():
    """Automatically retry on failure with exponential backoff"""
    response = api_client.get("/data")
    return response.json()
```

### With Error Classification

```python
from src.recovery_basic import ErrorRecoveryAgent

agent = ErrorRecoveryAgent()

# Automatically handles transient errors, retries, and fallbacks
result = agent.execute_with_recovery(
    operation="summarize_text",
    text="Your text here..."
)
```

## Key Concepts

### 1. Error Types

- **Transient**: Temporary failures (timeouts, network blips) → Retry
- **Retryable**: Fixable errors (token limit, format issues) → Modify and retry
- **Permanent**: Unrecoverable (auth failure, not found) → Fail or fallback

### 2. Recovery Strategies

- **Retry**: Try again with exponential backoff
- **Fallback**: Use alternative service or cached data
- **Self-Correction**: Fix the issue and retry (for LLM outputs)
- **Circuit Breaker**: Stop trying after repeated failures

### 3. Best Practices

✅ **DO:**
- Classify errors before choosing recovery strategy
- Use exponential backoff with jitter
- Set reasonable retry limits (3-5 attempts)
- Log all errors, even when recovered
- Validate outputs after recovery

❌ **DON'T:**
- Retry permanent errors
- Use unlimited retries
- Ignore error context
- Hide critical issues with successful recovery

## Next Steps

1. **Read the full README.md** for comprehensive pattern documentation
2. **Explore the source code** in `src/recovery_basic.py` and `src/recovery_advanced.py`
3. **Customize recovery strategies** for your specific use case
4. **Add monitoring** to track recovery metrics
5. **Integrate with your application** using the provided patterns

## Common Use Cases

### API Timeouts
```python
result = agent.execute_with_recovery(
    operation="call_api",
    endpoint="https://api.example.com/data",
    max_retries=3
)
```

### Invalid LLM Outputs
```python
result = agent.generate_with_validation(
    prompt="Generate JSON with keys: name, age, city",
    validator=validate_json,
    max_attempts=3
)
```

### Resource Exhaustion
```python
result = agent.execute_with_fallback(
    primary_operation=expensive_llm_call,
    fallback_operation=cheaper_alternative,
    fallback_on=[TokenLimitError, QuotaExceededError]
)
```

## Troubleshooting

**Issue**: "OpenAI API key not found"
- Ensure `.env` file exists in project root (`../../.env`)
- Verify `OPENAI_API_KEY` is set correctly

**Issue**: "Module not found"
- Activate virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -e .`

**Issue**: "Too many retries"
- Check error type classification
- Adjust retry limits in configuration
- Implement circuit breaker for repeated failures

## Support

- Full documentation: See `README.md`
- Source code: Check `src/` directory
- Pattern comparison: See "Comparison with Related Patterns" section in README

---

**Ready to build resilient AI systems? Start with the basic example and progressively add advanced features as needed!**
