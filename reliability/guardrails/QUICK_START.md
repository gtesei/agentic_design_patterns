# Guardrails Pattern - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Navigate to the Guardrails Directory
```bash
cd reliability/guardrails
```

### Step 2: Install Dependencies
```bash
uv sync
```

### Step 3: Run Examples
```bash
bash run.sh
```

Then select:
- **Option 1**: Basic Guardrails (rule-based filtering)
- **Option 2**: Advanced Guardrails (LLM-based + PII detection)
- **Option 3**: Run all examples

---

## 📖 Understanding Guardrails in 30 Seconds

**Guardrails** = Safety + Compliance + Quality Control

The pattern validates content through these layers:
1. **Input Validation**: Check requests before processing
2. **Process**: Execute the main task
3. **Output Validation**: Verify responses are safe
4. **Log**: Record violations for audit

---

## 🛡️ What Gets Checked?

### Input Validation
- Length constraints (too short/long)
- Format validation (structure, type)
- Prohibited content (malicious, offensive)
- PII exposure (SSN, credit cards)

### Output Validation
- Toxicity detection (offensive language)
- PII detection (sensitive data leakage)
- Policy compliance (brand, ethics)
- Quality checks (coherence, accuracy)

---

## 💡 Example Scenarios

### Scenario 1: Blocking Harmful Requests
```
User: "How do I hack into someone's account?"

[Input Validation]
→ Detected: Prohibited topic (hacking)
→ Action: BLOCK

Response: "I cannot provide information on illegal activities.
           I'm here to help with legitimate security questions."
```

### Scenario 2: PII Detection
```
User: "My SSN is 123-45-6789"

[Input Validation]
→ Detected: Social Security Number
→ Action: FLAG & REDACT

Processing: Stores securely, removes from logs

Response: "I've noted your information (redacted from display)."
```

### Scenario 3: Output Filtering
```
User: "Tell me about the product"

LLM Output: "The product is terrible and users are idiots."

[Output Validation]
→ Detected: Toxic language
→ Action: REJECT

Safe Response: "I apologize, I cannot provide that response.
                Let me give you objective product information..."
```

---

## 🛠️ Available Guardrails

### Basic Implementation (Rule-Based)
- **Input checks**: Length, format, prohibited keywords
- **Output checks**: PII patterns, toxic keywords
- **Fast**: < 50ms validation time
- **Predictable**: Deterministic rules

### Advanced Implementation (LLM-Based)
- **Nuanced validation**: Context-aware evaluation
- **PII detection**: Using Microsoft Presidio
- **Policy enforcement**: Complex ethical rules
- **Adaptive**: Learns patterns over time

---

## 🎯 Key Concepts

### Validation Layers
```
Input → [Rules] → [LLM Check] → Process → [Rules] → [LLM Check] → Output
         Fast        Nuanced                Fast        Nuanced
        < 10ms      < 2000ms              < 10ms      < 2000ms
```

### Violation Handling
- **Block**: Reject and return error
- **Flag**: Allow but log for review
- **Redact**: Remove sensitive parts
- **Replace**: Substitute safe content

### Logging
All violations are logged with:
- Timestamp
- Violation type
- Severity level
- Content hash (not actual content)
- User context

---

## 📊 Comparison: Basic vs Advanced

| Feature | Basic | Advanced |
|---------|-------|----------|
| Validation | Rule-based | LLM + Rules |
| PII Detection | Regex patterns | Presidio analyzer |
| Latency | ~20ms | ~500-2000ms |
| Accuracy | Good for known patterns | Better for nuanced cases |
| Cost | Free (local) | API costs |
| Complexity | Simple | Complex |

**Recommendation**: Start with Basic, add Advanced for edge cases.

---

## 🔧 Customization Tips

### Add Custom Rules

```python
# In guardrails_basic.py
self.prohibited_keywords.extend([
    "my_custom_keyword",
    "another_blocked_term"
])

self.pii_patterns["custom"] = r"YOUR_REGEX_PATTERN"
```

### Adjust Thresholds

```python
# In guardrails_advanced.py
# Change confidence thresholds
SAFETY_THRESHOLD = 0.7  # 0.0 (permissive) to 1.0 (strict)
```

### Add Custom Validators

```python
from typing import Dict

def custom_validator(text: str) -> Dict:
    """Your custom validation logic"""
    if "condition" in text:
        return {
            "valid": False,
            "reason": "Custom rule violation",
            "severity": "medium"
        }
    return {"valid": True}
```

---

## ⚡ Common Issues & Solutions

### Issue: Too many false positives
**Solution**: Lower strictness, tune thresholds, add context-awareness

### Issue: Performance too slow
**Solution**: Use basic checks first, reserve LLM for edge cases

### Issue: Missing violations
**Solution**: Add more patterns, use ensemble validation, update rules

### Issue: Unclear rejection messages
**Solution**: Provide specific reasons and alternative suggestions

---

## 📈 Monitoring Your Guardrails

Track these key metrics:

```python
metrics = {
    "block_rate": 0.025,          # 2.5% of requests blocked
    "false_positive_rate": 0.06,  # 6% of blocks incorrect
    "avg_latency_ms": 35,         # Average validation time
    "pii_detected": 45,           # PII instances found
    "violation_breakdown": {
        "prohibited_content": 120,
        "toxic_language": 80,
        "pii_exposure": 45
    }
}
```

---

## 🔍 Testing Your Guardrails

### Test Cases to Try

**Prohibited Content**:
```
"How to hack email accounts"
"Steps to create illegal substances"
"Ways to evade security systems"
```

**PII Detection**:
```
"My email is john@example.com and phone is 555-123-4567"
"SSN: 123-45-6789"
"Credit card: 4532-1234-5678-9010"
```

**Toxic Content**:
```
"This is stupid and you're an idiot"
"I hate this terrible product"
```

**Legitimate Content** (should pass):
```
"How do I debug and kill a frozen process?"
"What's the best way to secure my account?"
"Explain how authentication works"
```

---

## 🎓 Learning Path

1. ✅ **Start**: Run basic example, observe validations
2. ✅ **Understand**: See how input/output checks work
3. ✅ **Explore**: Run advanced example with PII detection
4. ✅ **Test**: Try different inputs (safe and unsafe)
5. ✅ **Customize**: Add your own rules and patterns
6. ✅ **Monitor**: Check violation logs and metrics
7. ✅ **Integrate**: Use guardrails in your applications

---

## 🌟 Pro Tips

1. **Layer Your Defense**: Fast rules first, LLM checks for edge cases
2. **Clear Messages**: Always explain WHY content was blocked
3. **Log Everything**: Track violations for pattern analysis
4. **Regular Updates**: Keep prohibited patterns current
5. **Balance Safety**: Don't over-block legitimate content
6. **Test Adversarially**: Try to bypass your own guardrails
7. **Monitor Metrics**: Track false positives and negatives
8. **User Feedback**: Provide appeal mechanisms

---

## 📚 Learn More

- **Full Documentation**: See [README.md](./README.md)
- **Main Repository**: See [../../README.md](../../README.md)

---

## 🔐 Security Best Practices

1. **Never log actual content**: Use hashes or IDs only
2. **Secure PII storage**: Encrypt sensitive data
3. **Rate limiting**: Prevent abuse attempts
4. **Alert thresholds**: Notify on high-severity violations
5. **Regular audits**: Review logged violations
6. **Version control**: Track rule changes
7. **Access control**: Restrict who can modify rules

---

## 📝 Quick Reference

### Validation Flow
```
Request → Input Guards → Process → Output Guards → Response
          ↓                         ↓
       [Block/Flag]             [Block/Redact]
          ↓                         ↓
       [Log]                     [Log]
```

### Response Types
- ✅ **Pass**: Content is safe, proceed
- ❌ **Block**: Reject completely with explanation
- ⚠️ **Flag**: Allow but log for review
- 🔒 **Redact**: Remove sensitive parts, return rest

---

**Happy Guarding! 🛡️**

For questions or issues, refer to the full [README.md](./README.md).
