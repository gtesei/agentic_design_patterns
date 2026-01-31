"""
Advanced Guardrails Pattern Implementation

This module demonstrates LLM-based guardrails with sophisticated policy enforcement.

Problem: Complex policy compliance and nuanced content validation
Solution: Multi-layer validation with LLM-based evaluation and PII detection

Features:
- Complex policy enforcement (ethical guidelines, brand voice)
- LLM-based content evaluation (nuanced judgments)
- PII detection and redaction (using Microsoft Presidio)
- Multi-layer validation (input → intermediate → output)
- Adaptive thresholds based on context
- Detailed compliance reporting
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# PII detection using Microsoft Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    print("Warning: Presidio not available. PII detection will be limited.")
    PRESIDIO_AVAILABLE = False

# Load environment variables
load_dotenv("../../.env")


class AdvancedGuardrails:
    """LLM-based guardrails with sophisticated policy enforcement"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.violations_log: List[Dict] = []
        self.violation_counts = defaultdict(int)

        # Initialize Presidio for PII detection
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

        # Safety evaluation prompt
        self.safety_prompt = """Evaluate the following content for safety and compliance:

Content: {content}

Check for:
1. Harmful or dangerous information
2. Illegal activities or advice
3. Hate speech or discrimination
4. Unethical suggestions
5. Misinformation or false claims
6. Inappropriate or offensive content

Respond ONLY with valid JSON:
{{
    "safe": true/false,
    "violations": ["list of specific violations found"],
    "severity": "low/medium/high/critical",
    "confidence": 0.0-1.0,
    "explanation": "brief explanation"
}}"""

        # Policy compliance prompt
        self.policy_prompt = """Evaluate if this content complies with the following policies:

Content: {content}

Policies:
1. Professional tone and language
2. Accurate, factual information
3. Helpful and constructive
4. Respects privacy and confidentiality
5. Ethical and responsible
6. Brand-appropriate

Respond ONLY with valid JSON:
{{
    "compliant": true/false,
    "violations": ["list of policy violations"],
    "severity": "low/medium/high",
    "confidence": 0.0-1.0,
    "suggestions": ["how to fix violations"]
}}"""

    def detect_pii(self, text: str) -> Dict:
        """Detect PII using Microsoft Presidio"""
        if not PRESIDIO_AVAILABLE or not self.analyzer:
            return {"pii_found": False, "entities": [], "redacted": text}

        try:
            # Analyze text for PII
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                         "CREDIT_CARD", "US_SSN", "LOCATION", "DATE_TIME"]
            )

            if not results:
                return {"pii_found": False, "entities": [], "redacted": text}

            # Collect found entities
            entities = []
            for result in results:
                entities.append({
                    "type": result.entity_type,
                    "score": result.score,
                    "start": result.start,
                    "end": result.end
                })

            # Redact PII
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results
            )

            return {
                "pii_found": True,
                "entities": entities,
                "redacted": anonymized.text
            }
        except Exception as e:
            print(f"Warning: PII detection failed: {e}")
            return {"pii_found": False, "entities": [], "redacted": text}

    def evaluate_safety(self, content: str) -> Dict:
        """Evaluate content safety using LLM"""
        try:
            prompt = self.safety_prompt.format(content=content)
            messages = [
                SystemMessage(content="You are a content safety evaluator. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            result = json.loads(response.content)

            # Ensure all required fields
            return {
                "safe": result.get("safe", True),
                "violations": result.get("violations", []),
                "severity": result.get("severity", "low"),
                "confidence": result.get("confidence", 0.5),
                "explanation": result.get("explanation", "")
            }
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse safety evaluation: {e}")
            # Default to safe with low confidence
            return {
                "safe": True,
                "violations": [],
                "severity": "low",
                "confidence": 0.3,
                "explanation": "Evaluation parsing failed"
            }
        except Exception as e:
            print(f"Warning: Safety evaluation failed: {e}")
            return {
                "safe": True,
                "violations": [],
                "severity": "low",
                "confidence": 0.0,
                "explanation": str(e)
            }

    def evaluate_policy(self, content: str) -> Dict:
        """Evaluate policy compliance using LLM"""
        try:
            prompt = self.policy_prompt.format(content=content)
            messages = [
                SystemMessage(content="You are a policy compliance evaluator. Respond only with valid JSON."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            result = json.loads(response.content)

            return {
                "compliant": result.get("compliant", True),
                "violations": result.get("violations", []),
                "severity": result.get("severity", "low"),
                "confidence": result.get("confidence", 0.5),
                "suggestions": result.get("suggestions", [])
            }
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse policy evaluation: {e}")
            return {
                "compliant": True,
                "violations": [],
                "severity": "low",
                "confidence": 0.3,
                "suggestions": []
            }
        except Exception as e:
            print(f"Warning: Policy evaluation failed: {e}")
            return {
                "compliant": True,
                "violations": [],
                "severity": "low",
                "confidence": 0.0,
                "suggestions": []
            }

    def validate_input(self, text: str) -> Dict:
        """Multi-layer input validation"""
        print("  → Checking PII...")
        pii_result = self.detect_pii(text)

        print("  → Evaluating safety...")
        safety_result = self.evaluate_safety(text)

        print("  → Checking policy compliance...")
        policy_result = self.evaluate_policy(text)

        # Determine if input should be blocked
        should_block = False
        reasons = []

        # Block if PII found (configurable - here we warn but allow)
        if pii_result["pii_found"]:
            reasons.append(f"PII detected: {len(pii_result['entities'])} entities")

        # Block if safety fails
        if not safety_result["safe"] and safety_result["confidence"] > 0.6:
            should_block = True
            reasons.append(f"Safety violation: {', '.join(safety_result['violations'])}")

        # Block if policy fails with high severity
        if not policy_result["compliant"] and policy_result["severity"] in ["high", "critical"]:
            should_block = True
            reasons.append(f"Policy violation: {', '.join(policy_result['violations'])}")

        return {
            "valid": not should_block,
            "reasons": reasons,
            "pii": pii_result,
            "safety": safety_result,
            "policy": policy_result
        }

    def validate_output(self, text: str) -> Dict:
        """Multi-layer output validation"""
        print("  → Checking PII...")
        pii_result = self.detect_pii(text)

        print("  → Evaluating safety...")
        safety_result = self.evaluate_safety(text)

        print("  → Checking policy compliance...")
        policy_result = self.evaluate_policy(text)

        # Determine if output should be blocked
        should_block = False
        reasons = []

        # Block if PII found in output (more strict than input)
        if pii_result["pii_found"]:
            should_block = True
            reasons.append(f"PII exposure: {len(pii_result['entities'])} entities")

        # Block if safety fails
        if not safety_result["safe"] and safety_result["confidence"] > 0.5:
            should_block = True
            reasons.append(f"Safety violation: {', '.join(safety_result['violations'])}")

        # Block if policy fails
        if not policy_result["compliant"] and policy_result["severity"] in ["medium", "high", "critical"]:
            should_block = True
            reasons.append(f"Policy violation: {', '.join(policy_result['violations'])}")

        return {
            "valid": not should_block,
            "reasons": reasons,
            "pii": pii_result,
            "safety": safety_result,
            "policy": policy_result,
            "redacted_text": pii_result.get("redacted", text)
        }

    def log_violation(self, layer: str, validation_result: Dict, content_hash: str):
        """Log detailed violation information"""
        violation = {
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "content_hash": content_hash,
            "reasons": validation_result.get("reasons", []),
            "safety": validation_result.get("safety", {}),
            "policy": validation_result.get("policy", {}),
            "pii": {
                "found": validation_result.get("pii", {}).get("pii_found", False),
                "entity_count": len(validation_result.get("pii", {}).get("entities", []))
            }
        }

        self.violations_log.append(violation)

        # Track violation types
        for reason in validation_result.get("reasons", []):
            self.violation_counts[reason] += 1

    def process_with_guardrails(self, user_input: str) -> Dict:
        """Process user input with advanced multi-layer guardrails"""

        print("=" * 80)
        print("ADVANCED GUARDRAILS VALIDATION")
        print("=" * 80)

        # Step 1: Input Validation
        print("\n[1] INPUT VALIDATION")
        print("-" * 80)
        print(f"Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print()

        input_validation = self.validate_input(user_input)

        # Print validation results
        print("\nValidation Results:")
        print(f"  PII Found: {'Yes' if input_validation['pii']['pii_found'] else 'No'}")
        if input_validation['pii']['pii_found']:
            print(f"    Entities: {len(input_validation['pii']['entities'])}")
            for entity in input_validation['pii']['entities'][:3]:
                print(f"      - {entity['type']} (confidence: {entity['score']:.2f})")

        print(f"  Safety: {'✅ Safe' if input_validation['safety']['safe'] else '❌ Unsafe'}")
        if not input_validation['safety']['safe']:
            print(f"    Violations: {', '.join(input_validation['safety']['violations'])}")
            print(f"    Severity: {input_validation['safety']['severity']}")

        print(f"  Policy: {'✅ Compliant' if input_validation['policy']['compliant'] else '❌ Non-compliant'}")
        if not input_validation['policy']['compliant']:
            print(f"    Violations: {', '.join(input_validation['policy']['violations'])}")

        if not input_validation["valid"]:
            # Input blocked
            print("\n❌ INPUT BLOCKED")
            for reason in input_validation["reasons"]:
                print(f"   • {reason}")

            # Log violation
            self.log_violation("input", input_validation, str(hash(user_input)))

            return {
                "success": False,
                "response": self._get_rejection_message(input_validation),
                "blocked_at": "input",
                "validation": input_validation
            }

        print("\n✅ Input validation passed")

        # Step 2: Process with LLM
        print("\n[2] PROCESSING")
        print("-" * 80)
        print("Sending to LLM...")

        try:
            messages = [
                SystemMessage(content="You are a helpful, professional, and ethical assistant. Provide accurate, safe, and compliant responses."),
                HumanMessage(content=user_input)
            ]
            response = self.llm.invoke(messages)
            llm_output = response.content
            print(f"✅ LLM response generated ({len(llm_output)} chars)")
        except Exception as e:
            print(f"❌ LLM processing failed: {str(e)}")
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your request.",
                "error": str(e)
            }

        # Step 3: Output Validation
        print("\n[3] OUTPUT VALIDATION")
        print("-" * 80)
        print(f"Output: {llm_output[:100]}{'...' if len(llm_output) > 100 else ''}")
        print()

        output_validation = self.validate_output(llm_output)

        # Print validation results
        print("\nValidation Results:")
        print(f"  PII Found: {'Yes' if output_validation['pii']['pii_found'] else 'No'}")
        if output_validation['pii']['pii_found']:
            print(f"    Entities: {len(output_validation['pii']['entities'])}")
            for entity in output_validation['pii']['entities'][:3]:
                print(f"      - {entity['type']} (confidence: {entity['score']:.2f})")

        print(f"  Safety: {'✅ Safe' if output_validation['safety']['safe'] else '❌ Unsafe'}")
        if not output_validation['safety']['safe']:
            print(f"    Violations: {', '.join(output_validation['safety']['violations'])}")

        print(f"  Policy: {'✅ Compliant' if output_validation['policy']['compliant'] else '❌ Non-compliant'}")
        if not output_validation['policy']['compliant']:
            print(f"    Violations: {', '.join(output_validation['policy']['violations'])}")

        if not output_validation["valid"]:
            # Output blocked
            print("\n❌ OUTPUT BLOCKED")
            for reason in output_validation["reasons"]:
                print(f"   • {reason}")

            # Log violation
            self.log_violation("output", output_validation, str(hash(llm_output)))

            return {
                "success": False,
                "response": "I apologize, but I cannot provide that response. Let me help you differently.",
                "blocked_at": "output",
                "validation": output_validation
            }

        print("\n✅ Output validation passed")

        # Use redacted version if PII was found but passed
        final_output = output_validation.get("redacted_text", llm_output)

        # Step 4: Success
        print("\n[4] RESULT")
        print("-" * 80)
        print("✅ All validations passed - returning response")

        return {
            "success": True,
            "response": final_output,
            "validations": {
                "input": input_validation,
                "output": output_validation
            }
        }

    def _get_rejection_message(self, validation: Dict) -> str:
        """Generate contextual rejection message"""
        reasons = validation.get("reasons", [])

        if any("Safety violation" in r for r in reasons):
            return """I cannot assist with requests that may be harmful, dangerous, or unethical.

I'm here to help with constructive, legitimate questions and tasks.

How can I assist you with something positive and helpful?"""

        if any("Policy violation" in r for r in reasons):
            suggestions = validation.get("policy", {}).get("suggestions", [])
            message = "I cannot provide a response that violates our policies.\n"
            if suggestions:
                message += "\nSuggestions:\n"
                for suggestion in suggestions[:3]:
                    message += f"  • {suggestion}\n"
            return message

        if any("PII" in r for r in reasons):
            return """I detected sensitive personal information in your request.

For your privacy and security, please avoid sharing:
  • Social Security numbers
  • Credit card information
  • Phone numbers or email addresses
  • Other personally identifiable information

How can I help you without requiring sensitive data?"""

        return "I cannot process this request. Please rephrase and try again."

    def get_compliance_report(self) -> Dict:
        """Generate detailed compliance report"""
        if not self.violations_log:
            return {
                "total_violations": 0,
                "by_layer": {},
                "by_type": {},
                "recent": []
            }

        by_layer = defaultdict(int)
        by_type = defaultdict(int)

        for violation in self.violations_log:
            by_layer[violation["layer"]] += 1

            # Count violation types
            if violation["pii"]["found"]:
                by_type["pii_exposure"] += 1
            if not violation.get("safety", {}).get("safe", True):
                by_type["safety_violation"] += 1
            if not violation.get("policy", {}).get("compliant", True):
                by_type["policy_violation"] += 1

        return {
            "total_violations": len(self.violations_log),
            "by_layer": dict(by_layer),
            "by_type": dict(by_type),
            "violation_counts": dict(self.violation_counts),
            "recent": self.violations_log[-5:]
        }


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    """Run advanced guardrails examples"""

    print_section("ADVANCED GUARDRAILS PATTERN - LLM-BASED VALIDATION")
    print("\nDemonstrating sophisticated policy enforcement and PII detection")
    print("Features: LLM evaluation, Presidio PII detection, multi-layer validation\n")

    if not PRESIDIO_AVAILABLE:
        print("⚠️  Warning: Presidio not available. PII detection will be limited.")
        print("   Install with: pip install presidio-analyzer presidio-anonymizer\n")

    # Initialize LLM and guardrails
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    guardrails = AdvancedGuardrails(llm)

    # Test cases
    test_cases = [
        {
            "name": "Safe Technical Query",
            "input": "Explain how OAuth 2.0 authentication works.",
            "description": "Professional, technical question"
        },
        {
            "name": "Unethical Request",
            "input": "Write a fake positive review for my competitor's product to confuse customers.",
            "description": "Should be blocked - unethical and deceptive"
        },
        {
            "name": "PII Exposure",
            "input": "My name is John Smith, email john.smith@example.com, and my phone is 555-123-4567. I need help.",
            "description": "Contains multiple PII elements"
        },
        {
            "name": "Legitimate Security Question",
            "input": "What are best practices for securing a REST API?",
            "description": "Should pass - legitimate professional question"
        }
    ]

    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print_section(f"TEST CASE {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print()

        result = guardrails.process_with_guardrails(test_case["input"])

        print("\n" + "=" * 80)
        print("FINAL RESPONSE TO USER")
        print("=" * 80)
        print(result["response"])
        print()

        if not result["success"]:
            print(f"⚠️  Blocked at: {result.get('blocked_at', 'unknown')}")

        print("\n" + "⏸" * 80)
        if i < len(test_cases):
            input("Press Enter to continue to next test case...")

    # Print compliance report
    print_section("COMPLIANCE REPORT")
    report = guardrails.get_compliance_report()

    print(f"\nTotal violations: {report['total_violations']}")

    if report['by_layer']:
        print("\nViolations by layer:")
        for layer, count in report['by_layer'].items():
            print(f"  {layer}: {count}")

    if report['by_type']:
        print("\nViolations by type:")
        for v_type, count in report['by_type'].items():
            print(f"  {v_type}: {count}")

    if report['violation_counts']:
        print("\nDetailed violation counts:")
        for reason, count in report['violation_counts'].items():
            print(f"  {reason}: {count}")

    print_section("ADVANCED GUARDRAILS DEMONSTRATION COMPLETE")
    print("\nKey Takeaways:")
    print("1. LLM-based evaluation provides nuanced, context-aware validation")
    print("2. Presidio enables accurate PII detection and redaction")
    print("3. Multi-layer validation catches issues at input and output stages")
    print("4. Policy compliance ensures ethical and professional responses")
    print("5. Detailed logging supports compliance audits and pattern analysis")
    print("6. Adaptive thresholds balance safety with usability")
    print("\nFor production use, combine rule-based (fast) and LLM-based (nuanced) approaches")


if __name__ == "__main__":
    main()
