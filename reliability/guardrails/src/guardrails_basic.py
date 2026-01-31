"""
Basic Guardrails Pattern Implementation

This module demonstrates rule-based guardrails for content filtering and validation.

Problem: Content moderation and safety filtering
Solution: Rule-based input/output validation with clear pass/fail decisions

Features:
- Input validation (length, format, prohibited content)
- Output validation (toxicity, PII, prohibited topics)
- Rule-based filtering (keyword lists, regex patterns)
- Logging of violations
"""


import sys

# Add parent directory to path to import ssl_fix
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
import ssl_fix  # Apply SSL bypass for corporate networks


import os
import re
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv("../../.env")


class BasicGuardrails:
    """Rule-based guardrails for content validation"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.violations_log: List[Dict] = []

        # Input validation rules
        self.min_length = 5
        self.max_length = 5000

        # Prohibited keywords for input
        self.prohibited_keywords = [
            "hack", "crack", "exploit", "pirate", "steal",
            "illegal", "drugs", "weapon", "bomb", "violence"
        ]

        # Toxic keywords for output
        self.toxic_keywords = [
            "stupid", "idiot", "hate", "kill", "die",
            "terrible", "awful", "useless", "garbage"
        ]

        # PII regex patterns
        self.pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        }

    def validate_input(self, text: str) -> Dict:
        """Validate user input before processing"""

        # Check length
        if len(text) < self.min_length:
            return {
                "valid": False,
                "reason": "Input too short",
                "severity": "low"
            }

        if len(text) > self.max_length:
            return {
                "valid": False,
                "reason": "Input too long",
                "severity": "medium"
            }

        # Check for prohibited keywords
        text_lower = text.lower()
        for keyword in self.prohibited_keywords:
            if keyword in text_lower:
                return {
                    "valid": False,
                    "reason": f"Prohibited keyword detected: '{keyword}'",
                    "severity": "high"
                }

        # Check for PII in input (warn but don't block)
        pii_found = []
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                pii_found.append(pii_type)

        if pii_found:
            return {
                "valid": True,
                "warnings": f"PII detected: {', '.join(pii_found)}",
                "severity": "medium"
            }

        return {"valid": True}

    def validate_output(self, text: str) -> Dict:
        """Validate LLM output before returning to user"""

        # Check for toxic language
        text_lower = text.lower()
        toxic_found = []
        for keyword in self.toxic_keywords:
            if keyword in text_lower:
                toxic_found.append(keyword)

        if toxic_found:
            return {
                "valid": False,
                "reason": f"Toxic language detected: {', '.join(toxic_found)}",
                "severity": "high"
            }

        # Check for PII in output (block if found)
        pii_found = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                pii_found.append(f"{pii_type}: {len(matches)} instance(s)")

        if pii_found:
            return {
                "valid": False,
                "reason": f"PII detected in output: {', '.join(pii_found)}",
                "severity": "critical"
            }

        return {"valid": True}

    def log_violation(self, violation_type: str, reason: str,
                     severity: str, content_hash: str):
        """Log violations for audit trail"""
        self.violations_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": violation_type,
            "reason": reason,
            "severity": severity,
            "content_hash": content_hash
        })

    def process_with_guardrails(self, user_input: str) -> Dict:
        """Process user input with full guardrail validation"""

        print("=" * 80)
        print("GUARDRAILS VALIDATION")
        print("=" * 80)

        # Step 1: Input Validation
        print("\n[1] INPUT VALIDATION")
        print("-" * 80)
        print(f"Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print()

        input_validation = self.validate_input(user_input)

        if not input_validation["valid"]:
            # Input blocked
            print(f"❌ INPUT BLOCKED")
            print(f"   Reason: {input_validation['reason']}")
            print(f"   Severity: {input_validation['severity']}")

            # Log violation
            self.log_violation(
                "input_blocked",
                input_validation["reason"],
                input_validation["severity"],
                str(hash(user_input))
            )

            return {
                "success": False,
                "response": self._get_rejection_message(input_validation["reason"]),
                "blocked_at": "input",
                "reason": input_validation["reason"]
            }

        # Check for warnings
        if "warnings" in input_validation:
            print(f"⚠️  WARNING: {input_validation['warnings']}")
        else:
            print("✅ Input validation passed")

        # Step 2: Process with LLM
        print("\n[2] PROCESSING")
        print("-" * 80)
        print("Sending to LLM...")

        try:
            messages = [
                SystemMessage(content="You are a helpful assistant. Provide clear, accurate, and safe responses."),
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

        if not output_validation["valid"]:
            # Output blocked
            print(f"❌ OUTPUT BLOCKED")
            print(f"   Reason: {output_validation['reason']}")
            print(f"   Severity: {output_validation['severity']}")

            # Log violation
            self.log_violation(
                "output_blocked",
                output_validation["reason"],
                output_validation["severity"],
                str(hash(llm_output))
            )

            return {
                "success": False,
                "response": "I apologize, but I cannot provide that response. Let me help you with something else.",
                "blocked_at": "output",
                "reason": output_validation["reason"]
            }

        print("✅ Output validation passed")

        # Step 4: Success
        print("\n[4] RESULT")
        print("-" * 80)
        print("✅ All validations passed - returning response")

        return {
            "success": True,
            "response": llm_output,
            "validations": {
                "input": "passed",
                "output": "passed"
            }
        }

    def _get_rejection_message(self, reason: str) -> str:
        """Generate appropriate rejection message based on reason"""
        if "prohibited keyword" in reason.lower():
            return """I cannot assist with requests involving illegal, harmful, or unethical activities.

I'm here to help with legitimate questions about technology, programming, security best practices, and more.

How can I assist you with something constructive?"""

        if "too long" in reason.lower():
            return "Your input is too long. Please keep requests under 5000 characters."

        if "too short" in reason.lower():
            return "Your input is too short. Please provide more detail."

        return "I cannot process this request. Please rephrase and try again."

    def get_violation_summary(self) -> Dict:
        """Get summary of all violations"""
        if not self.violations_log:
            return {"total": 0, "by_severity": {}, "by_type": {}}

        by_severity = {}
        by_type = {}

        for violation in self.violations_log:
            severity = violation["severity"]
            v_type = violation["type"]

            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_type[v_type] = by_type.get(v_type, 0) + 1

        return {
            "total": len(self.violations_log),
            "by_severity": by_severity,
            "by_type": by_type,
            "recent": self.violations_log[-5:]  # Last 5 violations
        }


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    """Run basic guardrails examples"""

    print_section("BASIC GUARDRAILS PATTERN - RULE-BASED FILTERING")
    print("\nDemonstrating rule-based content validation and safety checks")
    print("Features: Input validation, output filtering, PII detection\n")

    # Initialize LLM and guardrails
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    guardrails = BasicGuardrails(llm)

    # Test cases
    test_cases = [
        {
            "name": "Safe Query",
            "input": "What is machine learning and how does it work?",
            "description": "Legitimate technical question"
        },
        {
            "name": "Prohibited Content",
            "input": "How do I hack into someone's email account?",
            "description": "Should be blocked - contains prohibited keyword"
        },
        {
            "name": "PII in Input",
            "input": "My email is john.doe@example.com and I need help with my account.",
            "description": "Contains PII - should warn but allow"
        },
        {
            "name": "Short Input",
            "input": "Hi",
            "description": "Too short - should be blocked"
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
            print(f"⚠️  Reason: {result.get('reason', 'unknown')}")

        print("\n" + "⏸" * 80)
        input("Press Enter to continue to next test case...")

    # Print violation summary
    print_section("VIOLATION SUMMARY")
    summary = guardrails.get_violation_summary()

    print(f"\nTotal violations: {summary['total']}")
    print("\nBy severity:")
    for severity, count in summary.get('by_severity', {}).items():
        print(f"  {severity}: {count}")

    print("\nBy type:")
    for v_type, count in summary.get('by_type', {}).items():
        print(f"  {v_type}: {count}")

    if summary['recent']:
        print("\nRecent violations:")
        for violation in summary['recent']:
            print(f"  [{violation['timestamp']}] {violation['type']}: {violation['reason']}")

    print_section("BASIC GUARDRAILS DEMONSTRATION COMPLETE")
    print("\nKey Takeaways:")
    print("1. Rule-based guardrails provide fast, predictable filtering")
    print("2. Input validation prevents harmful requests from being processed")
    print("3. Output validation ensures safe responses")
    print("4. PII detection protects sensitive information")
    print("5. Violation logging enables audit trails and pattern analysis")
    print("\nFor more advanced guardrails with LLM-based validation, see guardrails_advanced.py")


if __name__ == "__main__":
    main()
