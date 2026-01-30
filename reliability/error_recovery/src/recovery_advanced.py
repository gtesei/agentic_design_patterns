"""
Error Recovery Pattern: Advanced Implementation

This example demonstrates advanced self-correction with:
- Self-correction loop: attempt → validate → diagnose → fix → retry
- Circuit breaker pattern implementation
- Multi-level fallback strategies
- Error pattern learning
- Verification step before returning results
- Rich error diagnostics and recovery history
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "../../..", ".env"))


# --- Circuit Breaker Implementation ---

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures

    When failures exceed threshold, circuit opens and rejects calls.
    After timeout, circuit enters half-open state to test recovery.
    """
    failure_threshold: int = 5
    timeout_duration: int = 30  # seconds
    success_threshold: int = 2  # successes needed to close from half-open

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    last_failure_time: Optional[float] = field(default=None, init=False)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.timeout_duration:
                print("\n🔄 Circuit breaker: OPEN → HALF_OPEN (testing recovery)")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                time_remaining = self.timeout_duration - (time.time() - self.last_failure_time)
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Retry in {time_remaining:.0f}s"
                )

        try:
            result = func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise

    def on_success(self):
        """Handle successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                print("\n✅ Circuit breaker: HALF_OPEN → CLOSED (recovered)")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            print("\n❌ Circuit breaker: HALF_OPEN → OPEN (still failing)")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            print(f"\n🔴 Circuit breaker: CLOSED → OPEN (threshold {self.failure_threshold} reached)")
            self.state = CircuitState.OPEN

    def get_state(self) -> str:
        """Get current circuit state"""
        return self.state.value


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


# --- Validation and Self-Correction ---

class ValidationError(Exception):
    """Error during output validation"""
    pass


class OutputValidator:
    """Validate LLM outputs and provide diagnostic feedback"""

    @staticmethod
    def validate_json(output: str) -> tuple[bool, Optional[str]]:
        """Validate JSON format"""
        try:
            json.loads(output)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON at position {e.pos}: {e.msg}"

    @staticmethod
    def validate_structure(output: str, required_keys: list[str]) -> tuple[bool, Optional[str]]:
        """Validate that output contains required structure"""
        try:
            data = json.loads(output)
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                return False, f"Missing required keys: {missing_keys}"
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Cannot parse JSON: {e.msg}"

    @staticmethod
    def validate_format(output: str, format_type: str) -> tuple[bool, Optional[str]]:
        """Validate output format"""
        validators = {
            "json": OutputValidator.validate_json,
            "non_empty": lambda x: (bool(x.strip()), "Output is empty" if not x.strip() else None),
        }

        validator = validators.get(format_type)
        if not validator:
            return True, None

        return validator(output)


# --- Error Pattern Learning ---

@dataclass
class ErrorPattern:
    """Track error patterns and successful recovery strategies"""
    error_type: str
    error_message: str
    recovery_strategy: str
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    avg_attempts_to_success: float = 0.0


class ErrorPatternLearner:
    """Learn from error patterns to improve recovery strategies"""

    def __init__(self):
        self.patterns: dict[str, ErrorPattern] = {}
        self.recovery_history: list[dict] = []

    def record_recovery_attempt(
        self,
        error_type: str,
        error_message: str,
        strategy: str,
        success: bool,
        attempts: int
    ):
        """Record a recovery attempt"""
        pattern_key = f"{error_type}:{strategy}"

        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = ErrorPattern(
                error_type=error_type,
                error_message=error_message,
                recovery_strategy=strategy
            )

        pattern = self.patterns[pattern_key]
        pattern.total_attempts += 1

        if success:
            pattern.success_count += 1
            # Update running average
            old_avg = pattern.avg_attempts_to_success
            old_count = pattern.success_count - 1
            pattern.avg_attempts_to_success = (
                (old_avg * old_count + attempts) / pattern.success_count
            )
        else:
            pattern.failure_count += 1

        # Store in history
        self.recovery_history.append({
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "strategy": strategy,
            "success": success,
            "attempts": attempts,
        })

    def get_best_strategy(self, error_type: str) -> Optional[str]:
        """Get the most successful recovery strategy for an error type"""
        relevant_patterns = [
            p for key, p in self.patterns.items()
            if p.error_type == error_type and p.success_count > 0
        ]

        if not relevant_patterns:
            return None

        # Sort by success rate, then by avg attempts
        relevant_patterns.sort(
            key=lambda p: (
                p.success_count / p.total_attempts,
                -p.avg_attempts_to_success
            ),
            reverse=True
        )

        return relevant_patterns[0].recovery_strategy

    def get_statistics(self) -> dict:
        """Get learning statistics"""
        if not self.patterns:
            return {"total_patterns": 0, "patterns": []}

        pattern_stats = []
        for pattern in self.patterns.values():
            success_rate = (
                pattern.success_count / pattern.total_attempts
                if pattern.total_attempts > 0 else 0
            )
            pattern_stats.append({
                "error_type": pattern.error_type,
                "strategy": pattern.recovery_strategy,
                "success_rate": f"{success_rate:.1%}",
                "total_attempts": pattern.total_attempts,
                "avg_attempts": f"{pattern.avg_attempts_to_success:.1f}",
            })

        return {
            "total_patterns": len(self.patterns),
            "total_recoveries": len(self.recovery_history),
            "patterns": pattern_stats,
        }


# --- Advanced Self-Correction Agent ---

class SelfCorrectionAgent:
    """
    Agent with advanced self-correction capabilities

    Features:
    - Validates outputs and self-corrects errors
    - Uses circuit breaker to prevent cascading failures
    - Learns from error patterns
    - Multi-level fallback strategies
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout_duration=20)
        self.validator = OutputValidator()
        self.learner = ErrorPatternLearner()

    def generate_with_validation(
        self,
        prompt: str,
        validator: Callable[[str], tuple[bool, Optional[str]]],
        max_attempts: int = 3,
    ) -> dict:
        """
        Generate output with automatic validation and self-correction

        Process:
        1. Generate output
        2. Validate output
        3. If invalid, diagnose issue and regenerate with correction prompt
        4. Repeat until valid or max attempts reached
        """
        print(f"\n{'='*70}")
        print("🤖 SELF-CORRECTION WITH VALIDATION")
        print(f"{'='*70}")

        errors = []
        operation_start = time.time()

        for attempt in range(1, max_attempts + 1):
            print(f"\n📍 Attempt {attempt}/{max_attempts}")

            try:
                # Generate output
                print("   Generating output...")
                output = self._generate(prompt)

                # Validate output
                print("   Validating output...")
                is_valid, error_msg = validator(output)

                if is_valid:
                    print("   ✅ Validation passed!")

                    # Record success
                    self.learner.record_recovery_attempt(
                        error_type="validation_error",
                        error_message=str(errors) if errors else "none",
                        strategy="self_correction",
                        success=True,
                        attempts=attempt
                    )

                    return {
                        "status": "success",
                        "output": output,
                        "attempts": attempt,
                        "errors_corrected": errors,
                        "duration": time.time() - operation_start,
                    }

                # Validation failed
                print(f"   ❌ Validation failed: {error_msg}")
                errors.append(error_msg)

                if attempt < max_attempts:
                    # Update prompt with correction guidance
                    prompt = self._create_correction_prompt(prompt, output, error_msg)
                    print("   🔧 Updating prompt with correction guidance...")

            except Exception as e:
                print(f"   ❌ Generation error: {str(e)}")
                errors.append(str(e))

        # All attempts failed
        print(f"\n⚠️  All {max_attempts} attempts failed")

        self.learner.record_recovery_attempt(
            error_type="validation_error",
            error_message=str(errors),
            strategy="self_correction",
            success=False,
            attempts=max_attempts
        )

        return {
            "status": "failed",
            "output": None,
            "attempts": max_attempts,
            "errors": errors,
            "duration": time.time() - operation_start,
        }

    def execute_with_circuit_breaker(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> dict:
        """Execute operation with circuit breaker protection"""
        print(f"\n{'='*70}")
        print(f"🔌 CIRCUIT BREAKER EXECUTION: {operation_name}")
        print(f"{'='*70}")
        print(f"Circuit state: {self.circuit_breaker.get_state().upper()}")

        try:
            result = self.circuit_breaker.call(operation, *args, **kwargs)
            return {
                "status": "success",
                "result": result,
                "circuit_state": self.circuit_breaker.get_state(),
            }
        except CircuitBreakerOpenError as e:
            print(f"\n⚠️  {str(e)}")
            # Use fallback when circuit is open
            return self._fallback_with_degradation(operation_name, e)
        except Exception as e:
            print(f"\n❌ Operation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "circuit_state": self.circuit_breaker.get_state(),
            }

    def _generate(self, prompt: str) -> str:
        """Generate output using LLM"""
        messages = [
            SystemMessage(content="You are a helpful assistant that produces accurate, well-formatted outputs."),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def _create_correction_prompt(
        self,
        original_prompt: str,
        failed_output: str,
        error_message: str
    ) -> str:
        """Create a prompt that guides the model to correct its error"""
        return f"""{original_prompt}

IMPORTANT CORRECTION NEEDED:
Your previous attempt produced invalid output.

Previous output:
```
{failed_output[:200]}...
```

Error: {error_message}

Please generate a corrected version that addresses this specific error.
Pay careful attention to the format requirements."""

    def _fallback_with_degradation(self, operation_name: str, error: Exception) -> dict:
        """Fallback strategy with graceful degradation"""
        print("\n🔀 EXECUTING FALLBACK STRATEGY")
        print("   Providing degraded/cached result...")

        return {
            "status": "degraded",
            "message": "Circuit breaker open - using fallback",
            "result": {
                "data": "Cached or simplified result",
                "quality": "degraded",
                "reason": str(error),
            },
            "circuit_state": self.circuit_breaker.get_state(),
        }


# --- Demonstration Scenarios ---

def demo_self_correction():
    """Demonstrate self-correction with validation"""
    print("\n" + "="*70)
    print("DEMO 1: Self-Correction Loop")
    print("="*70)
    print("Scenario: Generate JSON with specific structure")
    print("Expected: Auto-correct format errors through validation feedback")

    agent = SelfCorrectionAgent()

    # Validator for JSON with required keys
    def validate_person_json(output: str) -> tuple[bool, Optional[str]]:
        required_keys = ["name", "age", "city"]
        return agent.validator.validate_structure(output, required_keys)

    prompt = """Generate a JSON object representing a person with the following keys:
- name (string)
- age (number)
- city (string)

Output ONLY the JSON, no other text."""

    result = agent.generate_with_validation(
        prompt=prompt,
        validator=validate_person_json,
        max_attempts=3
    )

    print(f"\n📊 FINAL RESULT:")
    print(json.dumps(result, indent=2))


def demo_circuit_breaker():
    """Demonstrate circuit breaker pattern"""
    print("\n\n" + "="*70)
    print("DEMO 2: Circuit Breaker Pattern")
    print("="*70)
    print("Scenario: Repeated failures trigger circuit breaker")
    print("Expected: Circuit opens after threshold, then tests recovery")

    agent = SelfCorrectionAgent()

    # Simulated failing operation
    failure_count = 0

    def flaky_operation():
        nonlocal failure_count
        failure_count += 1

        # Fail first 3 times, then succeed
        if failure_count <= 3:
            raise Exception(f"Service unavailable (failure {failure_count})")

        return {"status": "success", "data": "Operation completed"}

    # Execute multiple times to trigger circuit breaker
    for i in range(6):
        print(f"\n--- Call {i+1} ---")
        result = agent.execute_with_circuit_breaker(
            operation=flaky_operation,
            operation_name="external_service_call"
        )

        print(f"Result status: {result['status']}")
        print(f"Circuit state: {result['circuit_state']}")

        # Wait a bit between calls
        if i < 5:
            time.sleep(2)


def demo_pattern_learning():
    """Demonstrate error pattern learning"""
    print("\n\n" + "="*70)
    print("DEMO 3: Error Pattern Learning")
    print("="*70)
    print("Scenario: Track recovery patterns to improve future strategies")

    agent = SelfCorrectionAgent()

    # Simulate various recovery scenarios
    scenarios = [
        ("timeout", "Connection timeout", "retry", True, 2),
        ("timeout", "Connection timeout", "retry", True, 1),
        ("token_limit", "Token limit exceeded", "reduce_input", True, 3),
        ("timeout", "Connection timeout", "fallback", False, 3),
        ("format_error", "Invalid JSON", "self_correction", True, 2),
        ("format_error", "Invalid JSON", "self_correction", True, 1),
    ]

    for error_type, error_msg, strategy, success, attempts in scenarios:
        agent.learner.record_recovery_attempt(
            error_type=error_type,
            error_message=error_msg,
            strategy=strategy,
            success=success,
            attempts=attempts
        )

    # Show learned patterns
    print("\n📚 LEARNED PATTERNS:")
    stats = agent.learner.get_statistics()
    print(json.dumps(stats, indent=2))

    # Get best strategy for specific error types
    print("\n💡 RECOMMENDED STRATEGIES:")
    for error_type in ["timeout", "format_error", "token_limit"]:
        best = agent.learner.get_best_strategy(error_type)
        print(f"   {error_type}: {best}")


# --- Main Demonstration ---

def main():
    """Run all advanced demonstrations"""

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║      Error Recovery Pattern - Advanced Implementation        ║
    ║                                                               ║
    ║  Demonstrates:                                                ║
    ║  • Self-correction loop with validation                       ║
    ║  • Circuit breaker pattern                                    ║
    ║  • Multi-level fallback strategies                            ║
    ║  • Error pattern learning                                     ║
    ║  • Rich error diagnostics                                     ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Run demonstrations
    demo_self_correction()
    demo_circuit_breaker()
    demo_pattern_learning()

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Examples Complete!                         ║
    ║                                                               ║
    ║  The Advanced Error Recovery agent demonstrated:              ║
    ║  ✓ Self-correction through validation feedback               ║
    ║  ✓ Circuit breaker preventing cascading failures             ║
    ║  ✓ Graceful degradation with fallback strategies             ║
    ║  ✓ Learning from error patterns for optimization             ║
    ║                                                               ║
    ║  Key Takeaways:                                               ║
    ║  • Validate outputs before returning to catch errors early    ║
    ║  • Circuit breakers protect against cascading failures        ║
    ║  • Self-correction can fix many errors automatically          ║
    ║  • Learning from patterns improves recovery over time         ║
    ║  • Always have fallback strategies for graceful degradation   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
