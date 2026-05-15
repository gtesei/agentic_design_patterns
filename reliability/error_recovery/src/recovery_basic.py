"""
Error Recovery Pattern: Basic Implementation

This example demonstrates basic error recovery with:
- Automatic retry with exponential backoff
- Error classification (transient vs. permanent)
- Multiple recovery strategies
- Error logging and tracking
"""


import sys

from pathlib import Path

ROOT_DIR = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "ssl_fix.py").exists()
)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from repo_support import configure_example

configure_example(__file__)


import json
import os
import random
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables


# --- Error Classification ---

class ErrorType(Enum):
    """Classification of errors for recovery strategy selection"""
    TRANSIENT = "transient"  # Temporary, retry immediately
    RETRYABLE = "retryable"  # Can be fixed, retry with modification
    PERMANENT = "permanent"  # Cannot be recovered
    UNKNOWN = "unknown"  # Needs investigation


class SimulatedError(Exception):
    """Base class for simulated errors"""
    pass


class TransientError(SimulatedError):
    """Temporary error (network timeout, rate limit, etc.)"""
    pass


class RetryableError(SimulatedError):
    """Error that can be fixed with modification"""
    pass


class PermanentError(SimulatedError):
    """Unrecoverable error"""
    pass


# --- Error Classifier ---

class ErrorClassifier:
    """Classify errors to determine appropriate recovery strategy"""

    # Keywords indicating error types
    TRANSIENT_KEYWORDS = [
        "timeout", "connection", "503", "429", "rate limit",
        "temporarily unavailable", "network", "reset"
    ]

    RETRYABLE_KEYWORDS = [
        "token limit", "invalid format", "validation failed",
        "quota", "bad request", "400"
    ]

    PERMANENT_KEYWORDS = [
        "unauthorized", "401", "403", "404", "not found",
        "invalid credentials", "authentication failed"
    ]

    @classmethod
    def classify(cls, error: Exception) -> ErrorType:
        """Classify an error based on its message and type"""
        error_str = str(error).lower()

        # Check by exception type first
        if isinstance(error, TransientError):
            return ErrorType.TRANSIENT
        if isinstance(error, RetryableError):
            return ErrorType.RETRYABLE
        if isinstance(error, PermanentError):
            return ErrorType.PERMANENT

        # Check by error message keywords
        if any(keyword in error_str for keyword in cls.TRANSIENT_KEYWORDS):
            return ErrorType.TRANSIENT
        if any(keyword in error_str for keyword in cls.RETRYABLE_KEYWORDS):
            return ErrorType.RETRYABLE
        if any(keyword in error_str for keyword in cls.PERMANENT_KEYWORDS):
            return ErrorType.PERMANENT

        return ErrorType.UNKNOWN


# --- Error Logger ---

class ErrorLogger:
    """Log errors with rich context for analysis"""

    def __init__(self):
        self.error_history = []

    def log_error(self, error: Exception, context: dict):
        """Log error with diagnostic context"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "classification": ErrorClassifier.classify(error).value,
            "context": context,
        }

        self.error_history.append(error_record)

        # Print formatted error
        print(f"\n❌ ERROR DETECTED:")
        print(f"   Type: {error_record['error_type']}")
        print(f"   Classification: {error_record['classification'].upper()}")
        print(f"   Message: {error_record['error_message']}")
        print(f"   Operation: {context.get('operation', 'unknown')}")
        print(f"   Attempt: {context.get('attempt', 'N/A')}")

    def log_recovery(self, strategy: str, success: bool, context: dict):
        """Log recovery attempt"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n🔧 RECOVERY ATTEMPT ({status}):")
        print(f"   Strategy: {strategy}")
        print(f"   Attempt: {context.get('attempt', 'N/A')}")
        if not success:
            print(f"   Reason: {context.get('reason', 'unknown')}")

    def get_error_stats(self) -> dict:
        """Get error statistics"""
        if not self.error_history:
            return {"total": 0}

        classifications = [e["classification"] for e in self.error_history]
        return {
            "total": len(self.error_history),
            "by_type": {
                classification: classifications.count(classification)
                for classification in set(classifications)
            },
            "recent_errors": self.error_history[-5:],
        }


# --- Error Recovery Agent ---

class ErrorRecoveryAgent:
    """Agent with multiple error recovery strategies"""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.logger = ErrorLogger()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def execute_with_recovery(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute operation with automatic error recovery

        Tries multiple strategies:
        1. Retry with exponential backoff (for transient errors)
        2. Retry with modification (for retryable errors)
        3. Fallback strategy (when retries fail)
        """
        print(f"\n{'='*70}")
        print(f"🚀 EXECUTING OPERATION: {operation_name}")
        print(f"{'='*70}")

        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"\n📍 Attempt {attempt}/{self.max_retries}")

                # Execute operation
                result = operation(*args, **kwargs)

                # Success!
                print(f"\n✅ SUCCESS on attempt {attempt}")
                return result

            except Exception as error:
                # Log the error
                context = {
                    "operation": operation_name,
                    "attempt": attempt,
                    "args": str(args)[:100],
                }
                self.logger.log_error(error, context)

                # Classify error
                error_type = ErrorClassifier.classify(error)

                # Check if this is the last attempt
                is_last_attempt = (attempt == self.max_retries)

                # Choose recovery strategy based on error type
                if error_type == ErrorType.PERMANENT:
                    print("\n🛑 Permanent error detected. Cannot recover.")
                    return self._fallback_strategy(operation_name, error)

                if error_type == ErrorType.TRANSIENT and not is_last_attempt:
                    # Retry with exponential backoff
                    self._retry_with_backoff(attempt)
                    continue

                if error_type == ErrorType.RETRYABLE and not is_last_attempt:
                    # Try to fix and retry
                    print("\n🔄 Retryable error - attempting modification...")
                    # In real implementation, modify parameters based on error
                    self._retry_with_backoff(attempt)
                    continue

                if is_last_attempt:
                    # Last attempt failed, use fallback
                    print(f"\n⚠️  All {self.max_retries} attempts exhausted.")
                    return self._fallback_strategy(operation_name, error)

        # Should not reach here
        return self._fallback_strategy(operation_name, Exception("Max retries exceeded"))

    def _retry_with_backoff(self, attempt: int):
        """Wait with exponential backoff before retry"""
        # Exponential backoff: 2^attempt seconds, max 10 seconds
        base_delay = 2 ** attempt
        max_delay = 10
        delay = min(base_delay, max_delay)

        # Add jitter (random 0-100% of delay)
        jitter = delay * random.random()
        total_delay = delay + jitter

        print(f"\n⏳ Waiting {total_delay:.2f}s before retry (backoff + jitter)...")
        time.sleep(total_delay)

    def _fallback_strategy(self, operation_name: str, error: Exception) -> dict:
        """Fallback strategy when recovery fails"""
        print("\n🔀 EXECUTING FALLBACK STRATEGY")
        print("   Using cached data or degraded response...")

        self.logger.log_recovery(
            strategy="fallback",
            success=True,
            context={"operation": operation_name}
        )

        return {
            "status": "partial_success",
            "message": "Operation failed, returning cached/degraded result",
            "error": str(error),
            "data": "Fallback result (cached or simplified version)",
        }


# --- Simulated Operations for Testing ---

class SimulatedOperations:
    """Simulated operations that can fail in different ways"""

    def __init__(self):
        self.attempt_count = {}

    def unreliable_api_call(self, endpoint: str) -> dict:
        """
        Simulates an unreliable API that fails with transient errors
        Succeeds on 3rd attempt
        """
        operation_id = f"api_{endpoint}"
        self.attempt_count[operation_id] = self.attempt_count.get(operation_id, 0) + 1

        current_attempt = self.attempt_count[operation_id]

        if current_attempt < 3:
            # Simulate transient failures
            errors = [
                "Connection timeout after 30 seconds",
                "503 Service Temporarily Unavailable",
                "Network connection reset by peer",
            ]
            raise TransientError(random.choice(errors))

        # Success on 3rd attempt
        return {
            "status": "success",
            "data": f"Data from {endpoint}",
            "timestamp": datetime.now().isoformat(),
        }

    def token_limit_operation(self, text: str) -> dict:
        """
        Simulates operation that hits token limits
        Succeeds with shorter input
        """
        operation_id = "token_limit"
        self.attempt_count[operation_id] = self.attempt_count.get(operation_id, 0) + 1

        current_attempt = self.attempt_count[operation_id]

        if current_attempt == 1 and len(text) > 100:
            raise RetryableError("Token limit exceeded: input too long")

        # Succeed on retry (simulating shortened input)
        return {
            "status": "success",
            "summary": f"Processed text with {len(text)} characters",
        }

    def authentication_operation(self) -> dict:
        """Simulates permanent authentication failure"""
        raise PermanentError("401 Unauthorized: Invalid API credentials")


# --- Main Demonstration ---

def main():
    """Demonstrate error recovery patterns"""

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         Error Recovery Pattern - Basic Implementation        ║
    ║                                                               ║
    ║  Demonstrates:                                                ║
    ║  • Automatic retry with exponential backoff                   ║
    ║  • Error classification (transient/retryable/permanent)       ║
    ║  • Multiple recovery strategies                               ║
    ║  • Error logging and tracking                                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    agent = ErrorRecoveryAgent(max_retries=3)
    simulator = SimulatedOperations()

    # Example 1: Transient Error Recovery
    print("\n" + "="*70)
    print("EXAMPLE 1: Transient Error (Network Timeout)")
    print("="*70)
    print("Scenario: API call fails with transient network issues")
    print("Expected: Success after 2-3 retries with backoff")

    result1 = agent.execute_with_recovery(
        operation=simulator.unreliable_api_call,
        operation_name="weather_api_call",
        endpoint="/api/weather"
    )
    print(f"\n📊 FINAL RESULT:")
    print(json.dumps(result1, indent=2))

    # Example 2: Retryable Error Recovery
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Retryable Error (Token Limit)")
    print("="*70)
    print("Scenario: Operation fails due to resource constraints")
    print("Expected: Success after modification (shortened input)")

    # Reset counter for new operation
    simulator.attempt_count = {}

    result2 = agent.execute_with_recovery(
        operation=simulator.token_limit_operation,
        operation_name="summarize_long_text",
        text="A" * 200  # Long text that will trigger error
    )
    print(f"\n📊 FINAL RESULT:")
    print(json.dumps(result2, indent=2))

    # Example 3: Permanent Error with Fallback
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Permanent Error (Authentication Failure)")
    print("="*70)
    print("Scenario: Operation fails with unrecoverable error")
    print("Expected: Immediate fallback strategy (no retries)")

    result3 = agent.execute_with_recovery(
        operation=simulator.authentication_operation,
        operation_name="protected_api_call"
    )
    print(f"\n📊 FINAL RESULT:")
    print(json.dumps(result3, indent=2))

    # Error Statistics
    print("\n\n" + "="*70)
    print("ERROR STATISTICS")
    print("="*70)
    stats = agent.logger.get_error_stats()
    print(json.dumps(stats, indent=2))

    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    Examples Complete!                         ║
    ║                                                               ║
    ║  The Error Recovery agent demonstrated:                       ║
    ║  ✓ Transient error recovery with exponential backoff         ║
    ║  ✓ Retryable error handling with parameter modification      ║
    ║  ✓ Permanent error detection with fallback strategy          ║
    ║  ✓ Comprehensive error logging and classification            ║
    ║                                                               ║
    ║  Key Takeaways:                                               ║
    ║  • Different errors need different recovery strategies        ║
    ║  • Exponential backoff prevents overwhelming services         ║
    ║  • Always have a fallback for unrecoverable errors           ║
    ║  • Log everything for debugging and pattern analysis          ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
