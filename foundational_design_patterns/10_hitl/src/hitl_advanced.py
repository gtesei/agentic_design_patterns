"""
Advanced Human-in-the-Loop (HITL) Implementation with Risk-Based Checkpoints

This module demonstrates an advanced HITL workflow with:
- Risk scoring system (low/medium/high)
- Conditional approval requirements based on risk level
- Multiple checkpoint types (financial, compliance, content quality)
- Comprehensive audit trail
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv


# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CheckpointType(Enum):
    """Types of checkpoints in the workflow."""
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    CONTENT_QUALITY = "content_quality"
    CUSTOMER_IMPACT = "customer_impact"


@dataclass
class RiskAssessment:
    """Risk assessment for a task."""
    risk_level: RiskLevel
    checkpoint_type: CheckpointType
    risk_factors: List[str]
    score: float  # 0-100
    requires_approval: bool
    justification: str


@dataclass
class ApprovalDecision:
    """Record of an approval decision."""
    timestamp: str
    decision: str  # approve, reject, escalate
    approver: str
    risk_assessment: Dict
    feedback: Optional[str] = None
    task_details: Optional[Dict] = None


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(title)}{Colors.ENDC}")


def print_risk_level(risk_level: RiskLevel):
    """Print risk level with appropriate color."""
    colors = {
        RiskLevel.LOW: Colors.GREEN,
        RiskLevel.MEDIUM: Colors.YELLOW,
        RiskLevel.HIGH: Colors.RED
    }
    color = colors.get(risk_level, Colors.ENDC)
    print(f"{color}{Colors.BOLD}Risk Level: {risk_level.value.upper()}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}{Colors.BOLD}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.ENDC}")


class RiskAnalyzer:
    """Analyzes tasks and assigns risk scores."""

    def __init__(self, api_key: str):
        """Initialize the risk analyzer."""
        self.client = OpenAI(api_key=api_key)

    def assess_financial_risk(self, amount: float, description: str) -> RiskAssessment:
        """Assess risk for financial transactions."""
        risk_factors = []
        score = 0

        # Amount-based risk
        if amount > 10000:
            risk_factors.append(f"High transaction amount: ${amount:,.2f}")
            score += 60
        elif amount > 1000:
            risk_factors.append(f"Moderate transaction amount: ${amount:,.2f}")
            score += 30
        else:
            risk_factors.append(f"Low transaction amount: ${amount:,.2f}")
            score += 10

        # Description-based risk (using AI)
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial risk analyst. Analyze the transaction description and provide a risk score from 0-40 and list any concerning factors."},
                    {"role": "user", "content": f"Transaction: {description}\nAmount: ${amount:,.2f}\nProvide risk score (0-40) and factors in JSON format: {{\"score\": <number>, \"factors\": [<list of strings>]}}"}
                ],
                temperature=0.3,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            score += result.get("score", 0)
            risk_factors.extend(result.get("factors", []))
        except Exception as e:
            print_warning(f"AI risk analysis failed: {str(e)}")
            score += 20
            risk_factors.append("Unable to perform detailed risk analysis")

        # Determine risk level
        if score >= 70:
            risk_level = RiskLevel.HIGH
        elif score >= 40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return RiskAssessment(
            risk_level=risk_level,
            checkpoint_type=CheckpointType.FINANCIAL,
            risk_factors=risk_factors,
            score=min(score, 100),
            requires_approval=risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH],
            justification=f"Financial transaction risk assessment based on amount and description analysis"
        )

    def assess_compliance_risk(self, content: str, regulations: List[str]) -> RiskAssessment:
        """Assess compliance risk for content."""
        risk_factors = []
        score = 0

        try:
            regulations_str = ", ".join(regulations)
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a compliance officer. Analyze content for potential regulatory violations."},
                    {"role": "user", "content": f"Content: {content}\n\nRegulations to check: {regulations_str}\n\nProvide risk assessment in JSON: {{\"score\": <0-100>, \"violations\": [<list>], \"concerns\": [<list>]}}"}
                ],
                temperature=0.2,
                max_tokens=300
            )

            result_text = response.choices[0].message.content.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()

            result = json.loads(result_text)
            score = result.get("score", 50)
            risk_factors.extend(result.get("violations", []))
            risk_factors.extend(result.get("concerns", []))

        except Exception as e:
            print_warning(f"Compliance analysis failed: {str(e)}")
            score = 50
            risk_factors.append("Unable to perform detailed compliance analysis")

        # Determine risk level
        if score >= 70:
            risk_level = RiskLevel.HIGH
        elif score >= 40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return RiskAssessment(
            risk_level=risk_level,
            checkpoint_type=CheckpointType.COMPLIANCE,
            risk_factors=risk_factors,
            score=score,
            requires_approval=risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH],
            justification=f"Compliance risk assessment for {regulations_str}"
        )

    def assess_customer_impact_risk(self, action: str, affected_customers: int) -> RiskAssessment:
        """Assess risk based on customer impact."""
        risk_factors = []
        score = 0

        # Customer count risk
        if affected_customers > 1000:
            risk_factors.append(f"High customer impact: {affected_customers:,} customers")
            score += 60
        elif affected_customers > 100:
            risk_factors.append(f"Moderate customer impact: {affected_customers:,} customers")
            score += 35
        else:
            risk_factors.append(f"Low customer impact: {affected_customers:,} customers")
            score += 15

        # Action severity
        high_risk_keywords = ['cancel', 'delete', 'terminate', 'suspend', 'charge', 'penalty']
        medium_risk_keywords = ['change', 'update', 'modify', 'notify']

        action_lower = action.lower()
        if any(keyword in action_lower for keyword in high_risk_keywords):
            risk_factors.append("Action contains high-impact keywords")
            score += 30
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            risk_factors.append("Action contains moderate-impact keywords")
            score += 15

        # Determine risk level
        if score >= 70:
            risk_level = RiskLevel.HIGH
        elif score >= 40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        return RiskAssessment(
            risk_level=risk_level,
            checkpoint_type=CheckpointType.CUSTOMER_IMPACT,
            risk_factors=risk_factors,
            score=score,
            requires_approval=risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH],
            justification=f"Customer impact assessment for action affecting {affected_customers:,} customers"
        )


class AdvancedHITLWorkflow:
    """Advanced HITL workflow with risk-based approvals."""

    def __init__(self, risk_analyzer: RiskAnalyzer):
        """Initialize the advanced HITL workflow."""
        self.risk_analyzer = risk_analyzer
        self.audit_log: List[ApprovalDecision] = []

    def display_risk_assessment(self, assessment: RiskAssessment):
        """Display detailed risk assessment."""
        print_section("🔍 RISK ASSESSMENT")

        print_risk_level(assessment.risk_level)
        print(f"{Colors.BOLD}Checkpoint Type:{Colors.ENDC} {assessment.checkpoint_type.value}")
        print(f"{Colors.BOLD}Risk Score:{Colors.ENDC} {assessment.score:.1f}/100")
        print(f"{Colors.BOLD}Requires Approval:{Colors.ENDC} {'Yes' if assessment.requires_approval else 'No'}")

        print(f"\n{Colors.BOLD}Risk Factors:{Colors.ENDC}")
        for factor in assessment.risk_factors:
            print(f"  • {factor}")

        print(f"\n{Colors.BOLD}Justification:{Colors.ENDC}")
        print(f"  {assessment.justification}")

    def get_approval(self, assessment: RiskAssessment, task_details: Dict[str, Any]) -> ApprovalDecision:
        """Get approval decision from human."""
        print_section("⚠️  APPROVAL REQUIRED")

        # Display task details
        print(f"\n{Colors.BOLD}Task Details:{Colors.ENDC}")
        for key, value in task_details.items():
            print(f"  {key}: {value}")

        print(f"\n{Colors.YELLOW}{Colors.BOLD}Approval Options:{Colors.ENDC}")
        print(f"  {Colors.GREEN}[A]{Colors.ENDC} Approve - Proceed with the action")
        print(f"  {Colors.RED}[R]{Colors.ENDC} Reject - Block the action")
        print(f"  {Colors.YELLOW}[E]{Colors.ENDC} Escalate - Send to higher authority")

        while True:
            decision = input(f"\n{Colors.BOLD}Your decision [A/R/E]:{Colors.ENDC} ").strip().upper()

            if decision in ['A', 'APPROVE']:
                feedback = input(f"{Colors.BOLD}Approval notes (optional):{Colors.ENDC} ").strip()
                return ApprovalDecision(
                    timestamp=datetime.now().isoformat(),
                    decision='approve',
                    approver='human_reviewer',
                    risk_assessment=asdict(assessment),
                    feedback=feedback or "Approved without additional notes",
                    task_details=task_details
                )
            elif decision in ['R', 'REJECT']:
                reason = input(f"{Colors.BOLD}Rejection reason:{Colors.ENDC} ").strip()
                return ApprovalDecision(
                    timestamp=datetime.now().isoformat(),
                    decision='reject',
                    approver='human_reviewer',
                    risk_assessment=asdict(assessment),
                    feedback=reason or "Rejected without specific reason",
                    task_details=task_details
                )
            elif decision in ['E', 'ESCALATE']:
                notes = input(f"{Colors.BOLD}Escalation notes:{Colors.ENDC} ").strip()
                return ApprovalDecision(
                    timestamp=datetime.now().isoformat(),
                    decision='escalate',
                    approver='human_reviewer',
                    risk_assessment=asdict(assessment),
                    feedback=notes or "Escalated for higher-level review",
                    task_details=task_details
                )
            else:
                print_error("Invalid option. Please choose A, R, or E.")

    def process_with_checkpoint(self, assessment: RiskAssessment,
                               task_details: Dict[str, Any],
                               auto_approve_low_risk: bool = True) -> bool:
        """
        Process a task with risk-based checkpoint.

        Returns:
            bool: True if approved, False otherwise
        """
        self.display_risk_assessment(assessment)

        # Auto-approve low-risk items if enabled
        if assessment.risk_level == RiskLevel.LOW and auto_approve_low_risk:
            print_section("✓ AUTO-APPROVED")
            print_success("Low-risk item automatically approved")

            decision = ApprovalDecision(
                timestamp=datetime.now().isoformat(),
                decision='approve',
                approver='system_auto',
                risk_assessment=asdict(assessment),
                feedback="Automatically approved - low risk",
                task_details=task_details
            )
            self.audit_log.append(decision)
            return True

        # Require human approval for medium/high risk
        if assessment.requires_approval:
            decision = self.get_approval(assessment, task_details)
            self.audit_log.append(decision)

            if decision.decision == 'approve':
                print_success("Action approved by human reviewer")
                return True
            elif decision.decision == 'reject':
                print_error("Action rejected by human reviewer")
                return False
            else:  # escalate
                print_warning("Action escalated to higher authority")
                return False

        return True

    def show_audit_trail(self):
        """Display comprehensive audit trail."""
        print_section("📋 COMPREHENSIVE AUDIT TRAIL")

        if not self.audit_log:
            print("No decisions recorded.")
            return

        for i, decision in enumerate(self.audit_log, 1):
            print(f"\n{Colors.BOLD}{'─'*80}{Colors.ENDC}")
            print(f"{Colors.BOLD}Decision #{i}{Colors.ENDC}")
            print(f"Timestamp: {decision.timestamp}")
            print(f"Decision: {decision.decision.upper()}")
            print(f"Approver: {decision.approver}")
            print(f"Risk Level: {decision.risk_assessment['risk_level']}")
            print(f"Risk Score: {decision.risk_assessment['score']:.1f}/100")
            print(f"Checkpoint: {decision.risk_assessment['checkpoint_type']}")

            if decision.feedback:
                print(f"Feedback: {decision.feedback}")

            if decision.task_details:
                print(f"{Colors.BOLD}Task Details:{Colors.ENDC}")
                for key, value in decision.task_details.items():
                    print(f"  {key}: {value}")

    def export_audit_log(self, filepath: str):
        """Export audit log to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump([asdict(d) for d in self.audit_log], f, indent=2)
            print_success(f"Audit log exported to: {filepath}")
        except Exception as e:
            print_error(f"Failed to export audit log: {str(e)}")


def run_financial_scenario(workflow: AdvancedHITLWorkflow, analyzer: RiskAnalyzer):
    """Demonstrate financial approval workflow."""
    print_header("SCENARIO: EXPENSE APPROVAL WORKFLOW")

    expenses = [
        (500, "Office supplies and equipment"),
        (2500, "Conference attendance and travel"),
        (15000, "New server infrastructure purchase"),
    ]

    for amount, description in expenses:
        print_section(f"Processing Expense: ${amount:,.2f}")
        print(f"Description: {description}")

        # Assess risk
        assessment = analyzer.assess_financial_risk(amount, description)

        # Process with checkpoint
        task_details = {
            "Amount": f"${amount:,.2f}",
            "Description": description,
            "Department": "Engineering",
            "Requested By": "John Doe"
        }

        approved = workflow.process_with_checkpoint(assessment, task_details)

        if approved:
            print_success(f"Expense of ${amount:,.2f} has been processed")
        else:
            print_error(f"Expense of ${amount:,.2f} was not approved")

        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")


def run_customer_impact_scenario(workflow: AdvancedHITLWorkflow, analyzer: RiskAnalyzer):
    """Demonstrate customer impact workflow."""
    print_header("SCENARIO: CUSTOMER NOTIFICATION WORKFLOW")

    actions = [
        ("Send product update notification", 50),
        ("Notify about pricing change", 500),
        ("Send service termination notice", 2000),
    ]

    for action, customer_count in actions:
        print_section(f"Processing Action: {action}")
        print(f"Affected Customers: {customer_count:,}")

        # Assess risk
        assessment = analyzer.assess_customer_impact_risk(action, customer_count)

        # Process with checkpoint
        task_details = {
            "Action": action,
            "Affected Customers": f"{customer_count:,}",
            "Scheduled For": "Next 24 hours",
            "Category": "Customer Communication"
        }

        approved = workflow.process_with_checkpoint(assessment, task_details)

        if approved:
            print_success(f"Customer action approved: {action}")
        else:
            print_error(f"Customer action blocked: {action}")

        input(f"\n{Colors.BOLD}Press Enter to continue...{Colors.ENDC}")


def main():
    """Main function to demonstrate advanced HITL workflow."""
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment variables.")
        print(f"Please ensure .env file exists at: {env_path}")
        sys.exit(1)

    # Initialize components
    analyzer = RiskAnalyzer(api_key)
    workflow = AdvancedHITLWorkflow(analyzer)

    print_header("ADVANCED HITL WITH RISK-BASED CHECKPOINTS")
    print("\nThis demo shows risk-based approval workflows.")
    print("Low-risk items are auto-approved, medium/high-risk require human review.\n")

    # Choose scenario
    scenarios = {
        "1": ("Financial Expense Approval", run_financial_scenario),
        "2": ("Customer Impact Assessment", run_customer_impact_scenario),
    }

    print_section("Available Scenarios")
    for key, (name, _) in scenarios.items():
        print(f"{key}. {name}")

    while True:
        choice = input(f"\n{Colors.BOLD}Select scenario (1-2) or 'q' to quit:{Colors.ENDC} ").strip()

        if choice.lower() == 'q':
            print("\nGoodbye!")
            sys.exit(0)

        if choice in scenarios:
            _, scenario_func = scenarios[choice]
            scenario_func(workflow, analyzer)
            break
        else:
            print_error("Invalid choice. Please select 1 or 2.")

    # Show audit trail
    workflow.show_audit_trail()

    # Export audit log
    audit_file = Path(__file__).parent / "audit_log.json"
    workflow.export_audit_log(str(audit_file))

    # Summary
    print_section("📊 WORKFLOW SUMMARY")
    total_decisions = len(workflow.audit_log)
    approved = sum(1 for d in workflow.audit_log if d.decision == 'approve')
    rejected = sum(1 for d in workflow.audit_log if d.decision == 'reject')
    escalated = sum(1 for d in workflow.audit_log if d.decision == 'escalate')
    auto_approved = sum(1 for d in workflow.audit_log if d.approver == 'system_auto')

    print(f"Total decisions: {total_decisions}")
    print(f"  {Colors.GREEN}Approved:{Colors.ENDC} {approved} ({auto_approved} auto-approved)")
    print(f"  {Colors.RED}Rejected:{Colors.ENDC} {rejected}")
    print(f"  {Colors.YELLOW}Escalated:{Colors.ENDC} {escalated}")


if __name__ == "__main__":
    main()
