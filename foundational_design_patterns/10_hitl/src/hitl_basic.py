"""
Basic Human-in-the-Loop (HITL) Implementation

This module demonstrates a simple HITL workflow for content generation
with a human approval checkpoint before publishing.
"""

import os
import sys
from pathlib import Path
from typing import Optional
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


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*len(title)}{Colors.ENDC}")


def print_content(content: str):
    """Print generated content in a box."""
    lines = content.split('\n')
    max_length = max(len(line) for line in lines) + 4

    print(f"\n{Colors.BLUE}+{'-'*max_length}+{Colors.ENDC}")
    for line in lines:
        print(f"{Colors.BLUE}|{Colors.ENDC} {line.ljust(max_length-2)} {Colors.BLUE}|{Colors.ENDC}")
    print(f"{Colors.BLUE}+{'-'*max_length}+{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}{Colors.BOLD}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.ENDC}")


class ContentGenerator:
    """Simple content generator using OpenAI."""

    def __init__(self, api_key: str):
        """Initialize the content generator."""
        self.client = OpenAI(api_key=api_key)

    def generate_content(self, content_type: str, topic: str,
                        additional_instructions: str = "") -> str:
        """Generate content using OpenAI."""
        prompts = {
            "blog": f"Write a short blog post (200-300 words) about: {topic}. {additional_instructions}",
            "email": f"Write a professional email about: {topic}. {additional_instructions}",
            "social": f"Write a social media post (2-3 sentences) about: {topic}. {additional_instructions}",
            "tweet": f"Write a tweet (280 characters max) about: {topic}. {additional_instructions}"
        }

        prompt = prompts.get(content_type, prompts["blog"])

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional content writer. Create engaging, clear, and concise content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print_error(f"Failed to generate content: {str(e)}")
            return None


class HITLWorkflow:
    """Basic Human-in-the-Loop workflow for content generation."""

    def __init__(self, generator: ContentGenerator):
        """Initialize the HITL workflow."""
        self.generator = generator
        self.decision_log = []

    def get_human_decision(self, content: str, content_type: str) -> tuple[str, Optional[str]]:
        """
        Present content to human and get approval decision.

        Returns:
            tuple: (decision, feedback) where decision is 'approve', 'reject', or 'modify'
        """
        print_section("🔍 HUMAN REVIEW CHECKPOINT")
        print(f"\n{Colors.BOLD}Content Type:{Colors.ENDC} {content_type.upper()}")
        print(f"{Colors.BOLD}Generated Content:{Colors.ENDC}")
        print_content(content)

        print(f"\n{Colors.YELLOW}{Colors.BOLD}Review Options:{Colors.ENDC}")
        print(f"  {Colors.GREEN}[A]{Colors.ENDC} Approve - Publish the content as-is")
        print(f"  {Colors.RED}[R]{Colors.ENDC} Reject - Discard the content")
        print(f"  {Colors.YELLOW}[M]{Colors.ENDC} Modify - Request changes to the content")

        while True:
            decision = input(f"\n{Colors.BOLD}Your decision [A/R/M]:{Colors.ENDC} ").strip().upper()

            if decision in ['A', 'APPROVE']:
                return 'approve', None
            elif decision in ['R', 'REJECT']:
                reason = input(f"{Colors.BOLD}Reason for rejection (optional):{Colors.ENDC} ").strip()
                return 'reject', reason or "No reason provided"
            elif decision in ['M', 'MODIFY']:
                feedback = input(f"{Colors.BOLD}What changes would you like?{Colors.ENDC} ").strip()
                if feedback:
                    return 'modify', feedback
                else:
                    print_warning("Please provide feedback for modifications.")
            else:
                print_error("Invalid option. Please choose A, R, or M.")

    def publish_content(self, content: str, content_type: str):
        """Simulate publishing content."""
        print_section("📤 PUBLISHING CONTENT")
        print(f"Publishing {content_type} content...")
        # In a real scenario, this would publish to a blog, send an email, post to social media, etc.
        print_success(f"{content_type.upper()} content published successfully!")

        # Log the publication
        self.decision_log.append({
            "action": "published",
            "content_type": content_type,
            "content": content
        })

    def run_workflow(self, content_type: str, topic: str) -> bool:
        """
        Run the complete HITL workflow.

        Returns:
            bool: True if content was published, False otherwise
        """
        print_header("BASIC HITL CONTENT GENERATION WORKFLOW")

        print_section("📝 CONTENT GENERATION")
        print(f"Content Type: {content_type}")
        print(f"Topic: {topic}")
        print("\nGenerating content...")

        # Generate initial content
        additional_instructions = ""
        max_attempts = 3
        attempt = 1

        while attempt <= max_attempts:
            content = self.generator.generate_content(content_type, topic, additional_instructions)

            if not content:
                print_error("Failed to generate content. Aborting workflow.")
                return False

            print_success(f"Content generated (Attempt {attempt}/{max_attempts})")

            # Human checkpoint
            decision, feedback = self.get_human_decision(content, content_type)

            # Log the decision
            self.decision_log.append({
                "attempt": attempt,
                "decision": decision,
                "feedback": feedback,
                "content": content
            })

            if decision == 'approve':
                self.publish_content(content, content_type)
                return True

            elif decision == 'reject':
                print_section("❌ CONTENT REJECTED")
                print(f"Reason: {feedback}")
                print_error("Workflow terminated by human reviewer.")
                return False

            elif decision == 'modify':
                print_section("🔄 REGENERATING CONTENT")
                print(f"Feedback: {feedback}")
                additional_instructions = f"Previous feedback: {feedback}. Please incorporate this feedback."
                attempt += 1

                if attempt <= max_attempts:
                    print(f"\nRegenerating content (Attempt {attempt}/{max_attempts})...")
                else:
                    print_warning(f"Maximum attempts ({max_attempts}) reached.")
                    print_error("Workflow terminated.")
                    return False

        return False

    def show_audit_trail(self):
        """Display the audit trail of decisions."""
        print_section("📋 AUDIT TRAIL")

        if not self.decision_log:
            print("No decisions recorded.")
            return

        for i, entry in enumerate(self.decision_log, 1):
            print(f"\n{Colors.BOLD}Entry {i}:{Colors.ENDC}")
            for key, value in entry.items():
                if key == "content":
                    print(f"  {key}: [Content omitted for brevity]")
                else:
                    print(f"  {key}: {value}")


def main():
    """Main function to demonstrate the HITL workflow."""
    # Load environment variables
    env_path = Path(__file__).parent.parent.parent.parent / '.env'
    load_dotenv(env_path)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_error("OPENAI_API_KEY not found in environment variables.")
        print(f"Please ensure .env file exists at: {env_path}")
        sys.exit(1)

    # Initialize components
    generator = ContentGenerator(api_key)
    workflow = HITLWorkflow(generator)

    # Example scenarios
    scenarios = [
        ("blog", "The benefits of adopting a human-in-the-loop approach in AI systems"),
        ("email", "Announcing a new AI safety feature to customers"),
        ("social", "Celebrating our company's commitment to ethical AI"),
    ]

    print_header("WELCOME TO BASIC HITL DEMO")
    print("\nThis demo shows a simple human-in-the-loop workflow for content generation.")
    print("You'll be asked to approve, reject, or request modifications to generated content.\n")

    # Let user choose a scenario
    print_section("Available Scenarios")
    for i, (content_type, topic) in enumerate(scenarios, 1):
        print(f"{i}. {Colors.BOLD}{content_type.upper()}{Colors.ENDC}: {topic}")

    while True:
        try:
            choice = input(f"\n{Colors.BOLD}Select a scenario (1-{len(scenarios)}) or 'q' to quit:{Colors.ENDC} ").strip()

            if choice.lower() == 'q':
                print("\nGoodbye!")
                sys.exit(0)

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(scenarios):
                content_type, topic = scenarios[choice_idx]
                break
            else:
                print_error(f"Please enter a number between 1 and {len(scenarios)}.")
        except ValueError:
            print_error("Invalid input. Please enter a number or 'q'.")

    # Run the workflow
    success = workflow.run_workflow(content_type, topic)

    # Show audit trail
    workflow.show_audit_trail()

    # Summary
    print_section("📊 WORKFLOW SUMMARY")
    if success:
        print_success("Content was successfully published!")
    else:
        print_error("Content was not published.")

    print(f"\nTotal decisions made: {len([e for e in workflow.decision_log if 'decision' in e])}")


if __name__ == "__main__":
    main()
