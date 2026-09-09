"""
Interactive Demonstration & Comparison CLI for AppleSupport Support Agent.

Usage:
    python demo.py                     # Run preset interactive demo
    python demo.py --interactive       # Interactive REPL: type your own messages
    python demo.py --compare           # Side-by-side comparison: Agent vs Baselines
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BRAND_NAME
from agent.pipeline import SupportAgent
from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline

# Terminal formatting
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
GRAY = "\033[90m"
RESET = "\033[0m"


def print_banner():
    print(f"{BOLD}{BLUE}======================================================================{RESET}")
    print(f"{BOLD}{BLUE}       AI Customer Support Agent for {BRAND_NAME} -- Interactive Demo{RESET}")
    print(f"{BOLD}{BLUE}======================================================================{RESET}\n")


def format_escalation(decision, reason):
    if decision.lower() == "escalate":
        badge = f"{BOLD}{RED}[ESCALATE TO HUMAN]{RESET}"
    else:
        badge = f"{BOLD}{GREEN}[AUTO-RESOLVE]{RESET}"
    return f"{badge} {reason}"


def run_single(agent, text):
    print(f"\n{BOLD}{CYAN}Customer Tweet:{RESET} \"{text}\"")
    print(f"{GRAY}Processing through classify -> retrieve -> escalate -> respond...{RESET}")
    
    result = agent.process_message(text)
    
    print(f"  {BOLD}• Detected Intent:{RESET} {CYAN}{result['intent']}{RESET} (confidence: {result['intent_confidence']:.2f})")
    print(f"  {BOLD}• Escalation Gate:{RESET} {format_escalation(result['escalation_decision'], result['escalation_reason'])}")
    print(f"  {BOLD}• Retrieved Convs:{RESET} {len(result['retrieved_conversations'])} historical pairs indexed")
    if result['retrieved_conversations']:
        top_ref = result['retrieved_conversations'][0]
        print(f"    {GRAY}Top match ({top_ref['similarity']:.2f} sim): \"{top_ref['customer_message'][:60]}...\"{RESET}")
    print(f"  {BOLD}{GREEN}• Drafted Reply:{RESET}\n    \"{result['drafted_reply']}\"")
    print(f"{GRAY}{'-' * 70}{RESET}")


def run_comparison(agent, trivial, simple, text):
    print(f"\n{BOLD}{CYAN}Input Message:{RESET} \"{text}\"")
    print(f"{BOLD}{'-' * 70}{RESET}")
    
    agent_res = agent.process_message(text)
    trivial_res = trivial.predict(text)
    simple_res = simple.predict(text)
    
    print(f"{BOLD}{'Component':<18} {'AI Agent (Ours)':<24} {'Simple (TF-IDF)':<22} {'Trivial':<15}{RESET}")
    print("-" * 75)
    print(f"{'Intent':<18} {agent_res['intent'][:22]:<24} {simple_res['intent'][:20]:<22} {trivial_res['intent'][:14]:<15}")
    agent_conf = f"{agent_res['intent_confidence']:.2f}"
    simple_conf = f"{simple_res['intent_confidence']:.2f}"
    print(f"{'Confidence':<18} {agent_conf:<24} {simple_conf:<22} {'N/A':<15}")
    print(f"{'Escalation':<18} {agent_res['escalation_decision'].upper():<24} {simple_res['escalation'].upper():<22} {trivial_res['escalation'].upper():<15}")
    print("\nDrafted Replies:")
    print(f"  {BOLD}AI Agent:{RESET} \"{agent_res['drafted_reply']}\"")
    print(f"  {BOLD}Simple:{RESET}   \"{simple_res['reply'][:140]}...\"")
    print(f"  {BOLD}Trivial:{RESET}  \"{trivial_res['reply'][:140]}...\"")
    print(f"{GRAY}{'=' * 75}{RESET}")


def interactive_repl(agent):
    print_banner()
    print(f"Type any customer tweet below to see how the agent handles it in real time.")
    print(f"Type {BOLD}'exit'{RESET} or {BOLD}'quit'{RESET} to leave.\n")

    while True:
        try:
            user_input = input(f"{BOLD}Customer Tweet > {RESET}").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting demo.")
                break
            run_single(agent, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo.")
            break


def main():
    parser = argparse.ArgumentParser(description="AppleSupport AI Support Agent Demo")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive REPL mode")
    parser.add_argument("--compare", "-c", action="store_true", help="Run side-by-side comparison with baselines")
    args = parser.parse_args()

    agent = SupportAgent()

    if args.interactive:
        interactive_repl(agent)
        return

    presets = [
        "My iPhone 14 screen turned completely black while listening to music and won't turn on.",
        "How do I cancel my Apple Music monthly subscription from my phone?",
        "I was charged $49.99 twice for iCloud storage! This is theft and fraud, I demand an immediate refund or I will sue!",
        "Thanks Apple Support, updating to the new iOS build fixed my Wi-Fi disconnect issue!",
        "Why does my battery drop by 30% in 20 minutes when using FaceTime?",
    ]

    if args.compare:
        print_banner()
        print(f"Initializing baselines for side-by-side comparison...\n")
        trivial = TrivialBaseline()
        trivial.fit()
        simple = SimpleBaseline()
        simple.fit()

        for p in presets[:3]:
            run_comparison(agent, trivial, simple, p)
        return

    print_banner()
    print(f"Running agent demonstration on {len(presets)} realistic test scenarios:\n")
    for text in presets:
        run_single(agent, text)

    print(f"\n{GREEN}[TIP] Try interactive mode:{RESET} python demo.py --interactive")
    print(f"{GREEN}[TIP] Try baseline comparison:{RESET} python demo.py --compare")


if __name__ == "__main__":
    main()
