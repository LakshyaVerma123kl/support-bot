"""
Full AI Agent pipeline.

Orchestrates: classify → retrieve → respond → escalate

Usage:
    python -m agent.pipeline
"""

import sys
import json
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, RESULTS_DIR, BRAND_NAME
from intents.classify import load_taxonomy, build_classification_prompt, classify_message
from agent.retriever import get_retriever
from agent.responder import draft_reply
from agent.escalation import decide_escalation


class SupportAgent:
    """End-to-end AI customer support agent."""

    def __init__(self):
        print(f"Initializing {BRAND_NAME} Support Agent...")

        # Load intent taxonomy and build classifier prompt
        self.taxonomy = load_taxonomy()
        self.classifier_prompt = build_classification_prompt(self.taxonomy)

        # Initialize retriever
        self.retriever = get_retriever()

        print("[OK] Agent ready\n")

    def process_message(self, customer_message):
        """
        Process a single customer message through the full pipeline.

        Args:
            customer_message: The incoming customer message text.

        Returns:
            dict with all intermediate and final outputs.
        """
        result = {
            'customer_message': customer_message,
            'brand': BRAND_NAME,
        }

        # Step 1: Classify intent
        classification = classify_message(customer_message, self.classifier_prompt)
        result['intent'] = classification['intent']
        result['intent_confidence'] = classification['confidence']
        result['intent_reasoning'] = classification['reasoning']

        # Step 2: Retrieve similar conversations
        retrieved = self.retriever.retrieve(customer_message)
        result['retrieved_conversations'] = [
            {
                'customer_message': r['customer_message'],
                'brand_reply': r['brand_reply'],
                'similarity': r['similarity'],
            }
            for r in retrieved
        ]

        # Step 3: Decide escalation
        escalation = decide_escalation(
            customer_message,
            classification['intent'],
            classification['confidence'],
        )
        result['escalation_decision'] = escalation['decision']
        result['escalation_reason'] = escalation['reason']
        result['escalation_urgency'] = escalation['urgency_level']
        result['escalation_method'] = escalation['method']

        # Step 4: Draft reply (even if escalating, provide a suggested reply)
        reply = draft_reply(
            customer_message,
            classification['intent'],
            retrieved,
        )
        result['drafted_reply'] = reply

        return result

    def process_batch(self, messages, show_progress=True):
        """
        Process a batch of customer messages.

        Args:
            messages: List of message dicts with 'text' and optional 'conversation_id'.
            show_progress: Whether to show a progress bar.

        Returns:
            List of result dicts.
        """
        results = []
        iterator = tqdm(messages, desc="Processing messages") if show_progress else messages

        for msg in iterator:
            text = msg if isinstance(msg, str) else msg.get('text', '')
            try:
                result = self.process_message(text)
                if isinstance(msg, dict):
                    result['conversation_id'] = msg.get('conversation_id', '')
                results.append(result)
            except Exception as e:
                print(f"\n  [!] Error processing message: {str(e)[:100]}")
                results.append({
                    'customer_message': text,
                    'error': str(e),
                })

            # Small delay to respect rate limits
            time.sleep(0.5)

        return results


def main():
    """Demo the full pipeline on a few sample messages."""
    agent = SupportAgent()

    # Load a few sample customer messages
    path = PROCESSED_DATA_DIR / "subsample.jsonl"
    sample_messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            conv = json.loads(line)
            for msg in conv['messages']:
                if not msg['is_brand'] and len(msg['text']) > 20:
                    sample_messages.append(msg['text'])
                    break

    print(f"Processing {len(sample_messages)} sample messages...\n")
    print("=" * 70)

    for text in sample_messages:
        result = agent.process_message(text)

        print(f"\n[MSG] Customer: \"{result['customer_message'][:120]}\"")
        print(f"   [TAG]  Intent: {result['intent']} "
              f"(confidence: {result['intent_confidence']:.2f})")
        print(f"   [SEARCH] Retrieved {len(result['retrieved_conversations'])} "
              f"similar conversations")
        print(f"   [GATE] Escalation: {result['escalation_decision'].upper()} "
              f"({result['escalation_reason']})")
        print(f"   [REPLY] Reply: \"{result['drafted_reply'][:200]}\"")
        print("-" * 70)


if __name__ == "__main__":
    main()
