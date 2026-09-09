"""
Trivial Baseline.

Strategy:
- Intent: Always predict the most frequent intent
- Reply: Always return a canned "We're looking into this" message
- Escalation: Never escalate

This establishes the absolute floor for performance.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, INTENT_TAXONOMY_PATH, BRAND_NAME


class TrivialBaseline:
    """Most-frequent intent + canned reply baseline."""

    def __init__(self):
        self.most_frequent_intent = "other"
        self.canned_reply = (
            f"Thank you for reaching out! We'd like to help. "
            f"Please DM us your details so we can look into this for you."
        )

    def fit(self, data_path=None):
        """
        Determine the most frequent intent.
        Since we don't have pre-labelled data, we use the taxonomy
        and pick the first intent as a proxy.
        """
        if INTENT_TAXONOMY_PATH.exists():
            with open(INTENT_TAXONOMY_PATH, 'r') as f:
                taxonomy = json.load(f)
            if taxonomy.get('intents'):
                # Use the first intent as "most common" (a true trivial baseline
                # would compute this from labelled data, but this is close enough)
                self.most_frequent_intent = taxonomy['intents'][0]['name']

        print(f"  Trivial baseline: always predict '{self.most_frequent_intent}'")
        print(f"  Canned reply: \"{self.canned_reply[:60]}...\"")

    def predict(self, customer_message):
        """Make a prediction for a single message."""
        return {
            'intent': self.most_frequent_intent,
            'reply': self.canned_reply,
            'escalation': 'auto',
            'escalation_reason': 'Trivial baseline never escalates',
        }


def main():
    """Demo the trivial baseline."""
    baseline = TrivialBaseline()
    baseline.fit()

    test_messages = [
        "My iPhone won't turn on after the update!",
        "How do I cancel my Apple Music subscription?",
        "I've been charged twice for the same app!",
    ]

    print(f"\nTrivial Baseline Demo ({BRAND_NAME}):")
    print("=" * 50)
    for msg in test_messages:
        result = baseline.predict(msg)
        print(f"\n  [MSG] \"{msg}\"")
        print(f"  [TAG] Intent: {result['intent']}")
        print(f"  [REPLY] Reply: \"{result['reply'][:80]}...\"")
        print(f"  [GATE] Escalation: {result['escalation']}")


if __name__ == "__main__":
    main()
