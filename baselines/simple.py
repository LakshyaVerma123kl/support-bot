"""
Simple Baseline (TF-IDF Nearest Neighbor).

Strategy:
- Intent: Predict using TF-IDF similarity to intent examples in the taxonomy
- Reply: Return the brand reply from the most similar historical conversation
- Escalation: Keyword-based (escalate if message contains trigger words)

This is a reasonable non-LLM approach that should beat the trivial baseline
but underperform the full LLM agent.
"""

import sys
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, INTENT_TAXONOMY_PATH, BRAND_NAME,
    ESCALATION_KEYWORDS, TFIDF_MAX_FEATURES
)


class SimpleBaseline:
    """TF-IDF nearest-neighbor baseline."""

    def __init__(self):
        self.intent_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
        )
        self.reply_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2),
        )
        self.intent_examples = []  # (text, intent_name)
        self.intent_matrix = None
        self.conversation_pairs = []  # (customer_text, brand_reply)
        self.reply_matrix = None

    def fit(self, data_path=None):
        """Build the TF-IDF indexes for intent classification and reply retrieval."""
        # Load intent taxonomy for classification
        if INTENT_TAXONOMY_PATH.exists():
            with open(INTENT_TAXONOMY_PATH, 'r') as f:
                taxonomy = json.load(f)

            for intent in taxonomy.get('intents', []):
                for example in intent.get('examples', []):
                    self.intent_examples.append((example, intent['name']))

        if self.intent_examples:
            texts = [t for t, _ in self.intent_examples]
            self.intent_matrix = self.intent_vectorizer.fit_transform(texts)
            print(f"  Intent index: {len(self.intent_examples)} examples across "
                  f"{len(set(i for _, i in self.intent_examples))} intents")
        else:
            print("  [!] No intent taxonomy found, will default to 'other'")

        # Load conversation pairs for reply retrieval
        subsample_path = PROCESSED_DATA_DIR / "subsample.jsonl"
        if subsample_path.exists():
            with open(subsample_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conv = json.loads(line)
                    messages = conv['messages']
                    for i, msg in enumerate(messages):
                        if not msg['is_brand'] and len(msg['text']) > 10:
                            for j in range(i + 1, len(messages)):
                                if messages[j]['is_brand'] and len(messages[j]['text']) > 10:
                                    self.conversation_pairs.append(
                                        (msg['text'], messages[j]['text'])
                                    )
                                    break

        if self.conversation_pairs:
            texts = [t for t, _ in self.conversation_pairs]
            self.reply_matrix = self.reply_vectorizer.fit_transform(texts)
            print(f"  Reply index: {len(self.conversation_pairs)} conversation pairs")
        else:
            print("  [!] No conversation pairs found")

    def classify_intent(self, message):
        """Classify intent using TF-IDF similarity to taxonomy examples."""
        if self.intent_matrix is None:
            return 'other', 0.0

        query = self.intent_vectorizer.transform([message])
        similarities = cosine_similarity(query, self.intent_matrix).flatten()

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.intent_examples[best_idx][1]

        return best_intent, float(best_score)

    def retrieve_reply(self, message):
        """Find the most similar historical conversation and return its brand reply."""
        if self.reply_matrix is None:
            return "Thank you for reaching out. Please DM us for assistance."

        query = self.reply_vectorizer.transform([message])
        similarities = cosine_similarity(query, self.reply_matrix).flatten()

        best_idx = np.argmax(similarities)
        return self.conversation_pairs[best_idx][1]

    def check_escalation(self, message):
        """Keyword-based escalation check with word boundaries."""
        text_lower = message.lower()
        matched = [
            kw for kw in ESCALATION_KEYWORDS
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
        ]
        if matched:
            return 'escalate', f'Keywords detected: {", ".join(matched)}'
        return 'auto', 'No escalation signals detected'

    def predict(self, customer_message):
        """Make a full prediction for a customer message."""
        intent, confidence = self.classify_intent(customer_message)
        reply = self.retrieve_reply(customer_message)
        escalation, reason = self.check_escalation(customer_message)

        return {
            'intent': intent,
            'intent_confidence': confidence,
            'reply': reply,
            'escalation': escalation,
            'escalation_reason': reason,
        }


def main():
    """Demo the simple baseline."""
    baseline = SimpleBaseline()
    baseline.fit()

    test_messages = [
        "My iPhone won't turn on after the update!",
        "How do I cancel my Apple Music subscription?",
        "I've been charged twice for the same app! I'm going to sue!",
    ]

    print(f"\nSimple Baseline Demo ({BRAND_NAME}):")
    print("=" * 60)
    for msg in test_messages:
        result = baseline.predict(msg)
        print(f"\n  [MSG] \"{msg}\"")
        print(f"  [TAG] Intent: {result['intent']} ({result['intent_confidence']:.2f})")
        print(f"  [REPLY] Reply: \"{result['reply'][:100]}...\"")
        print(f"  [GATE] Escalation: {result['escalation']} ({result['escalation_reason']})")


if __name__ == "__main__":
    main()
