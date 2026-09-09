"""
Intent classifier using few-shot LLM prompting.

Uses Llama 3.1 8B via Groq for fast, high-quota classification.
Each message is classified into one of the taxonomy intents with a confidence score.

Usage:
    python -m intents.classify
"""

import sys
import json
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INTENT_TAXONOMY_PATH, MODEL_CLASSIFY, CONFIDENCE_THRESHOLD, BRAND_NAME
)
from llm import chat_completion_json


def load_taxonomy():
    """Load the intent taxonomy."""
    if not INTENT_TAXONOMY_PATH.exists():
        print(f"[X] Taxonomy not found at {INTENT_TAXONOMY_PATH}")
        print("  Run: python -m intents.discover")
        sys.exit(1)

    with open(INTENT_TAXONOMY_PATH, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
    return taxonomy['intents']


def build_classification_prompt(taxonomy):
    """Build the system prompt for intent classification."""
    intent_descriptions = "\n".join(
        f"  - **{intent['name']}**: {intent['description']}\n"
        f"    Examples: {'; '.join(intent['examples'][:3])}"
        for intent in taxonomy
    )

    system_prompt = f"""You are an intent classifier for {BRAND_NAME} customer support on Twitter.

Given a customer message, classify it into exactly ONE of these intents:

{intent_descriptions}

Respond in JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this intent was chosen"
}}

Rules:
- "confidence" should be between 0.0 and 1.0
- If the message doesn't clearly fit any intent, use "other" with low confidence
- Consider the full context of the message, not just keywords
- Be precise — don't default to "other" unless truly ambiguous"""

    return system_prompt


def classify_message(message_text, system_prompt):
    """Classify a single customer message."""
    try:
        result = chat_completion_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify this customer message:\n\n\"{message_text}\""},
            ],
            model=MODEL_CLASSIFY,
            temperature=0.1,
            max_tokens=256,
        )

        intent = result.get('intent', 'other')
        confidence = float(result.get('confidence', 0.0))
        reasoning = result.get('reasoning', '')

        # Apply confidence threshold — if below threshold, fall back to "other"
        if confidence < CONFIDENCE_THRESHOLD:
            intent = 'other'

        return {
            'intent': intent,
            'confidence': confidence,
            'reasoning': reasoning,
        }
    except Exception as e:
        return {
            'intent': 'other',
            'confidence': 0.0,
            'reasoning': f'Classification fallback due to error: {str(e)[:60]}',
        }


def classify_batch(messages, system_prompt, show_progress=True):
    """Classify a batch of messages."""
    results = []
    iterator = tqdm(messages, desc="Classifying") if show_progress else messages

    for msg in iterator:
        text = msg if isinstance(msg, str) else msg.get('text', '')
        result = classify_message(text, system_prompt)
        result['text'] = text
        results.append(result)

    return results


def get_classifier():
    """Return a ready-to-use classifier function."""
    taxonomy = load_taxonomy()
    system_prompt = build_classification_prompt(taxonomy)

    def classifier(message_text):
        return classify_message(message_text, system_prompt)

    return classifier


def main():
    """Demo: classify a few sample messages."""
    taxonomy = load_taxonomy()
    system_prompt = build_classification_prompt(taxonomy)

    # Load a few sample messages
    from config import PROCESSED_DATA_DIR
    path = PROCESSED_DATA_DIR / "subsample.jsonl"

    sample_messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            conv = json.loads(line)
            for msg in conv['messages']:
                if not msg['is_brand'] and len(msg['text']) > 20:
                    sample_messages.append(msg['text'])
                    break

    print(f"Classifying {len(sample_messages)} sample messages...\n")
    for msg_text in sample_messages:
        result = classify_message(msg_text, system_prompt)
        print(f"Message: \"{msg_text[:100]}...\"")
        print(f"  Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"  Reason: {result['reasoning']}")
        print()


if __name__ == "__main__":
    main()
