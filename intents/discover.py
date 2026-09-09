"""
Discover intent taxonomy from customer messages using LLM clustering.

Steps:
1. Sample ~200 customer messages from the subsample
2. Send batches to the LLM for thematic clustering
3. Merge and refine into 8-15 distinct intents
4. Generate 3-5 example messages per intent
5. Save taxonomy as JSON

Usage:
    python -m intents.discover
"""

import sys
import json
import random
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, INTENT_TAXONOMY_PATH, RANDOM_SEED,
    INTENT_DISCOVERY_SAMPLE, MODEL_GENERATE, BRAND_NAME
)
from llm import chat_completion, chat_completion_json


def sample_customer_messages(n=INTENT_DISCOVERY_SAMPLE):
    """Sample customer messages from the subsample data."""
    path = PROCESSED_DATA_DIR / "subsample.jsonl"
    if not path.exists():
        print(f"[X] Subsample not found at {path}")
        print("  Run: python -m data.sample")
        sys.exit(1)

    messages = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            for msg in conv['messages']:
                if not msg['is_brand'] and len(msg['text']) > 20:
                    messages.append({
                        'text': msg['text'],
                        'conversation_id': conv['conversation_id'],
                    })

    random.seed(RANDOM_SEED)
    random.shuffle(messages)
    sampled = messages[:n]
    print(f"Sampled {len(sampled)} customer messages for intent discovery")
    return sampled


def discover_intents_batch(messages_batch):
    """Ask LLM to identify common intents in a batch of messages."""
    messages_text = "\n".join(
        f"{i+1}. \"{m['text']}\""
        for i, m in enumerate(messages_batch)
    )

    prompt = f"""You are analyzing customer support messages sent to {BRAND_NAME} on Twitter.

Below are {len(messages_batch)} real customer messages. Identify the main INTENT categories 
that these messages fall into. For each intent:
- Give it a short snake_case name (e.g., "device_issue", "billing_question")
- Write a one-sentence description
- List which message numbers belong to it

Messages:
{messages_text}

Respond in JSON format:
{{
    "intents": [
        {{
            "name": "intent_name",
            "description": "What this intent covers",
            "message_numbers": [1, 3, 7]
        }}
    ]
}}"""

    result = chat_completion_json(
        messages=[
            {"role": "system", "content": "You are an expert at customer support intent analysis. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model=MODEL_GENERATE,
        temperature=0.3,
        max_tokens=2048,
    )
    return result.get('intents', [])


def merge_intents(all_intent_batches):
    """Merge and consolidate intents from multiple batches into a unified taxonomy."""
    # Flatten all discovered intents
    all_intents = []
    for batch in all_intent_batches:
        all_intents.extend(batch)

    intent_descriptions = "\n".join(
        f"- {intent['name']}: {intent['description']}"
        for intent in all_intents
    )

    prompt = f"""You have analyzed customer support messages for {BRAND_NAME} and discovered 
these raw intent categories:

{intent_descriptions}

Many of these overlap or are too specific. Consolidate them into a final taxonomy of 
8-15 distinct, mutually exclusive intents that cover the full range of customer issues 
for {BRAND_NAME}.

For each final intent, provide:
- A clear snake_case name
- A concise description (1-2 sentences)
- 3-5 realistic example customer messages

Also add a catch-all "other" intent for messages that don't fit elsewhere.

Respond in JSON format:
{{
    "taxonomy": [
        {{
            "name": "intent_name",
            "description": "What this intent covers",
            "examples": [
                "Example message 1",
                "Example message 2",
                "Example message 3"
            ]
        }}
    ]
}}"""

    result = chat_completion_json(
        messages=[
            {"role": "system", "content": "You are an expert at designing customer support intent taxonomies. Create a clean, practical taxonomy. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        model=MODEL_GENERATE,
        temperature=0.2,
        max_tokens=4096,
    )
    return result.get('taxonomy', [])


def save_taxonomy(taxonomy):
    """Save the intent taxonomy to JSON."""
    INTENT_TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "brand": BRAND_NAME,
        "num_intents": len(taxonomy),
        "intents": taxonomy,
    }

    with open(INTENT_TAXONOMY_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Saved taxonomy with {len(taxonomy)} intents to {INTENT_TAXONOMY_PATH}")
    for intent in taxonomy:
        print(f"  - {intent['name']}: {intent['description']}")


def main():
    """Run intent discovery pipeline."""
    if INTENT_TAXONOMY_PATH.exists():
        print(f"[OK] Taxonomy already exists at {INTENT_TAXONOMY_PATH}")
        with open(INTENT_TAXONOMY_PATH, 'r', encoding='utf-8') as f:
            taxonomy = json.load(f)
        print(f"  {taxonomy['num_intents']} intents defined")
        for intent in taxonomy['intents']:
            print(f"  - {intent['name']}: {intent['description']}")
        return taxonomy

    print(f"Discovering intent taxonomy for {BRAND_NAME}...")

    # Sample customer messages
    messages = sample_customer_messages()

    # Process in batches of 40 (to fit in context window)
    batch_size = 40
    all_intent_batches = []

    for i in tqdm(range(0, len(messages), batch_size), desc="Analyzing batches"):
        batch = messages[i:i + batch_size]
        intents = discover_intents_batch(batch)
        all_intent_batches.append(intents)

    # Merge into unified taxonomy
    print("\nConsolidating into unified taxonomy...")
    taxonomy = merge_intents(all_intent_batches)

    # Save
    save_taxonomy(taxonomy)

    return taxonomy


if __name__ == "__main__":
    main()
