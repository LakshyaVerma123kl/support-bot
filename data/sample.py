"""
Create reproducible subsamples from the processed conversations.

Produces:
- data/processed/subsample.jsonl     — 2000 conversations for development
- data/processed/eval_holdout.jsonl  — 500 conversations for evaluation

Usage:
    python -m data.sample
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, RANDOM_SEED,
    SUBSAMPLE_CONVERSATIONS, EVAL_HOLDOUT_CONVERSATIONS
)


def load_conversations():
    """Load all processed conversations."""
    path = PROCESSED_DATA_DIR / "conversations.jsonl"
    if not path.exists():
        print(f"[X] Processed data not found at {path}")
        print("  Run: python -m data.preprocess")
        sys.exit(1)

    conversations = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))

    print(f"Loaded {len(conversations):,} conversations")
    return conversations


def create_samples(conversations):
    """Create stratified subsamples."""
    random.seed(RANDOM_SEED)

    # Shuffle deterministically
    shuffled = conversations.copy()
    random.shuffle(shuffled)

    total_needed = SUBSAMPLE_CONVERSATIONS + EVAL_HOLDOUT_CONVERSATIONS

    if len(shuffled) < total_needed:
        print(f"[!] Only {len(shuffled)} conversations available, "
              f"need {total_needed}. Adjusting split...")
        split_point = int(len(shuffled) * 0.8)
        subsample = shuffled[:split_point]
        eval_holdout = shuffled[split_point:]
    else:
        subsample = shuffled[:SUBSAMPLE_CONVERSATIONS]
        eval_holdout = shuffled[SUBSAMPLE_CONVERSATIONS:
                                SUBSAMPLE_CONVERSATIONS + EVAL_HOLDOUT_CONVERSATIONS]

    return subsample, eval_holdout


def save_sample(conversations, filename):
    """Save a sample to JSONL."""
    path = PROCESSED_DATA_DIR / filename
    with open(path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    print(f"  [OK] Saved {len(conversations):,} conversations to {path}")
    return path


def main():
    """Create subsamples."""
    # Check if already done
    subsample_path = PROCESSED_DATA_DIR / "subsample.jsonl"
    holdout_path = PROCESSED_DATA_DIR / "eval_holdout.jsonl"

    if subsample_path.exists() and holdout_path.exists():
        print("[OK] Subsamples already exist:")
        for p in [subsample_path, holdout_path]:
            count = sum(1 for _ in open(p, 'r', encoding='utf-8'))
            print(f"  {p.name}: {count:,} conversations")
        return

    conversations = load_conversations()
    subsample, eval_holdout = create_samples(conversations)

    print(f"\nCreating subsamples (seed={RANDOM_SEED}):")
    save_sample(subsample, "subsample.jsonl")
    save_sample(eval_holdout, "eval_holdout.jsonl")

    # Print distribution info
    for name, sample in [("Subsample", subsample), ("Eval holdout", eval_holdout)]:
        turns = [c['num_turns'] for c in sample]
        print(f"\n  {name}:")
        print(f"    Conversations: {len(sample):,}")
        print(f"    Avg turns: {sum(turns)/len(turns):.1f}")
        print(f"    Total messages: {sum(turns):,}")


if __name__ == "__main__":
    main()
