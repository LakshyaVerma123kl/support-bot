"""
Preprocess the raw Twitter Customer Support dataset.

Steps:
1. Load the raw CSV
2. Filter to the chosen brand (AppleSupport)
3. Reconstruct multi-turn conversation threads
4. Clean text (remove anonymized @mentions, normalize whitespace)
5. Save processed conversations as JSONL

Usage:
    python -m data.preprocess
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, RAW_CSV_FILENAME, BRAND_NAME
)


def clean_text(text):
    """Clean a tweet's text content."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove anonymized @mentions like @12345
    text = re.sub(r'@\d+', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_raw_data():
    """Load the raw CSV dataset."""
    csv_path = RAW_DATA_DIR / RAW_CSV_FILENAME
    if not csv_path.exists():
        print(f"[X] Raw data not found at {csv_path}")
        print("  Run: python -m data.download")
        sys.exit(1)

    print(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total tweets: {len(df):,}")
    return df


def identify_brand(df, brand_name):
    """
    Identify the brand's author_id.
    Brand tweets have inbound=False and typically contain the brand name
    in their text or are frequently referenced.
    """
    # Outbound tweets (from brands)
    outbound = df[df['inbound'] == False].copy()

    # Find author_ids that appear most frequently in outbound tweets
    author_counts = outbound['author_id'].value_counts()

    print(f"\nTop 10 brand accounts by tweet volume:")
    for author_id, count in author_counts.head(10).items():
        # Sample a tweet to see the content
        sample_text = outbound[outbound['author_id'] == author_id]['text'].iloc[0]
        print(f"  {author_id}: {count:,} tweets — \"{sample_text[:80]}...\"")

    # Check direct match in outbound author_ids
    exact_match = [a for a in author_counts.index if str(a).lower() == brand_name.lower()]
    if exact_match:
        brand_author_id = exact_match[0]
    else:
        # Fallback: search for tweets containing the brand name
        brand_tweets = outbound[
            outbound['text'].str.contains(brand_name, case=False, na=False)
        ]
        if len(brand_tweets) > 0:
            brand_author_id = brand_tweets['author_id'].value_counts().index[0]
        else:
            brand_author_id = author_counts.index[0]

    brand_tweet_count = len(outbound[outbound['author_id'] == brand_author_id])
    print(f"\n[OK] Identified brand '{brand_name}' as author_id={brand_author_id}")
    print(f"  Total brand tweets: {brand_tweet_count:,}")
    return brand_author_id


def build_threads(df, brand_author_id):
    """
    Reconstruct multi-turn conversation threads.

    A thread is a sequence of tweets connected via
    in_response_to_tweet_id / response_tweet_id.

    Returns list of conversation dicts.
    """
    print("\nBuilding conversation threads...")

    # Build adjacency: tweet_id -> list of response tweet_ids
    # Also index tweets by tweet_id
    tweet_index = {}
    response_map = defaultdict(list)  # tweet_id -> [response_tweet_ids]
    parent_map = {}  # tweet_id -> parent_tweet_id

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing tweets"):
        tid = row['tweet_id']
        tweet_index[tid] = row

        # Parse response_tweet_id (can be comma-separated)
        if pd.notna(row.get('response_tweet_id')):
            resp_ids = str(row['response_tweet_id']).split(',')
            for rid in resp_ids:
                rid = rid.strip()
                if rid:
                    try:
                        response_map[tid].append(int(float(rid)))
                    except (ValueError, OverflowError):
                        pass

        # Map child -> parent
        if pd.notna(row.get('in_response_to_tweet_id')):
            try:
                parent_id = int(float(row['in_response_to_tweet_id']))
                parent_map[tid] = parent_id
            except (ValueError, OverflowError):
                pass

    # Find conversation roots: tweets that involve the brand and have no parent
    # A root is either:
    #   - An inbound tweet to the brand (no parent, or parent not in dataset)
    #   - A brand outbound tweet that starts a thread
    brand_tweets = set(
        df[df['author_id'] == brand_author_id]['tweet_id'].values
    )

    # Find all tweets that are in conversations involving this brand
    involved_tweets = set()
    for tid in brand_tweets:
        # Walk up to root
        current = tid
        chain = [current]
        visited = set()
        while current in parent_map and current not in visited:
            visited.add(current)
            current = parent_map[current]
            if current in tweet_index:
                chain.append(current)
        involved_tweets.update(chain)

        # Walk down from brand tweet
        queue = [tid]
        visited_down = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited_down:
                continue
            visited_down.add(curr)
            involved_tweets.add(curr)
            for child in response_map.get(curr, []):
                if child in tweet_index:
                    queue.append(child)

    # Find roots of these conversations
    roots = set()
    for tid in involved_tweets:
        current = tid
        visited = set()
        while current in parent_map and parent_map[current] in tweet_index and current not in visited:
            visited.add(current)
            current = parent_map[current]
        roots.add(current)

    print(f"  Found {len(roots):,} conversation roots involving {BRAND_NAME}")

    # Build threads from roots
    conversations = []
    for root_id in tqdm(roots, desc="Building threads"):
        if root_id not in tweet_index:
            continue

        thread = []
        queue = [root_id]
        visited = set()

        while queue:
            curr_id = queue.pop(0)
            if curr_id in visited or curr_id not in tweet_index:
                continue
            visited.add(curr_id)

            row = tweet_index[curr_id]
            thread.append({
                'tweet_id': int(curr_id),
                'author_id': str(row['author_id']),
                'is_brand': row['author_id'] == brand_author_id,
                'inbound': bool(row.get('inbound', True)),
                'text': clean_text(row['text']),
                'created_at': str(row.get('created_at', '')),
            })

            # Add children
            for child_id in response_map.get(curr_id, []):
                if child_id not in visited:
                    queue.append(child_id)

        if len(thread) < 2:
            continue  # Skip single-tweet "conversations"

        # Sort thread by tweet_id (proxy for chronological order)
        thread.sort(key=lambda t: t['tweet_id'])

        # Only keep conversations that have both customer and brand messages
        has_brand = any(t['is_brand'] for t in thread)
        has_customer = any(not t['is_brand'] for t in thread)
        if has_brand and has_customer:
            conversations.append({
                'conversation_id': int(root_id),
                'brand': BRAND_NAME,
                'num_turns': len(thread),
                'messages': thread,
            })

    print(f"  [OK] Built {len(conversations):,} complete conversations")
    return conversations


def save_conversations(conversations):
    """Save conversations as JSONL."""
    output_path = PROCESSED_DATA_DIR / "conversations.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Saved {len(conversations):,} conversations to {output_path} ({size_mb:.1f} MB)")
    return output_path


def compute_stats(conversations):
    """Print summary statistics about the processed conversations."""
    num_convs = len(conversations)
    turn_counts = [c['num_turns'] for c in conversations]
    customer_msgs = sum(
        1 for c in conversations
        for m in c['messages'] if not m['is_brand']
    )
    brand_msgs = sum(
        1 for c in conversations
        for m in c['messages'] if m['is_brand']
    )

    print(f"\n{'='*50}")
    print(f"Dataset Statistics for {BRAND_NAME}")
    print(f"{'='*50}")
    print(f"  Total conversations:    {num_convs:,}")
    print(f"  Total customer messages: {customer_msgs:,}")
    print(f"  Total brand messages:    {brand_msgs:,}")
    print(f"  Avg turns/conversation:  {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"  Min turns:               {min(turn_counts)}")
    print(f"  Max turns:               {max(turn_counts)}")
    print(f"  Median turns:            {sorted(turn_counts)[len(turn_counts)//2]}")
    print(f"{'='*50}")


def main():
    """Run the full preprocessing pipeline."""
    # Check if already processed
    output_path = PROCESSED_DATA_DIR / "conversations.jsonl"
    if output_path.exists():
        print(f"[OK] Processed data already exists at {output_path}")
        # Load and show stats
        conversations = []
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
        compute_stats(conversations)
        return conversations

    # Load raw data
    df = load_raw_data()

    # Identify brand
    brand_author_id = identify_brand(df, BRAND_NAME)

    # Build conversation threads
    conversations = build_threads(df, brand_author_id)

    # Save
    save_conversations(conversations)

    # Stats
    compute_stats(conversations)

    return conversations


if __name__ == "__main__":
    main()
