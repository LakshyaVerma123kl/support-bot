"""
TF-IDF-based retriever for finding similar historical conversations.

Builds an index over customer messages and retrieves the top-K most similar
past conversations for grounding reply generation.
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DATA_DIR, RESULTS_DIR, RETRIEVER_TOP_K, TFIDF_MAX_FEATURES
)


class ConversationRetriever:
    """Retrieve similar historical conversations using TF-IDF cosine similarity."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
        )
        self.conversation_index = []  # List of (customer_msg, brand_reply, full_conv)
        self.tfidf_matrix = None
        self._is_fitted = False

    def build_index(self, conversations=None):
        """Build the TF-IDF index from conversations."""
        if conversations is None:
            conversations = self._load_conversations()

        print("Building retriever index...")

        # Extract (customer_message, brand_reply) pairs from conversations
        self.conversation_index = []
        for conv in conversations:
            messages = conv['messages']
            # Find customer → brand reply pairs
            for i, msg in enumerate(messages):
                if not msg['is_brand'] and len(msg['text']) > 10:
                    # Find the next brand reply
                    brand_reply = None
                    for j in range(i + 1, len(messages)):
                        if messages[j]['is_brand'] and len(messages[j]['text']) > 10:
                            brand_reply = messages[j]['text']
                            break

                    if brand_reply:
                        self.conversation_index.append({
                            'customer_message': msg['text'],
                            'brand_reply': brand_reply,
                            'conversation_id': conv['conversation_id'],
                            'full_thread': [m['text'] for m in messages],
                        })

        if not self.conversation_index:
            print("[!] No conversation pairs found!")
            return

        # Build TF-IDF matrix on customer messages
        customer_texts = [item['customer_message'] for item in self.conversation_index]
        min_df = 2 if len(customer_texts) > 10 else 1
        self.vectorizer.set_params(min_df=min_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(customer_texts)
        self._is_fitted = True

        print(f"  [OK] Indexed {len(self.conversation_index):,} message-reply pairs")
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,}")

    def retrieve(self, query_text, top_k=RETRIEVER_TOP_K):
        """
        Retrieve the top-K most similar conversations for a given query.

        Args:
            query_text: The customer message to find similar conversations for.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'customer_message', 'brand_reply', 'similarity', etc.
        """
        if not self._is_fitted:
            raise RuntimeError("Retriever not fitted. Call build_index() first.")

        query_vec = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum similarity threshold
                item = self.conversation_index[idx].copy()
                item['similarity'] = float(similarities[idx])
                results.append(item)

        return results

    def save_index(self):
        """Save the fitted index to disk."""
        index_path = RESULTS_DIR / "retriever_index.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'conversation_index': self.conversation_index,
                'tfidf_matrix': self.tfidf_matrix,
            }, f)
        print(f"  [OK] Saved retriever index to {index_path}")

    def load_index(self):
        """Load a previously saved index."""
        index_path = RESULTS_DIR / "retriever_index.pkl"
        if not index_path.exists():
            return False

        with open(index_path, 'rb') as f:
            data = pickle.load(f)

        self.vectorizer = data['vectorizer']
        self.conversation_index = data['conversation_index']
        self.tfidf_matrix = data['tfidf_matrix']
        self._is_fitted = True
        print(f"  [OK] Loaded retriever index ({len(self.conversation_index):,} pairs)")
        return True

    def _load_conversations(self):
        """Load conversations from the subsample."""
        path = PROCESSED_DATA_DIR / "subsample.jsonl"
        if not path.exists():
            print(f"[X] Subsample not found at {path}")
            sys.exit(1)

        conversations = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
        return conversations


# Module-level singleton
_retriever = None


def get_retriever():
    """Get or create the retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = ConversationRetriever()
        if not _retriever.load_index():
            _retriever.build_index()
            _retriever.save_index()
    return _retriever
