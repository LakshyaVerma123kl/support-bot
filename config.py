"""
Hiver SDE Intern — AI Customer Support Agent
Central configuration for the entire pipeline.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding — tweet data contains Unicode that cp1252 can't handle
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
GOLDEN_SET_DIR = PROJECT_ROOT / "evaluation" / "golden_set"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR, GOLDEN_SET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Brand Configuration
# ──────────────────────────────────────────────
BRAND_NAME = "AppleSupport"
# The author_id for the brand will be discovered during preprocessing

# ──────────────────────────────────────────────
# LLM Configuration (Groq Free Tier)
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model selection:
# Default is qwen/qwen3.8-27b (sub-second ~0.28s latency, direct JSON mode)
# Alternative supported models on Groq: openai/gpt-oss-120b, openai/gpt-oss-20b
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
MODEL_CLASSIFY = DEFAULT_MODEL
MODEL_GENERATE = DEFAULT_MODEL
MODEL_JUDGE = DEFAULT_MODEL

# Rate limit settings (free tier)
RATE_LIMITS = {
    "qwen/qwen3.8-27b": {
        "rpm": 30,       # Requests per minute
        "rpd": 14400,    # Requests per day
        "tpd": 500000,   # Tokens per day
    },
    "openai/gpt-oss-120b": {
        "rpm": 30,
        "rpd": 10000,
        "tpd": 500000,
    },
    "openai/gpt-oss-20b": {
        "rpm": 30,
        "rpd": 14400,
        "tpd": 500000,
    },
}

# Retry settings
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, exponential backoff

# ──────────────────────────────────────────────
# Sampling Configuration
# ──────────────────────────────────────────────
RANDOM_SEED = 42
SUBSAMPLE_CONVERSATIONS = 2000    # For pipeline development
EVAL_HOLDOUT_CONVERSATIONS = 500  # For evaluation
GOLDEN_SET_SIZE = 200             # Hand-labelled examples
INTENT_DISCOVERY_SAMPLE = 200    # Messages for intent taxonomy discovery

# ──────────────────────────────────────────────
# Intent Classification
# ──────────────────────────────────────────────
INTENT_TAXONOMY_PATH = PROJECT_ROOT / "intents" / "taxonomy.json"
CONFIDENCE_THRESHOLD = 0.6  # Below this → "unknown" intent

# ──────────────────────────────────────────────
# Retriever Configuration
# ──────────────────────────────────────────────
RETRIEVER_TOP_K = 5  # Number of similar conversations to retrieve
TFIDF_MAX_FEATURES = 10000

# ──────────────────────────────────────────────
# Escalation Configuration
# ──────────────────────────────────────────────
ESCALATION_KEYWORDS = [
    "urgent", "emergency", "legal", "lawyer", "sue", "lawsuit",
    "attorney", "police", "fraud", "scam", "stolen", "hack",
    "data breach", "privacy", "gdpr", "complaint", "bbb",
    "ftc", "consumer protection", "class action", "refund",
    "supervisor", "manager", "escalate",
]

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
JUDGE_DIMENSIONS = [
    "relevance",      # Does the reply address the customer's issue?
    "tone",           # Does it match the brand's voice?
    "accuracy",       # Is it factually correct?
    "helpfulness",    # Does it provide actionable next steps?
    "completeness",   # Is anything missing?
]
JUDGE_AGREEMENT_SAMPLE = 50  # Examples for human-judge agreement

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
RAW_CSV_FILENAME = "twcs.csv"
