"""
Tests for central configuration.
"""

from pathlib import Path
from config import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    RESULTS_DIR, BRAND_NAME, MODEL_CLASSIFY, MODEL_GENERATE, MODEL_JUDGE,
    JUDGE_DIMENSIONS, ESCALATION_KEYWORDS
)


def test_paths_exist():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert RESULTS_DIR.exists()


def test_brand_configuration():
    assert BRAND_NAME == "AppleSupport"


def test_model_definitions():
    assert isinstance(MODEL_CLASSIFY, str) and len(MODEL_CLASSIFY) > 0
    assert isinstance(MODEL_GENERATE, str) and len(MODEL_GENERATE) > 0
    assert isinstance(MODEL_JUDGE, str) and len(MODEL_JUDGE) > 0


def test_judge_dimensions():
    expected = ["relevance", "tone", "accuracy", "helpfulness", "completeness"]
    assert JUDGE_DIMENSIONS == expected


def test_escalation_keywords():
    assert isinstance(ESCALATION_KEYWORDS, list)
    assert len(ESCALATION_KEYWORDS) > 10
    assert "fraud" in ESCALATION_KEYWORDS
    assert "sue" in ESCALATION_KEYWORDS
