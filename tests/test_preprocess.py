"""
Tests for text preprocessing and cleaning.
"""

from data.preprocess import clean_text


def test_clean_text_anonymized_mentions():
    raw = "@115858 @AppleSupport My screen is cracked @12345"
    cleaned = clean_text(raw)
    assert "@115858" not in cleaned
    assert "@12345" not in cleaned
    assert "screen is cracked" in cleaned


def test_clean_text_urls():
    raw = "Check this out https://apple.co/support and http://example.com"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "http://" not in cleaned
    assert "[URL]" in cleaned


def test_clean_text_whitespace():
    raw = "  Lots   of    spaces   and\nnewlines\t\t"
    cleaned = clean_text(raw)
    assert cleaned == "Lots of spaces and newlines"


def test_clean_text_empty_and_nan():
    assert clean_text("") == ""
    assert clean_text(None) == ""
