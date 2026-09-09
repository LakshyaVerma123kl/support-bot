"""
Tests for trivial and simple baselines.
"""

from baselines.trivial import TrivialBaseline
from baselines.simple import SimpleBaseline


def test_trivial_baseline():
    baseline = TrivialBaseline()
    baseline.fit()
    res = baseline.predict("My phone is broken!")
    assert "intent" in res
    assert res["escalation"] == "auto"
    assert "Please DM us" in res["reply"]


def test_simple_baseline_escalation_keywords():
    baseline = SimpleBaseline()
    # Check keyword escalation directly
    decision, reason = baseline.check_escalation("I am going to sue you for fraud!")
    assert decision == "escalate"
    assert "sue" in reason or "fraud" in reason

    decision, reason = baseline.check_escalation("How do I update my apps?")
    assert decision == "auto"
