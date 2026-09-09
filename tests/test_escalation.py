"""
Tests for escalation gating rules.
"""

from agent.escalation import check_keyword_escalation, decide_escalation


def test_keyword_escalation_triggers():
    should_esc, kw = check_keyword_escalation("This is an unauthorized charge, total scam!")
    assert should_esc is True
    assert "scam" in kw


def test_keyword_escalation_legal():
    should_esc, kw = check_keyword_escalation("I am calling my lawyer to handle this lawsuit.")
    assert should_esc is True
    assert "lawyer" in kw or "lawsuit" in kw


def test_keyword_escalation_clean():
    should_esc, kw = check_keyword_escalation("How do I update to iOS 17?")
    assert should_esc is False
    assert len(kw) == 0


def test_low_confidence_escalation():
    # Confidence below 0.4 should trigger automatic escalation
    res = decide_escalation("my device thing", "other", confidence=0.25)
    assert res['decision'] == 'escalate'
    assert res['method'] == 'low_confidence'
