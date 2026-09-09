"""
Escalation decision engine.

Determines whether a customer message should be auto-handled by the AI agent
or escalated to a human support agent, with a stated reason.

Uses a combination of:
1. Rule-based keyword detection (for obvious escalation signals)
2. LLM judgment (for nuanced cases)
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_CLASSIFY, BRAND_NAME, ESCALATION_KEYWORDS
)
from llm import chat_completion_json


def check_keyword_escalation(message_text):
    """
    Rule-based escalation check using keyword matching.

    Returns:
        (should_escalate: bool, matched_keywords: list)
    """
    text_lower = message_text.lower()
    matched = [kw for kw in ESCALATION_KEYWORDS if kw in text_lower]
    return len(matched) > 0, matched


def check_llm_escalation(customer_message, intent, confidence):
    """
    LLM-based escalation decision for nuanced cases.

    Args:
        customer_message: The customer's message text.
        intent: Classified intent.
        confidence: Classification confidence score.

    Returns:
        dict with 'decision', 'reason', 'urgency_level'
    """
    prompt = f"""You are a customer support escalation decision system for {BRAND_NAME}.

Analyze this customer message and decide if it should be:
- "auto": Handled automatically by the AI agent
- "escalate": Forwarded to a human support agent

Customer message: "{customer_message}"
Detected intent: {intent}
Classification confidence: {confidence:.2f}

ESCALATE if any of these apply:
- The customer is extremely frustrated, angry, or threatening
- The issue involves account security, fraud, or data breach
- The issue requires accessing customer account details
- Legal threats or regulatory mentions
- The customer explicitly asks for a human/manager/supervisor
- Complex technical issue that requires investigation
- Billing disputes over significant amounts
- The issue could cause reputational risk to the brand

AUTO-HANDLE if:
- It's a common, well-understood issue with a standard resolution
- The customer is asking a general question
- The customer needs to be directed to a standard resource (website, DM)
- The issue matches patterns the brand routinely handles on Twitter

Respond in JSON format:
{{
    "decision": "auto" or "escalate",
    "reason": "One sentence explaining why",
    "urgency_level": "low" or "medium" or "high" or "critical"
}}"""

    try:
        result = chat_completion_json(
            messages=[
                {"role": "system", "content": "You are an escalation decision system. Be precise and err on the side of escalating edge cases. Respond only with valid JSON."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL_CLASSIFY,
            temperature=0.1,
            max_tokens=256,
        )

        return {
            'decision': result.get('decision', 'escalate'),
            'reason': result.get('reason', 'Standard evaluation'),
            'urgency_level': result.get('urgency_level', 'medium'),
        }
    except Exception as e:
        return {
            'decision': 'escalate',
            'reason': f'Automated routing fallback: {str(e)[:60]}',
            'urgency_level': 'medium',
        }


def decide_escalation(customer_message, intent, confidence):
    """
    Full escalation decision combining rules and LLM judgment.

    Args:
        customer_message: The customer's message.
        intent: Classified intent name.
        confidence: Classification confidence score.

    Returns:
        dict: {
            'decision': 'auto' | 'escalate',
            'reason': str,
            'urgency_level': str,
            'method': 'keyword' | 'llm' | 'low_confidence',
        }
    """
    # Check 1: Low classification confidence → escalate
    if confidence < 0.4:
        return {
            'decision': 'escalate',
            'reason': f'Low classification confidence ({confidence:.2f}) — '
                      f'message is ambiguous and needs human review.',
            'urgency_level': 'medium',
            'method': 'low_confidence',
        }

    # Check 2: Keyword-based escalation
    should_escalate, matched_keywords = check_keyword_escalation(customer_message)
    if should_escalate:
        return {
            'decision': 'escalate',
            'reason': f'Message contains escalation signals: {", ".join(matched_keywords)}',
            'urgency_level': 'high',
            'method': 'keyword',
        }

    # Check 3: LLM judgment for nuanced cases
    llm_result = check_llm_escalation(customer_message, intent, confidence)
    llm_result['method'] = 'llm'
    return llm_result
