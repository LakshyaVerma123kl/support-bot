"""
Shared pytest fixtures and test helpers.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_messages():
    """Sample realistic customer messages."""
    return [
        "My iPhone 13 screen is black and won't turn on even after charging.",
        "How do I cancel my Apple Music subscription? I keep getting charged.",
        "I was charged twice for the same app on my credit card! This is fraud, I want a refund now!",
        "Thanks Apple, the reset fixed my wifi connection!",
        "My battery drains from 100% to 20% in 1 hour after updating to iOS 17.",
    ]


@pytest.fixture
def sample_taxonomy():
    """Mock intent taxonomy."""
    return [
        {
            "name": "hardware_defect",
            "description": "Physical device failures and screen issues",
            "examples": ["Screen won't turn on", "Button is broken", "Speaker stopped working"]
        },
        {
            "name": "billing_and_purchases",
            "description": "Billing, subscriptions, charges, and refunds",
            "examples": ["Cancel subscription", "Double charged", "Refund request"]
        },
        {
            "name": "battery_and_power",
            "description": "Battery drain, charging problems, overheating",
            "examples": ["Battery dies quickly", "Phone won't charge", "Device gets very hot"]
        },
        {
            "name": "other",
            "description": "Miscellaneous inquiries",
            "examples": ["Thanks for help", "Hello", "Where is the store"]
        }
    ]
