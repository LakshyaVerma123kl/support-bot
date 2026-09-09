"""
Tests for TF-IDF conversation retriever.
"""

from agent.retriever import ConversationRetriever


def test_retriever_build_and_retrieve():
    retriever = ConversationRetriever()
    
    mock_conversations = [
        {
            "conversation_id": 101,
            "messages": [
                {"author_id": "123", "is_brand": False, "text": "My iPhone screen is completely black."},
                {"author_id": "AppleSupport", "is_brand": True, "text": "Try a force restart by holding the power and volume buttons."}
            ]
        },
        {
            "conversation_id": 102,
            "messages": [
                {"author_id": "456", "is_brand": False, "text": "How do I cancel my Apple Music trial?"},
                {"author_id": "AppleSupport", "is_brand": True, "text": "Go to Settings > Subscriptions to manage your plan."}
            ]
        },
        {
            "conversation_id": 103,
            "messages": [
                {"author_id": "789", "is_brand": False, "text": "My battery drains so fast on iOS 17."},
                {"author_id": "AppleSupport", "is_brand": True, "text": "Check Settings > Battery to see which apps are consuming power."}
            ]
        }
    ]

    retriever.build_index(mock_conversations)
    assert retriever._is_fitted
    assert len(retriever.conversation_index) == 3

    # Query matching screen problem
    results = retriever.retrieve("My iPhone display screen stays black", top_k=2)
    assert len(results) > 0
    assert "force restart" in results[0]['brand_reply'].lower()


def test_retriever_empty_query():
    retriever = ConversationRetriever()
    mock_convs = [
        {
            "conversation_id": 1,
            "messages": [
                {"author_id": "u", "is_brand": False, "text": "Device wifi broken"},
                {"author_id": "AppleSupport", "is_brand": True, "text": "Reset network settings"}
            ]
        }
    ]
    retriever.build_index(mock_convs)
    results = retriever.retrieve("completely unrelated astronaut spacecraft orbit", top_k=1)
    # Below similarity threshold
    assert isinstance(results, list)
