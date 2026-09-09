"""
Reply generation grounded in historical brand conversations.

Uses Llama 3.3 70B to draft replies that match the brand's tone and
are informed by similar past conversations retrieved from the index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_GENERATE, BRAND_NAME
from llm import chat_completion


def draft_reply(customer_message, intent, retrieved_conversations, brand=BRAND_NAME):
    """
    Draft a reply to a customer message, grounded in historical examples.

    Args:
        customer_message: The incoming customer message.
        intent: Classified intent of the message.
        retrieved_conversations: List of similar past conversations from retriever.
        brand: Brand name for tone matching.

    Returns:
        str: The drafted reply text.
    """
    # Format retrieved examples for the prompt
    examples_text = ""
    for i, conv in enumerate(retrieved_conversations[:3], 1):
        examples_text += f"""
Example {i} (similarity: {conv.get('similarity', 0):.2f}):
  Customer: "{conv['customer_message']}"
  {brand} replied: "{conv['brand_reply']}"
"""

    prompt = f"""You are a customer support agent for {brand} on Twitter.

A customer has sent the following message, which has been classified as intent: "{intent}".

Customer message: "{customer_message}"

Here are similar past conversations where {brand} successfully handled similar issues:
{examples_text}

Draft a helpful, professional reply that:
1. Acknowledges the customer's issue
2. Matches {brand}'s typical tone (helpful, concise, empathetic, professional)
3. Provides actionable next steps or a resolution
4. Stays within the character-appropriate length for Twitter (concise)
5. Is grounded in how {brand} has historically handled similar issues

Important: Do NOT make up specific information (like ticket numbers, phone numbers) 
that wasn't in the examples. If the issue needs investigation, offer to help via DM.

Reply:"""

    try:
        reply = chat_completion(
            messages=[
                {"role": "system", "content": f"You are {brand}'s customer support agent on Twitter. Write concise, helpful replies in the brand's voice. Keep replies under 280 characters when possible."},
                {"role": "user", "content": prompt},
            ],
            model=MODEL_GENERATE,
            temperature=0.4,
            max_tokens=256,
        )

        # Clean up the reply
        reply = reply.strip()
        # Remove any quotation marks wrapping the whole reply
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]

        return reply
    except Exception as e:
        # Fallback to nearest historical reply if available, or brand default
        if retrieved_conversations and len(retrieved_conversations) > 0:
            return retrieved_conversations[0]['brand_reply']
        return f"Thanks for reaching out. We'd like to look into this for you. Please DM us your device details and iOS version so we can help. [URL]"
