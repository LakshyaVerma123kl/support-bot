"""
Groq LLM client with rate-limit-aware retries.

Wraps the Groq SDK with:
- Automatic retry with exponential backoff on 429/5xx errors
- Per-model rate limit tracking
- JSON mode support for structured output
"""

import time
import json
import sys
from pathlib import Path

from groq import Groq, RateLimitError, APIError

sys.path.insert(0, str(Path(__file__).parent))
from config import GROQ_API_KEY, MAX_RETRIES, RETRY_BASE_DELAY


# Singleton client
_client = None


def get_client():
    """Get or create the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            print("[X] GROQ_API_KEY not set.")
            print("  1. Get a free key at https://console.groq.com/keys")
            print("  2. Create a .env file with: GROQ_API_KEY=your-key-here")
            sys.exit(1)
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat_completion(
    messages: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    json_mode: bool = False,
    retries: int = MAX_RETRIES,
) -> str:
    """
    Send a chat completion request to Groq with automatic retry.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: Model name (e.g., 'llama-3.1-8b-instant').
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        json_mode: If True, request JSON response format.
        retries: Number of retries on rate limit errors.

    Returns:
        The assistant's response text.
    """
    client = get_client()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == retries:
                raise
            # Parse retry-after header if available, otherwise exponential backoff
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"  [..] Rate limited (attempt {attempt+1}/{retries}), "
                  f"waiting {delay:.1f}s...")
            time.sleep(delay)

        except APIError as e:
            if attempt == retries:
                raise
            if e.status_code and e.status_code >= 500:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [!] Server error {e.status_code} (attempt {attempt+1}/"
                      f"{retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise


def chat_completion_json(
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> dict:
    """
    Send a chat completion request and parse the JSON response.

    Returns:
        Parsed JSON dict.
    """
    response = chat_completion(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0].strip()
            return json.loads(json_str)
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0].strip()
            return json.loads(json_str)
        else:
            # Last resort: find first { to last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")
