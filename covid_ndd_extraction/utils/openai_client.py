"""Shared OpenAI client factory."""

from __future__ import annotations

from openai import OpenAI

from covid_ndd_extraction.config import settings


def get_openai_client(api_key: str | None = None) -> OpenAI:
    """Return an authenticated OpenAI client.

    Uses *api_key* if provided, otherwise falls back to ``settings.openai_api_key``
    (which is read from the ``OPENAI_API_KEY`` environment variable / .env file).
    The OpenAI SDK itself will also check ``OPENAI_API_KEY`` if both are None/empty.
    """
    key = api_key or settings.openai_api_key or None
    return OpenAI(api_key=key)
