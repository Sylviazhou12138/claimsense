"""
Entity extraction module.
Uses Gemini structured output (response_schema=ExtractedEntities) to pull
key fields from a claim document without free-text parsing.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from models.schemas import ExtractedEntities

_SYSTEM_PROMPT = (
    "You are an insurance claim analyst. Extract structured information from the "
    "provided claim document text. Be precise: only fill fields you can directly "
    "support from the document. Use null for fields not found. "
    "For claimed_amount, include the currency symbol (e.g. \"$12,500\"). "
    "For dates, prefer ISO-8601 (YYYY-MM-DD) where possible. "
    "For raw_text_excerpt, copy the single most evidential sentence (≤200 chars)."
)

_DEFAULT_MODEL = "gemini-2.5-flash"


def _build_client() -> genai.Client:
    """Return a Gemini client using GEMINI_API_KEY from the environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in the environment.")
    return genai.Client(api_key=api_key)


def extract_entities(
    document_text: str, model: str = _DEFAULT_MODEL
) -> ExtractedEntities:
    """
    Call Gemini with structured output to extract claim entities.

    Args:
        document_text: Full text (or concatenated chunks) of the claim document.
        model: Gemini model ID to use.

    Returns:
        Validated ExtractedEntities instance.
    """
    client = _build_client()
    truncated = document_text[:12_000]

    response = client.models.generate_content(
        model=model,
        contents=f"{_SYSTEM_PROMPT}\n\nClaim document:\n\n{truncated}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedEntities,
            temperature=0,
        ),
    )

    return ExtractedEntities.model_validate_json(response.text)


def parties_list(entities: ExtractedEntities) -> list[str]:
    """Aggregate all named parties from extracted entities into a flat list."""
    parties: list[str] = []
    if entities.claimant_name:
        parties.append(entities.claimant_name)
    if entities.insured_name and entities.insured_name != entities.claimant_name:
        parties.append(entities.insured_name)
    parties.extend(entities.third_parties)
    return list(dict.fromkeys(parties))  # deduplicate while preserving order
