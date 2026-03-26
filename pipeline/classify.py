"""
Risk classification module.
Sends the document text and extracted entities to Gemini to produce
a claim type, subtype, risk score (0-100), and a risk level label.
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types

from models.schemas import ClassificationResult, ExtractedEntities

_SYSTEM_PROMPT = (
    "You are a senior insurance underwriter. Given a claim document and the "
    "extracted entities, determine:\n"
    "1. claim_type: one of Auto | Property | Medical | Liability | Life | Other\n"
    "2. claim_subtype: a specific sub-category (e.g. \"Collision\", \"Fire Damage\", "
    "\"Hospitalisation\", \"Slip and Fall\")\n"
    "3. risk_score: integer 0-100 reflecting overall claim risk "
    "(fraud indicators, severity, coverage gaps)\n"
    "4. risk_level: \"Routine\" (0-39) | \"Needs Review\" (40-69) | \"Escalate\" (70-100)\n"
    "5. risk_reasoning: exactly ONE sentence explaining the primary risk driver\n\n"
    "Be calibrated: most everyday claims score 10-40. Reserve 70+ for clear red "
    "flags (inconsistent dates, amounts far above policy limits, prior fraud "
    "indicators, catastrophic injuries)."
)

_DEFAULT_MODEL = "gemini-2.5-flash"


def _build_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _entities_summary(entities: ExtractedEntities) -> str:
    """Render extracted entities as a compact JSON string for the prompt."""
    return json.dumps(entities.model_dump(exclude_none=True), ensure_ascii=False)


def classify_claim(
    document_text: str,
    entities: ExtractedEntities,
    model: str = _DEFAULT_MODEL,
) -> ClassificationResult:
    """
    Classify a claim and compute its risk score.

    Args:
        document_text: Full or truncated claim text for context.
        entities: Pre-extracted entities from the extraction step.
        model: Gemini model ID to use.

    Returns:
        Validated ClassificationResult instance.
    """
    client = _build_client()

    contents = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Extracted entities:\n{_entities_summary(entities)}\n\n"
        f"Document excerpt (first 4000 chars):\n{document_text[:4_000]}"
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ClassificationResult,
            temperature=0,
        ),
    )

    return ClassificationResult.model_validate_json(response.text)
