"""
Summary generation module.
Assembles all pipeline outputs into the final ClaimBrief.
Calls Gemini to write the adjuster_brief with inline document citations.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from models.schemas import (
    ClaimBrief,
    ClassificationResult,
    ComplianceResult,
    DocumentChunk,
    ExtractedEntities,
    SimilarCase,
)
from pipeline.extract import parties_list

_SYSTEM_PROMPT = (
    "You are a claims adjuster assistant. Write a concise 2-3 sentence summary "
    "for the adjuster based on the claim data provided. The summary must:\n"
    "- State the claim type, key facts, and the risk level.\n"
    "- Include at least one inline document citation in the form [Doc p.N] where N "
    "is the page number, or [Doc para N] if no page numbers are available.\n"
    "- Flag any missing fields or compliance concerns if present.\n"
    "Keep the tone professional and factual. Do not exceed 100 words."
)

_DEFAULT_MODEL = "gemini-2.5-flash"


def _build_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _citation_hint(chunks: list[DocumentChunk]) -> str:
    """Build a compact hint string showing page/paragraph references."""
    refs: list[str] = []
    for c in chunks[:6]:
        if c.source_page:
            refs.append(f"p.{c.source_page}: \"{c.text[:80]}...\"")
        else:
            refs.append(f"para {c.chunk_index + 1}: \"{c.text[:80]}...\"")
    return "\n".join(refs)


def _similar_case_summaries(cases: list[SimilarCase]) -> list[str]:
    """Convert SimilarCase objects to short human-readable strings."""
    summaries: list[str] = []
    for case in cases:
        pct = int(case.similarity_score * 100)
        summaries.append(f"[{case.case_id}] ({pct}% match) {case.summary[:120]}")
    return summaries


def generate_brief(
    chunks: list[DocumentChunk],
    entities: ExtractedEntities,
    classification: ClassificationResult,
    compliance: ComplianceResult,
    similar_cases: list[SimilarCase],
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Call Gemini to write the adjuster_brief with inline citations.

    Returns:
        The adjuster brief as a plain string.
    """
    client = _build_client()
    citation_context = _citation_hint(chunks)
    missing_str = (
        ", ".join(compliance.missing_fields) if compliance.missing_fields else "None"
    )

    user_content = (
        f"Claim Type: {classification.claim_type} - {classification.claim_subtype}\n"
        f"Risk Level: {classification.risk_level} (score {classification.risk_score}/100)\n"
        f"Risk Reasoning: {classification.risk_reasoning}\n"
        f"Incident Date: {entities.incident_date or 'Unknown'}\n"
        f"Claimed Amount: {entities.claimed_amount or 'Unknown'}\n"
        f"Parties: {', '.join(parties_list(entities)) or 'Unknown'}\n"
        f"Missing Fields: {missing_str}\n\n"
        f"Document citation reference:\n{citation_context}"
    )

    response = client.models.generate_content(
        model=model,
        contents=f"{_SYSTEM_PROMPT}\n\n{user_content}",
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=200,
        ),
    )

    return (response.text or "").strip()


def build_claim_brief(
    chunks: list[DocumentChunk],
    entities: ExtractedEntities,
    classification: ClassificationResult,
    compliance: ComplianceResult,
    similar_cases: list[SimilarCase],
    model: str = _DEFAULT_MODEL,
) -> ClaimBrief:
    """
    Orchestrate all pipeline outputs into the final ClaimBrief object.

    Args:
        chunks: Document chunks from ingest.
        entities: Extracted entities.
        classification: Risk classification result.
        compliance: Compliance check result.
        similar_cases: Retrieved historical cases.
        model: Gemini model for brief generation.

    Returns:
        Fully populated ClaimBrief.
    """
    brief_text = generate_brief(
        chunks, entities, classification, compliance, similar_cases, model
    )

    return ClaimBrief(
        claim_type=classification.claim_type,
        claim_subtype=classification.claim_subtype,
        risk_level=classification.risk_level,
        risk_score=classification.risk_score,
        incident_date=entities.incident_date or "Unknown",
        claimed_amount=entities.claimed_amount or "Unknown",
        parties_involved=parties_list(entities),
        policy_number=entities.policy_number,
        missing_fields=compliance.missing_fields,
        rag_similar_cases=_similar_case_summaries(similar_cases),
        adjuster_brief=brief_text,
        risk_reasoning=classification.risk_reasoning,
        compliance_notes=compliance.compliance_notes,
        extracted_entities=entities,
    )
