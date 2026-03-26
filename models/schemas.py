"""
Pydantic data models for ClaimSense.
Defines all structured types shared across pipeline modules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Extraction layer ──────────────────────────────────────────────────────────

class ExtractedEntities(BaseModel):
    """Raw entities pulled from the claim document by the LLM."""

    incident_date: str | None = Field(
        None, description="Date the incident occurred, ISO-8601 preferred"
    )
    reported_date: str | None = Field(
        None, description="Date the claim was reported"
    )
    policy_number: str | None = Field(None, description="Policy or contract number")
    claimed_amount: str | None = Field(
        None, description="Total amount claimed, include currency symbol"
    )
    claimant_name: str | None = Field(None, description="Full name of the claimant")
    insured_name: str | None = Field(None, description="Full name of the insured party")
    third_parties: list[str] = Field(
        default_factory=list,
        description="Names of third parties, witnesses, or involved persons",
    )
    location: str | None = Field(None, description="Location where the incident occurred")
    vehicle_info: str | None = Field(
        None, description="Vehicle make/model/plate if relevant"
    )
    injury_description: str | None = Field(
        None, description="Description of injuries, if any"
    )
    property_description: str | None = Field(
        None, description="Description of damaged property, if any"
    )
    coverage_type: str | None = Field(
        None, description="Type of coverage mentioned in the document"
    )
    raw_text_excerpt: str = Field(
        "", description="Short excerpt (≤200 chars) most relevant to the claim"
    )


# ── Classification layer ──────────────────────────────────────────────────────

ClaimType = Literal["Auto", "Property", "Medical", "Liability", "Life", "Other"]
ClaimSubtype = str  # free-form; e.g. "Collision", "Fire Damage", "Hospitalisation"
RiskLevel = Literal["Routine", "Needs Review", "Escalate"]


class ClassificationResult(BaseModel):
    """Risk scoring and claim categorisation output."""

    claim_type: ClaimType
    claim_subtype: str = Field(description="More specific sub-category")
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100, description="0 = minimal risk, 100 = extreme risk")
    risk_reasoning: str = Field(
        description="One sentence explaining the primary risk driver"
    )


# ── Compliance layer ──────────────────────────────────────────────────────────

class ComplianceResult(BaseModel):
    """Fields required for processing and which are missing."""

    required_fields: list[str] = Field(
        description="All fields that must be present for this claim type"
    )
    missing_fields: list[str] = Field(
        description="Subset of required_fields that could not be found"
    )
    compliance_notes: str = Field(
        default="", description="Optional free-text compliance observations"
    )


# ── RAG layer ─────────────────────────────────────────────────────────────────

class SimilarCase(BaseModel):
    """A retrieved historical case from ChromaDB."""

    case_id: str
    summary: str
    similarity_score: float = Field(ge=0.0, le=1.0)


# ── Final output ──────────────────────────────────────────────────────────────

class ClaimBrief(BaseModel):
    """
    Top-level structured output returned to the adjuster.
    Aggregates extraction, classification, compliance, and RAG results.
    """

    claim_type: str
    claim_subtype: str
    risk_level: RiskLevel
    risk_score: int = Field(ge=0, le=100)
    incident_date: str
    claimed_amount: str
    parties_involved: list[str]
    policy_number: str | None
    missing_fields: list[str]
    rag_similar_cases: list[str] = Field(
        description="Short human-readable summaries of the 2 most similar historical cases"
    )
    adjuster_brief: str = Field(
        description="2-3 sentence natural-language summary with inline document citations"
    )
    # Internal detail fields (not surfaced in card view but available in JSON)
    risk_reasoning: str = ""
    compliance_notes: str = ""
    extracted_entities: ExtractedEntities | None = None


# ── Ingest layer ──────────────────────────────────────────────────────────────

class DocumentChunk(BaseModel):
    """A single text chunk produced by the ingest pipeline."""

    chunk_index: int
    text: str
    source_page: int | None = None  # available for PDF inputs
    char_start: int = 0
    char_end: int = 0
