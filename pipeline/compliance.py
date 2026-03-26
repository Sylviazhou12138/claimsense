"""
Compliance checking module.
Rule-based: compares extracted entities against a per-claim-type
required-fields checklist and reports any missing items.
No LLM call needed — this is a deterministic data-quality gate.
"""

from __future__ import annotations

from models.schemas import ClassificationResult, ComplianceResult, ExtractedEntities

# Required fields per claim type.
# Keys map to attribute names on ExtractedEntities.
_COMMON_REQUIRED: list[str] = [
    "incident_date",
    "claimed_amount",
    "claimant_name",
    "policy_number",
]

_TYPE_EXTRA_REQUIRED: dict[str, list[str]] = {
    "Auto": ["vehicle_info", "location"],
    "Property": ["location", "property_description"],
    "Medical": ["claimant_name", "injury_description"],
    "Liability": ["location", "third_parties"],
    "Life": ["insured_name", "incident_date"],
    "Other": [],
}

# Human-readable labels for the JSON field names
_FIELD_LABELS: dict[str, str] = {
    "incident_date": "Incident Date",
    "reported_date": "Reported Date",
    "claimed_amount": "Claimed Amount",
    "claimant_name": "Claimant Name",
    "insured_name": "Insured Name",
    "policy_number": "Policy Number",
    "vehicle_info": "Vehicle Information",
    "location": "Incident Location",
    "injury_description": "Injury Description",
    "property_description": "Property Description",
    "third_parties": "Third-Party Information",
}


def _is_field_present(entities: ExtractedEntities, field: str) -> bool:
    """Return True if the field has a non-empty value on the entities object."""
    value = getattr(entities, field, None)
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value).strip())


def _required_fields_for(claim_type: str) -> list[str]:
    """Combine common + type-specific required fields."""
    extras = _TYPE_EXTRA_REQUIRED.get(claim_type, [])
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for f in _COMMON_REQUIRED + extras:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def check_compliance(
    entities: ExtractedEntities,
    classification: ClassificationResult,
) -> ComplianceResult:
    """
    Identify missing required fields for the given claim type.

    Args:
        entities: Extracted document entities.
        classification: Output from the classify step.

    Returns:
        ComplianceResult with lists of required and missing fields.
    """
    required = _required_fields_for(classification.claim_type)
    missing_keys = [f for f in required if not _is_field_present(entities, f)]

    missing_labels = [_FIELD_LABELS.get(k, k) for k in missing_keys]
    required_labels = [_FIELD_LABELS.get(k, k) for k in required]

    notes = ""
    if missing_keys:
        notes = (
            f"This {classification.claim_type} claim is missing "
            f"{len(missing_keys)} required field(s). "
            "Adjuster should request supplemental documentation before processing."
        )

    return ComplianceResult(
        required_fields=required_labels,
        missing_fields=missing_labels,
        compliance_notes=notes,
    )
