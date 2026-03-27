---
title: ClaimSense
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ClaimSense — AI-Powered Insurance Claim Triage

ClaimSense ingests an insurance claim document (PDF or plain text) and produces a
structured **ClaimBrief**: a risk-scored, entity-enriched, compliance-checked review
package ready for the adjuster.

## Quick Start (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY

# 3. Launch the Gradio UI
python app.py

# 4. Launch with a preloaded demo claim
python app.py --demo
```

The UI runs at `http://localhost:7860` by default.

## Pipeline Overview

```
Document (PDF / text)
        │
        ▼
  pipeline/ingest.py       → list[DocumentChunk]
        │
        ▼
  pipeline/extract.py      → ExtractedEntities      (Gemini structured output)
        │
        ├──▶ pipeline/classify.py   → ClassificationResult  (Gemini structured output)
        │
        ├──▶ pipeline/compliance.py → ComplianceResult      (rule-based, no LLM)
        │
        ├──▶ pipeline/rag.py        → list[SimilarCase]     (ChromaDB + gemini-embedding-001)
        │
        └──▶ pipeline/summarize.py  → ClaimBrief            (Gemini, free-text brief)
```

## Output Structure (`ClaimBrief`)

| Field | Description |
|---|---|
| `claim_type` | Auto / Property / Medical / Liability / Life / Other |
| `claim_subtype` | Fine-grained category (e.g. "Rear-end Collision") |
| `risk_level` | `Routine` / `Needs Review` / `Escalate` |
| `risk_score` | Integer 0–100 |
| `incident_date` | ISO-8601 date or "Unknown" |
| `claimed_amount` | Currency string (e.g. "$20,346") |
| `parties_involved` | List of named parties |
| `policy_number` | Policy/contract number or null |
| `missing_fields` | Fields required but absent from the document |
| `rag_similar_cases` | 2 closest historical cases with similarity % |
| `adjuster_brief` | 2-3 sentence natural-language summary with citations |

## Project Structure

```
claimsense/
├── app.py                  # Gradio entry point
├── pipeline/
│   ├── ingest.py           # PDF/text parsing + chunking
│   ├── extract.py          # Gemini entity extraction
│   ├── classify.py         # Risk scoring + claim classification
│   ├── compliance.py       # Missing-field rule checks
│   ├── rag.py              # In-memory ChromaDB retrieval (seeded with 5 cases)
│   └── summarize.py        # Adjuster brief generation
├── models/
│   └── schemas.py          # All Pydantic models
├── data/
│   └── sample_claims/      # 3 synthetic demo claims (.txt)
├── .env.example
└── requirements.txt
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |

On Hugging Face Spaces, set this as a **Secret** in the Space settings (Settings → Variables and secrets).

## Calling Pipeline Modules Independently

Each module exposes a clean public function:

```python
from pipeline.ingest import ingest, full_text
from pipeline.extract import extract_entities
from pipeline.classify import classify_claim
from pipeline.compliance import check_compliance
from pipeline.rag import retrieve_similar
from pipeline.summarize import build_claim_brief
```
