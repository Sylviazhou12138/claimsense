"""
RAG (Retrieval-Augmented Generation) module.
Manages a ChromaDB collection of historical claim cases.
Uses an in-memory (ephemeral) client so it works on Hugging Face Spaces
and other stateless environments without a persistent filesystem.
Seeds the collection with 5 synthetic cases on every startup.
At query time, embeds the current claim via Gemini and retrieves the 2 closest matches.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from google import genai

from models.schemas import SimilarCase

# Module-level singleton so the collection is seeded only once per process
_collection: chromadb.Collection | None = None

_COLLECTION_NAME = "claim_history"
_EMBED_MODEL = "gemini-embedding-001"  # Gemini embedding model
_TOP_K = 2

# ── Seed data ─────────────────────────────────────────────────────────────────

_SEED_CASES: list[dict[str, Any]] = [
    {
        "id": "HIST-001",
        "text": (
            "Auto collision claim. Claimant rear-ended at a stop light. "
            "Both vehicles sustained moderate damage. No injuries. "
            "Claimed $8,200. Policy verified. Settled within 14 days."
        ),
        "metadata": {"type": "Auto", "subtype": "Rear-end Collision", "outcome": "Settled"},
    },
    {
        "id": "HIST-002",
        "text": (
            "Residential fire damage claim. Kitchen fire caused by unattended cooking. "
            "Structural damage to one room. Claimed $45,000. "
            "Fire department report corroborated. Settled after inspection."
        ),
        "metadata": {"type": "Property", "subtype": "Fire Damage", "outcome": "Settled"},
    },
    {
        "id": "HIST-003",
        "text": (
            "Medical hospitalisation claim. Claimant admitted for appendectomy. "
            "3-day stay, surgery and anaesthesia costs. Claimed $22,500. "
            "All medical records provided. Approved and paid."
        ),
        "metadata": {"type": "Medical", "subtype": "Surgery", "outcome": "Approved"},
    },
    {
        "id": "HIST-004",
        "text": (
            "Fraudulent auto theft claim. Vehicle reported stolen but later found "
            "undamaged at claimant's second property. Inconsistent statements. "
            "Claim denied after SIU investigation."
        ),
        "metadata": {"type": "Auto", "subtype": "Theft", "outcome": "Denied - Fraud"},
    },
    {
        "id": "HIST-005",
        "text": (
            "Slip-and-fall liability claim at commercial property. "
            "Claimant alleges wet floor without warning signage. "
            "Minor knee injury. Claimed $15,000. Liability confirmed by CCTV. "
            "Settled for $11,000."
        ),
        "metadata": {"type": "Liability", "subtype": "Slip and Fall", "outcome": "Settled"},
    },
]


# ── Gemini embedding function for ChromaDB ────────────────────────────────────

class GeminiEmbeddingFunction(EmbeddingFunction):  # type: ignore[type-arg]
    """ChromaDB-compatible embedding function backed by Gemini."""

    def __init__(self, api_key: str, model: str = _EMBED_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        result = self._client.models.embed_content(
            model=self._model,
            contents=list(input),
        )
        return [e.values for e in result.embeddings]


# ── Client + collection helpers ───────────────────────────────────────────────

def _get_embed_fn() -> GeminiEmbeddingFunction:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    return GeminiEmbeddingFunction(api_key=api_key)


def _get_or_init_collection() -> chromadb.Collection:
    """
    Return the singleton in-memory collection, seeding it on first call.
    Uses EphemeralClient so no filesystem access is required (works on HF Spaces).
    """
    global _collection
    if _collection is not None:
        return _collection

    client = chromadb.EphemeralClient()
    embed_fn = _get_embed_fn()
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    # Seed with historical cases
    collection.add(
        ids=[c["id"] for c in _SEED_CASES],
        documents=[c["text"] for c in _SEED_CASES],
        metadatas=[c["metadata"] for c in _SEED_CASES],
    )
    _collection = collection
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_similar(query_text: str, top_k: int = _TOP_K) -> list[SimilarCase]:
    """
    Embed query_text and return the top_k most similar historical cases.

    Args:
        query_text: Typically the full claim text or a condensed summary.
        top_k: Number of results to return.

    Returns:
        List of SimilarCase objects sorted by descending similarity.
    """
    collection = _get_or_init_collection()

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "distances", "metadatas"],
    )

    cases: list[SimilarCase] = []
    docs = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    for doc, dist, case_id in zip(docs, distances, ids):
        similarity = max(0.0, round(1.0 - float(dist), 4))
        cases.append(SimilarCase(case_id=case_id, summary=doc, similarity_score=similarity))

    return cases


def add_case(case_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
    """Add a new case to the in-memory collection for the current session."""
    collection = _get_or_init_collection()
    collection.add(
        ids=[case_id],
        documents=[text],
        metadatas=[metadata or {}],
    )
