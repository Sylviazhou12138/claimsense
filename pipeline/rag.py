"""
RAG (Retrieval-Augmented Generation) module.
Manages a local ChromaDB collection of historical claim cases.
On first run, seeds the collection with 5 synthetic cases.
At query time, embeds the current claim via Gemini and retrieves the 2 closest matches.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from google import genai

from models.schemas import SimilarCase

_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
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


def _get_collection(persist: bool = True) -> chromadb.Collection:
    """Return (and create if needed) the claim_history ChromaDB collection."""
    if persist:
        _DB_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(_DB_PATH))
    else:
        client = chromadb.EphemeralClient()

    embed_fn = _get_embed_fn()
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def seed_collection(persist: bool = True) -> None:
    """
    Populate the collection with seed cases if not already present.
    Safe to call multiple times (idempotent).
    """
    collection = _get_collection(persist=persist)
    existing_ids = set(collection.get(ids=[c["id"] for c in _SEED_CASES])["ids"])
    new_cases = [c for c in _SEED_CASES if c["id"] not in existing_ids]
    if not new_cases:
        return
    collection.add(
        ids=[c["id"] for c in new_cases],
        documents=[c["text"] for c in new_cases],
        metadatas=[c["metadata"] for c in new_cases],
    )


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
    collection = _get_collection()
    seed_collection()  # ensure seed data exists

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
    """Add a new case to the persistent collection (e.g. after a claim is resolved)."""
    collection = _get_collection()
    collection.add(
        ids=[case_id],
        documents=[text],
        metadatas=[metadata or {}],
    )
