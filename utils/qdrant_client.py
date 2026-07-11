"""Helpers for connecting to and preparing Qdrant."""

from __future__ import annotations

import os

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client configured from environment variables."""
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url:
        raise RuntimeError("QDRANT_URL environment variable is required")
    if not qdrant_api_key:
        raise RuntimeError("QDRANT_API_KEY environment variable is required")

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = "insight_copilot_datasets",
    vector_size: int = 384,
) -> None:
    """Create the Qdrant collection when it does not already exist."""
    existing_collections = {
        collection.name for collection in client.get_collections().collections
    }
    if collection_name in existing_collections:
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def fetch_dataset_history(
    client: QdrantClient,
    collection_name: str = "insight_copilot_datasets",
    limit: int = 100,
) -> list[dict]:
    """Fetch and format previously analyzed dataset signatures, newest first.

    Shared by the FastAPI `/memory/history` route and the Streamlit sidebar so
    both surfaces stay in sync instead of maintaining duplicate scroll logic.
    """
    scroll_result = client.scroll(
        collection_name=collection_name,
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )
    points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result

    history = []
    for point in points:
        payload: dict = point.payload or {}
        history.append(
            {
                "signature": str(payload.get("dataset_signature", "unknown-dataset")),
                "timestamp": str(payload.get("timestamp", "")),
                "notes": str(payload.get("notes", "")),
            }
        )
    history.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return history
