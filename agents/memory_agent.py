"""Long-term dataset memory backed by Qdrant."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid5, NAMESPACE_URL

from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

from utils.qdrant_client import ensure_collection, get_qdrant_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "insight_copilot_datasets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
SIMILARITY_THRESHOLD = 0.8
_embedding_model: SentenceTransformer | None = None


def run_memory_agent(profiler_output: dict) -> dict:
    """Store and retrieve similar dataset signatures from Qdrant."""
    if not isinstance(profiler_output, dict):
        raise TypeError("profiler_output must be a dict")

    signature_text = _build_signature_text(profiler_output)
    dataset_signature = _build_dataset_label(profiler_output)

    try:
        client = get_qdrant_client()
        ensure_collection(
            client,
            collection_name=COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
        )

        model = _get_embedding_model()
        embedding = model.encode(signature_text).tolist()

        search_results = _search_similar_points(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding,
            limit=3,
        )

        similar_past_datasets = [
            {
                "signature": str(hit.payload.get("dataset_signature", "unknown-dataset")),
                "similarity": float(hit.score),
                "notes": str(hit.payload.get("notes", "")),
            }
            for hit in search_results
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, signature_text)),
                    vector=embedding,
                    payload={
                        "dataset_signature": dataset_signature,
                        "signature_text": signature_text,
                        "notes": "",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )

        return {
            "embedded": True,
            "dataset_signature": dataset_signature,
            "similar_past_datasets": similar_past_datasets,
            "retrieved_context": _build_retrieved_context(
                similar_past_datasets=similar_past_datasets,
            ),
        }
    except Exception as exc:
        logger.exception("Memory agent failed")
        return {
            "embedded": False,
            "dataset_signature": dataset_signature,
            "similar_past_datasets": [],
            "retrieved_context": f"Memory storage unavailable: {exc}",
        }


def update_memory_notes(profiler_output: dict, insight_summary: str) -> None:
    """Update the stored Qdrant point with a short insight summary for future recall."""
    if not insight_summary:
        return
    try:
        client = get_qdrant_client()
        signature_text = _build_signature_text(profiler_output)
        point_id = str(uuid5(NAMESPACE_URL, signature_text))
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"notes": insight_summary[:200]},
            points=[point_id],
        )
        logger.debug("Updated memory notes for point %s", point_id)
    except Exception:
        logger.exception("Failed to update memory notes")


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(f"Embedding model '{EMBEDDING_MODEL_NAME}' is unavailable: {exc}") from exc
    return _embedding_model


def _search_similar_points(
    client: Any,
    collection_name: str,
    embedding: list[float],
    limit: int,
) -> list[Any]:
    """Search Qdrant using whichever query API the installed SDK exposes."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=limit,
            with_payload=True,
        )
        return list(getattr(response, "points", response))

    if hasattr(client, "search"):
        return list(
            client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
            )
        )

    raise RuntimeError("Qdrant client does not expose query_points or search")


def _build_signature_text(profiler_output: dict[str, Any]) -> str:
    n_rows = int(profiler_output.get("n_rows", 0))
    n_cols = int(profiler_output.get("n_cols", 0))
    columns = profiler_output.get("columns", [])

    column_descriptors = []
    for column in columns:
        name = str(column.get("name", "unknown"))
        dtype = str(column.get("dtype", "unknown"))
        column_descriptors.append(f"{name}:{dtype}")

    return (
        f"rows={n_rows}; cols={n_cols}; "
        f"columns=[{', '.join(column_descriptors)}]"
    )


def _build_dataset_label(profiler_output: dict[str, Any]) -> str:
    n_cols = int(profiler_output.get("n_cols", 0))
    dtypes = [
        str(column.get("dtype", "unknown")).lower()
        for column in profiler_output.get("columns", [])
    ]

    has_numeric = any(
        token in dtype
        for dtype in dtypes
        for token in ("int", "float", "double", "decimal", "number")
    )
    has_datetime = any("date" in dtype or "time" in dtype for dtype in dtypes)
    has_boolean = any("bool" in dtype for dtype in dtypes)
    has_text = any(
        token in dtype for dtype in dtypes for token in ("object", "string", "str", "category")
    )

    type_labels = []
    if has_numeric:
        type_labels.append("numeric")
    if has_text:
        type_labels.append("categorical")
    if has_datetime:
        type_labels.append("datetime")
    if has_boolean:
        type_labels.append("boolean")
    if not type_labels:
        type_labels.append("unknown")

    if len(type_labels) > 1:
        type_summary = "mixed-" + "-".join(type_labels)
    else:
        type_summary = type_labels[0]

    return f"{n_cols}-col-{type_summary}"


def _build_retrieved_context(
    similar_past_datasets: list[dict[str, Any]],
) -> str:
    if not similar_past_datasets:
        return "No similar dataset found"

    top_match = similar_past_datasets[0]
    if float(top_match.get("similarity", 0.0)) < SIMILARITY_THRESHOLD:
        return "No highly similar dataset found in memory. This appears to be a new pattern."

    notes = str(top_match.get("notes", "")).strip() or "none recorded"
    return f"Similar to {top_match['signature']}, past insights: {notes}"
