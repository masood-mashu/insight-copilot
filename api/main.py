"""FastAPI application for Insight Copilot."""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from orchestrator.pipeline import run_pipeline
from utils.qdrant_client import ensure_collection, get_qdrant_client

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Insight Copilot API", version="0.1.0")


@app.get("/health")
def health_check() -> dict:
    """Return a basic health check payload."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)) -> dict:
    """Analyze an uploaded CSV and return the combined pipeline report."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    raw_bytes = await file.read()
    if not raw_bytes.strip():
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        logger.exception("Failed to parse uploaded CSV")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc

    try:
        report = await asyncio.to_thread(run_pipeline, df)
        return report
    except Exception as exc:
        logger.exception("Pipeline execution failed")
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Pipeline execution failed",
                "error": str(exc),
            },
        )


@app.get("/memory/history")
def memory_history() -> JSONResponse:
    """List previously analyzed dataset signatures stored in Qdrant."""
    try:
        client = get_qdrant_client()
        ensure_collection(client)
        scroll_result = client.scroll(
            collection_name="insight_copilot_datasets",
            with_payload=True,
            with_vectors=False,
            limit=100,
        )
        points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result
        history = []
        for point in points:
            payload: dict[str, Any] = point.payload or {}
            history.append(
                {
                    "signature": str(payload.get("dataset_signature", "unknown-dataset")),
                    "timestamp": str(payload.get("timestamp", "")),
                    "notes": str(payload.get("notes", "")),
                }
            )
        history.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return JSONResponse(content={"history": history})
    except Exception as exc:
        logger.exception("Failed to read memory history")
        return JSONResponse(content={"history": [], "error": str(exc)}, status_code=200)
