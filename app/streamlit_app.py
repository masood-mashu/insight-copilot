"""Streamlit interface for Insight Copilot."""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from orchestrator.pipeline import run_pipeline
from utils.qdrant_client import ensure_collection, get_qdrant_client

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
USE_LOCAL_PIPELINE = os.getenv("USE_LOCAL_PIPELINE", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}

st.set_page_config(
    page_title="Insight Copilot",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    """Render the Insight Copilot Streamlit UI."""
    st.title("Insight Copilot")
    st.caption("Autonomous EDA with profiling, insights, long-term memory, and recommendations.")

    with st.sidebar:
        st.subheader("Analysis History")
        for item in _fetch_history():
            st.write(f"- `{item['signature']}`")
            if item.get("timestamp"):
                st.caption(item["timestamp"])

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV to run the agent pipeline.")
        return

    if st.button("Analyze Dataset", type="primary"):
        with st.status("Running Insight Copilot pipeline...", expanded=True) as status:
            report = _analyze_uploaded_csv(uploaded_file, status)
            if report is None:
                status.update(label="Pipeline failed", state="error")
                return

            st.write("Profiler Agent complete")
            st.json(report["profiler"])
            st.write("Insight Agent complete")
            st.json(report["insight"])
            st.write("Memory Agent complete")
            st.json(report["memory"])
            st.write("Recommender Agent complete")
            st.json(report["recommendation"])
            status.update(label="Pipeline complete", state="complete")

        _render_report(report)


def _fetch_history() -> list[dict]:
    if USE_LOCAL_PIPELINE:
        return _fetch_history_from_qdrant()

    try:
        response = requests.get(f"{API_BASE_URL}/memory/history", timeout=30)
        response.raise_for_status()
        payload = response.json()
        return payload.get("history", [])
    except Exception:
        return []


def _analyze_uploaded_csv(uploaded_file, status) -> dict | None:
    if USE_LOCAL_PIPELINE:
        return _run_local_pipeline(uploaded_file, status)
    return _run_api_pipeline(uploaded_file, status)


def _run_local_pipeline(uploaded_file, status) -> dict | None:
    try:
        st.write("Reading uploaded CSV...")
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        st.write("Running local agent pipeline...")
        return run_pipeline(df)
    except Exception as exc:
        st.error("Local pipeline failed")
        st.code(str(exc), language="text")
        return None


def _run_api_pipeline(uploaded_file, status) -> dict | None:
    st.write("Uploading dataset to the API...")
    response = requests.post(
        f"{API_BASE_URL}/analyze",
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
        timeout=300,
    )
    if not response.ok:
        st.error(f"API returned {response.status_code}")
        st.code(_format_backend_error(response), language="json")
        return None
    return response.json()


def _fetch_history_from_qdrant() -> list[dict]:
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
            payload = point.payload or {}
            history.append(
                {
                    "signature": str(payload.get("dataset_signature", "unknown-dataset")),
                    "timestamp": str(payload.get("timestamp", "")),
                    "notes": str(payload.get("notes", "")),
                }
            )
        history.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return history
    except Exception:
        return []


def _render_report(report: dict) -> None:
    if report["insight"].get("degraded"):
        err_msg = report["insight"].get("error_details", "No error details available.")
        st.warning(
            f"⚠️ The Insight Agent returned degraded output for this run. "
            f"The Gemini API may have timed out or returned an unparseable response. "
            f"Error details: `{err_msg}`. "
            "Re-run the analysis for richer findings."
        )

    st.subheader("Summary")
    st.write(report["insight"]["summary"])

    st.subheader("Findings")
    for finding in report["insight"]["findings"]:
        icon = _finding_icon(finding["type"])
        st.markdown(f"{icon} **{finding['type'].title()}**: {finding['detail']}")

    st.subheader("Memory Context")
    st.info(report["memory"]["retrieved_context"])

    st.subheader("Recommendations")
    recommendation = report["recommendation"]
    st.write(f"**Suggested approach:** {recommendation['suggested_approach']}")
    st.write("**Suggested models**")
    st.write("\n".join(f"- {model}" for model in recommendation["suggested_models"]))
    st.write("**Cleaning steps**")
    st.write("\n".join(f"- {step}" for step in recommendation["cleaning_steps"]))
    st.write("**Visualization suggestions**")
    st.write("\n".join(f"- {viz}" for viz in recommendation["visualization_suggestions"]))


def _finding_icon(finding_type: str) -> str:
    mapping = {
        "correlation": "[Correlation]",
        "anomaly": "[Anomaly]",
        "pattern": "[Pattern]",
    }
    return mapping.get(finding_type, "-")


def _format_backend_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        return str(payload)
    except Exception:
        return response.text or "Unknown backend error"


if __name__ == "__main__":
    main()
