"""LLM-backed insight generation for profiled datasets."""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ALLOWED_FINDING_TYPES = {"correlation", "anomaly", "pattern"}


class Finding(BaseModel):
    """Validated finding emitted by the insight agent."""

    type: Literal["correlation", "anomaly", "pattern"]
    detail: str = Field(min_length=1)


class InsightResponse(BaseModel):
    """Validated response schema for Gemini insight output."""

    summary: str = Field(min_length=1)
    findings: list[Finding]


def run_insight_agent(profiler_output: dict) -> dict:
    """Generate plain-English insights from profiler output.

    Args:
        profiler_output: A profiler dictionary matching the profiler agent schema.

    Returns:
        A dictionary with keys `summary` and `findings`, where `findings` is a
        list of objects containing `type` and `detail`.
    """
    if not isinstance(profiler_output, dict):
        raise TypeError("profiler_output must be a dict")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not configured; returning fallback insight output")
        return _fallback_insight_output(profiler_output)

    client = genai.Client(api_key=api_key)
    prompt = _build_prompt(profiler_output)

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=InsightResponse,
                    system_instruction=(
                        "You are an expert exploratory data analyst. "
                        "Return ONLY valid JSON with this exact schema: "
                        '{"summary": "string", "findings": [{"type": "correlation|anomaly|pattern", "detail": "string"}]}. '
                        "Do not include markdown fences or extra text."
                    ),
                ),
            )
            logger.info("Gemini raw response (attempt %s): %.500s", attempt + 1, response.text)
            validated = _validate_insight_output(response.parsed, profiler_output)
            logger.debug("Insight agent completed on attempt %s", attempt + 1)
            return validated
        except Exception:
            logger.exception("Insight agent attempt %s failed", attempt + 1)

    logger.warning("All Gemini attempts exhausted; returning degraded fallback output")
    return _fallback_insight_output(profiler_output)


def _build_prompt(profiler_output: dict[str, Any]) -> str:
    return (
        "Analyze this dataset profile and produce a concise 2-3 sentence overview "
        "plus 1-3 findings. Prefer issues that can be inferred directly from the profile. "
        "Only label a finding as correlation when the evidence is explicit in the provided profile; "
        "otherwise prefer anomaly or pattern.\n\n"
        f"Profiler output:\n{profiler_output}"
    )


def _validate_insight_output(payload: Any, profiler_output: dict | None = None) -> dict:
    if isinstance(payload, InsightResponse):
        insight = payload
    elif isinstance(payload, dict):
        insight = InsightResponse.model_validate(payload)
    else:
        raise ValueError("Insight response must be a JSON object")

    normalized_findings = [
        {"type": item.type, "detail": item.detail.strip()}
        for item in insight.findings[:3]
        if item.type in ALLOWED_FINDING_TYPES and item.detail.strip()
    ]

    findings_degraded = False
    if not normalized_findings:
        logger.warning("All Gemini findings filtered out; substituting fallback findings")
        normalized_findings = _fallback_insight_output(profiler_output or {})["findings"]
        findings_degraded = True

    result = {
        "summary": insight.summary.strip(),
        "findings": normalized_findings,
    }
    if findings_degraded:
        result["degraded"] = True
    return result


def _fallback_insight_output(profiler_output: dict[str, Any]) -> dict:
    n_rows = int(profiler_output.get("n_rows", 0))
    n_cols = int(profiler_output.get("n_cols", 0))
    logger.info("Fallback insight path invoked for %d-row, %d-col dataset", n_rows, n_cols)
    columns = profiler_output.get("columns", [])
    duplicate_rows = int(profiler_output.get("duplicate_rows", 0))
    categorical_candidates = [
        column
        for column in columns
        if isinstance(column, dict)
        and "stats" not in column
        and int(column.get("unique_count", 0)) > 1
        and int(column.get("unique_count", 0)) <= 10
    ]
    binary_or_low_card_numeric = [
        column
        for column in columns
        if isinstance(column, dict)
        and "stats" in column
        and int(column.get("unique_count", 0)) > 1
        and int(column.get("unique_count", 0)) <= 5
    ]

    high_null_columns = [
        column["name"]
        for column in columns
        if isinstance(column, dict) and float(column.get("null_pct", 0.0)) >= 20.0
    ]

    findings = []
    if high_null_columns:
        findings.append(
            {
                "type": "anomaly",
                "detail": (
                    f"Columns with notable missingness include {', '.join(map(str, high_null_columns[:3]))}."
                ),
            }
        )
    if duplicate_rows > 0:
        findings.append(
            {
                "type": "pattern",
                "detail": f"The dataset contains {duplicate_rows} duplicate rows that may affect analysis quality.",
            }
        )
    if categorical_candidates:
        label_name = str(categorical_candidates[0].get("name", "label"))
        label_count = int(categorical_candidates[0].get("unique_count", 0))
        findings.append(
            {
                "type": "pattern",
                "detail": (
                    f"The column '{label_name}' has {label_count} distinct categories and may be a natural target "
                    "or grouping variable for classification and comparison plots."
                ),
            }
        )
    elif binary_or_low_card_numeric:
        label_name = str(binary_or_low_card_numeric[0].get("name", "label"))
        label_count = int(binary_or_low_card_numeric[0].get("unique_count", 0))
        findings.append(
            {
                "type": "pattern",
                "detail": (
                    f"The numeric column '{label_name}' has only {label_count} distinct values, which suggests a "
                    "classification-style target or segmentation variable."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "type": "pattern",
                "detail": "The profile is structurally complete, but deeper relationships require column-level analysis or modeling.",
            }
        )

    summary = f"This dataset contains {n_rows} rows and {n_cols} columns. "
    if categorical_candidates:
        summary += (
            f"It mixes measured features with a low-cardinality categorical column like "
            f"'{categorical_candidates[0].get('name', 'label')}', which makes it a good fit for grouped analysis "
            "or classification-style exploration. "
        )
    elif binary_or_low_card_numeric:
        summary += (
            f"It includes a low-cardinality numeric field like '{binary_or_low_card_numeric[0].get('name', 'label')}', "
            "which may serve as a target or segment variable. "
        )
    else:
        summary += "The current overview is based on schema, completeness, uniqueness, and basic numeric summaries."

    return {"summary": summary, "findings": findings[:3], "degraded": True}
