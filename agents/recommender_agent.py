"""Recommendation synthesis for next-step analysis."""

from __future__ import annotations

from typing import Any


def run_recommender_agent(insight_output: dict, memory_output: dict) -> dict:
    """Synthesize analysis and memory context into actionable recommendations.

    Args:
        insight_output: The insight agent output containing summary and findings.
        memory_output: The memory agent output containing similar dataset context.

    Returns:
        A dictionary with suggested approach, models, cleaning steps, and
        visualization suggestions.
    """
    if not isinstance(insight_output, dict):
        raise TypeError("insight_output must be a dict")
    if not isinstance(memory_output, dict):
        raise TypeError("memory_output must be a dict")

    findings = insight_output.get("findings", [])
    dataset_signature = str(memory_output.get("dataset_signature", "unknown-dataset"))
    retrieved_context = str(memory_output.get("retrieved_context", ""))

    cleaning_steps = _build_cleaning_steps(findings)
    suggested_models = _build_models(dataset_signature, findings)
    visualization_suggestions = _build_visualizations(findings)
    suggested_approach = _build_approach(dataset_signature, retrieved_context)

    return {
        "suggested_approach": suggested_approach,
        "suggested_models": suggested_models,
        "cleaning_steps": cleaning_steps,
        "visualization_suggestions": visualization_suggestions,
    }


def _build_cleaning_steps(findings: list[Any]) -> list[str]:
    steps = []
    anomaly_text = " ".join(
        str(item.get("detail", ""))
        for item in findings
        if isinstance(item, dict) and str(item.get("type", "")).lower() == "anomaly"
    ).lower()

    if "missing" in anomaly_text or "null" in anomaly_text:
        steps.append("Review high-missing columns and impute or drop them based on business value")
    if "zero" in anomaly_text or "0" in anomaly_text:
        steps.append("Treat biologically implausible zero values as missing before modeling")
    if "outlier" in anomaly_text or "skew" in anomaly_text:
        steps.append("Review skewed numeric columns and cap or transform extreme outliers")
    if "duplicate" in anomaly_text:
        steps.append("Deduplicate repeated rows before downstream modeling")

    steps.extend(
        [
            "Standardize data types and validate categorical labels for consistency",
            "Check outliers in numeric columns before training or visualization",
        ]
    )
    return steps[:4]


def _build_models(dataset_signature: str, findings: list[Any]) -> list[str]:
    findings_text = " ".join(
        str(item.get("detail", "")) for item in findings if isinstance(item, dict)
    ).lower()

    if "binary" in findings_text or "classification" in findings_text or "outcome" in findings_text:
        return ["Logistic Regression", "Random Forest Classifier"]
    if "time" in dataset_signature or "datetime" in dataset_signature:
        return ["XGBoost", "Prophet"]
    if "categorical" in dataset_signature and "numeric" in dataset_signature:
        return ["Random Forest", "XGBoost"]
    if "numeric" in dataset_signature:
        return ["Linear Regression", "Random Forest Regressor"]
    if "pattern" in findings_text or "segment" in findings_text:
        return ["K-Means", "Isolation Forest"]
    return ["Logistic Regression", "Random Forest"]


def _build_visualizations(findings: list[Any]) -> list[str]:
    findings_text = " ".join(
        str(item.get("detail", "")) for item in findings if isinstance(item, dict)
    ).lower()
    suggestions = ["Missing or zero-coded values by column (horizontal bar)"]

    if "binary" in findings_text or "outcome" in findings_text:
        suggestions.append("Outcome class balance (bar chart)")

    suggestions.extend(
        [
            "Numeric feature distributions split by Outcome",
            "Numeric correlation heatmap",
        ]
    )

    finding_types = {
        str(item.get("type", "")).lower()
        for item in findings
        if isinstance(item, dict)
    }
    if "pattern" in finding_types:
        suggestions.append("Feature averages grouped by Outcome")

    return suggestions[:4]


def _build_approach(dataset_signature: str, retrieved_context: str) -> str:
    normalized_context = retrieved_context.lower()
    if "memory storage unavailable" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {dataset_signature}"
    if "no similar dataset found" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {dataset_signature}"
    if "no highly similar dataset found" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {dataset_signature}"
    return (
        f"Leverage prior memory from similar datasets and begin with a baseline workflow tailored to {dataset_signature}"
    )
