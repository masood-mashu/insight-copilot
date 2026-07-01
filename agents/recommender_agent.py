"""Recommendation synthesis for next-step analysis."""

from __future__ import annotations

from typing import Any


def run_recommender_agent(
    insight_output: dict,
    memory_output: dict,
    profiler_output: dict | None = None,
) -> dict:
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
    if profiler_output is not None and not isinstance(profiler_output, dict):
        raise TypeError("profiler_output must be a dict when provided")

    findings = insight_output.get("findings", [])
    dataset_signature = str(memory_output.get("dataset_signature", "unknown-dataset"))
    retrieved_context = str(memory_output.get("retrieved_context", ""))
    target_hint = _infer_target_hint(profiler_output, findings)

    cleaning_steps = _build_cleaning_steps(findings, target_hint)
    suggested_models = _build_models(dataset_signature, findings, target_hint)
    visualization_suggestions = _build_visualizations(findings, target_hint)
    suggested_approach = _build_approach(dataset_signature, retrieved_context, target_hint)

    return {
        "suggested_approach": suggested_approach,
        "suggested_models": suggested_models,
        "cleaning_steps": cleaning_steps,
        "visualization_suggestions": visualization_suggestions,
    }


def _build_cleaning_steps(findings: list[Any], target_hint: dict[str, Any] | None) -> list[str]:
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
    if target_hint and target_hint.get("kind") == "categorical_target":
        steps.append(
            f"Validate class balance and label consistency in '{target_hint['name']}' before training"
        )
    return steps[:4]


def _build_models(
    dataset_signature: str,
    findings: list[Any],
    target_hint: dict[str, Any] | None,
) -> list[str]:
    findings_text = " ".join(
        str(item.get("detail", "")) for item in findings if isinstance(item, dict)
    ).lower()

    if target_hint and target_hint.get("kind") == "categorical_target":
        if int(target_hint.get("unique_count", 0)) == 2:
            return ["Logistic Regression", "Random Forest Classifier"]
        return ["Random Forest Classifier", "XGBoost"]
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


def _build_visualizations(
    findings: list[Any],
    target_hint: dict[str, Any] | None,
) -> list[str]:
    findings_text = " ".join(
        str(item.get("detail", "")) for item in findings if isinstance(item, dict)
    ).lower()
    suggestions = []

    if any(str(item.get("type", "")).lower() == "anomaly" for item in findings if isinstance(item, dict)):
        suggestions.append("Missing or zero-coded values by column (horizontal bar)")
    else:
        suggestions.append("Numeric feature distributions (histogram grid)")

    if target_hint and target_hint.get("kind") == "categorical_target":
        suggestions.append(f"{target_hint['name']} class balance (bar chart)")
        suggestions.append(f"Numeric feature distributions split by {target_hint['name']}")
    elif "binary" in findings_text or "outcome" in findings_text:
        suggestions.append("Outcome class balance (bar chart)")
        suggestions.append("Numeric feature distributions split by Outcome")
    else:
        suggestions.append("Numeric pairplot or grouped scatter matrix")

    suggestions.extend(
        [
            "Numeric correlation heatmap",
        ]
    )

    finding_types = {
        str(item.get("type", "")).lower()
        for item in findings
        if isinstance(item, dict)
    }
    if "pattern" in finding_types:
        if target_hint and target_hint.get("kind") == "categorical_target":
            suggestions.append(f"Feature averages grouped by {target_hint['name']}")
        else:
            suggestions.append("Feature averages grouped by the most relevant category")

    return suggestions[:4]


def _build_approach(
    dataset_signature: str,
    retrieved_context: str,
    target_hint: dict[str, Any] | None,
) -> str:
    task_hint = dataset_signature
    if target_hint and target_hint.get("kind") == "categorical_target":
        task_hint = f"{target_hint['name']} classification on {dataset_signature}"
    normalized_context = retrieved_context.lower()
    if "memory storage unavailable" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {task_hint}"
    if "no similar dataset found" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {task_hint}"
    if "no highly similar dataset found" in normalized_context:
        return f"Start with general-purpose exploratory analysis and baseline modeling for {task_hint}"
    return (
        f"Leverage prior memory from similar datasets and begin with a baseline workflow tailored to {task_hint}"
    )


def _infer_target_hint(
    profiler_output: dict | None,
    findings: list[Any],
) -> dict[str, Any] | None:
    if not profiler_output:
        return _infer_target_hint_from_findings(findings)

    columns = profiler_output.get("columns", [])
    categorical_candidates = [
        column
        for column in columns
        if isinstance(column, dict)
        and "stats" not in column
        and 1 < int(column.get("unique_count", 0)) <= 10
    ]
    if categorical_candidates:
        candidate = categorical_candidates[0]
        return {
            "kind": "categorical_target",
            "name": str(candidate.get("name", "label")),
            "unique_count": int(candidate.get("unique_count", 0)),
        }

    low_card_numeric = [
        column
        for column in columns
        if isinstance(column, dict)
        and "stats" in column
        and 1 < int(column.get("unique_count", 0)) <= 5
        and str(column.get("name", "")).lower() not in {"id", "index"}
    ]
    if low_card_numeric:
        candidate = low_card_numeric[0]
        return {
            "kind": "categorical_target",
            "name": str(candidate.get("name", "label")),
            "unique_count": int(candidate.get("unique_count", 0)),
        }

    return _infer_target_hint_from_findings(findings)


def _infer_target_hint_from_findings(findings: list[Any]) -> dict[str, Any] | None:
    findings_text = " ".join(
        str(item.get("detail", "")) for item in findings if isinstance(item, dict)
    )
    if "outcome" in findings_text.lower():
        return {"kind": "categorical_target", "name": "Outcome", "unique_count": 2}
    return None
