"""Unit tests for agents/recommender_agent.py."""

from __future__ import annotations

from agents.recommender_agent import _infer_target_hint, _looks_like_target_name, run_recommender_agent


def _column(name: str, unique_count: int, has_stats: bool = False) -> dict:
    column: dict = {"name": name, "unique_count": unique_count}
    if has_stats:
        column["stats"] = {"mean": 1.0, "std": 1.0, "min": 0.0, "max": 2.0}
    return column


class TestLooksLikeTargetName:
    def test_matches_real_target_keywords(self) -> None:
        for name in ("target", "label", "class", "outcome", "response", "target_label"):
            assert _looks_like_target_name(name), name

    def test_does_not_false_positive_on_names_containing_letter_y(self) -> None:
        # Regression test: a bare "y" keyword used to substring-match any name
        # containing the letter y, silently mislabeling ordinary columns as
        # ML targets.
        for name in ("city", "salary", "quantity", "type", "category", "yield", "year"):
            assert not _looks_like_target_name(name), name

    def test_matches_standalone_y_token(self) -> None:
        assert _looks_like_target_name("y")
        assert _looks_like_target_name("x_y")


class TestInferTargetHint:
    def test_picks_up_low_cardinality_categorical_named_outcome(self) -> None:
        profiler_output = {
            "columns": [
                _column("outcome", unique_count=2),
                _column("city", unique_count=3),
            ]
        }
        hint = _infer_target_hint(profiler_output, findings=[])
        assert hint is not None
        assert hint["name"] == "outcome"

    def test_ignores_low_cardinality_categorical_not_named_like_a_target(self) -> None:
        profiler_output = {
            "columns": [
                _column("city", unique_count=3),
                _column("segment", unique_count=4),
            ]
        }
        hint = _infer_target_hint(profiler_output, findings=[])
        assert hint is None


def test_run_recommender_agent_does_not_pick_city_as_target() -> None:
    insight_output = {"summary": "test", "findings": []}
    memory_output = {"dataset_signature": "4-col-mixed-numeric-categorical", "has_match": False}
    profiler_output = {
        "columns": [
            _column("id", unique_count=10, has_stats=True),
            _column("outcome", unique_count=2, has_stats=True),
            _column("city", unique_count=3),
        ]
    }

    result = run_recommender_agent(
        insight_output=insight_output,
        memory_output=memory_output,
        profiler_output=profiler_output,
    )

    assert "city" not in result["suggested_approach"]
    assert "outcome" in result["suggested_approach"]
