"""Pipeline orchestration for Insight Copilot."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from agents.insight_agent import run_insight_agent
from agents.memory_agent import run_memory_agent, update_memory_notes
from agents.profiler_agent import run_profiler_agent
from agents.recommender_agent import run_recommender_agent

logger = logging.getLogger(__name__)

try:
    import lyzr  # type: ignore
except ImportError:  # pragma: no cover - optional hackathon integration
    lyzr = None


def run_pipeline(df: pd.DataFrame) -> dict:
    """Run the full multi-agent analysis pipeline on a DataFrame.

    Args:
        df: The uploaded dataset as a pandas DataFrame.

    Returns:
        A combined report dictionary with profiler, insight, memory, and
        recommendation outputs.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    profiler_output = run_profiler_agent(df)
    insight_output, memory_output = asyncio.run(
        _run_parallel_agents(profiler_output=profiler_output)
    )

    try:
        update_memory_notes(
            insight_output.get("summary", ""),
            memory_output.get("memory_point_id"),
        )
    except Exception:
        logger.debug("Memory notes update skipped")

    recommendation_output = run_recommender_agent(
        insight_output=insight_output,
        memory_output=memory_output,
        profiler_output=profiler_output,
    )

    report = {
        "profiler": profiler_output,
        "insight": insight_output,
        "memory": memory_output,
        "recommendation": recommendation_output,
    }

    if lyzr is not None:
        logger.debug("Lyzr package detected; pipeline is compatible with external orchestration wrappers")

    return report


async def _run_parallel_agents(profiler_output: dict[str, Any]) -> tuple[dict, dict]:
    insight_task = asyncio.to_thread(run_insight_agent, profiler_output)
    memory_task = asyncio.to_thread(run_memory_agent, profiler_output)
    return await asyncio.gather(insight_task, memory_task)
