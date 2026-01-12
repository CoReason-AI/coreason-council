# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import asyncio
import functools
import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import click

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.utils.logger import logger

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


def async_command(f: F) -> Callable[..., Any]:
    """Decorator to run a click command in an async loop."""

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(f(*args, **kwargs))

    return wrapper


@click.command()
@click.argument("query")
@click.option("--max-rounds", default=3, help="Maximum number of debate rounds.")
@click.option("--entropy-threshold", default=0.1, help="Entropy threshold for consensus.")
@click.option("--show-trace", is_flag=True, default=False, help="Display the full debate transcript.")
@async_command
async def run_council(query: str, max_rounds: int, entropy_threshold: float, show_trace: bool) -> None:
    """
    Run a Council session for a given QUERY.
    """
    logger.info(f"Initializing Council for query: '{query}'")

    # 1. Select Panel
    panel_selector = PanelSelector()
    proposers, personas = panel_selector.select_panel(query)
    click.echo(f"Selected Panel: {[p.name for p in personas]}")

    # 2. Initialize Components
    # Using JaccardDissenter for deterministic entropy
    dissenter = JaccardDissenter()
    # Using MockAggregator as per current phase requirements
    aggregator = MockAggregator()

    # 3. Initialize Speaker
    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=dissenter,
        aggregator=aggregator,
        entropy_threshold=entropy_threshold,
        max_rounds=max_rounds,
    )

    # 4. Resolve Query
    click.echo("Session started... (Check logs for details)")
    verdict, trace = await speaker.resolve_query(query)

    # 5. Output Results
    click.echo("\n--- FINAL VERDICT ---")
    click.echo(f"Content: {verdict.content}")
    click.echo(f"Confidence: {verdict.confidence_score}")
    click.echo(f"Supporting Evidence: {verdict.supporting_evidence}")
    if verdict.alternatives:
        click.echo("\n--- ALTERNATIVES (Deadlock) ---")
        for alt in verdict.alternatives:
            click.echo(f"Option: {alt.label} - Supported by {len(alt.supporters)} proposers")

    click.echo(f"\nSession ID: {trace.session_id}")

    # 6. Optional Trace Display
    if show_trace:
        click.echo("\n--- DEBATE TRANSCRIPT ---")
        for entry in trace.transcripts:
            # Simple formatting: [Time] Actor (Action): Content
            click.echo(f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.actor} ({entry.action}):")
            click.echo(f"  {entry.content}")
            click.echo("-" * 40)

        click.echo("\n--- VOTE TALLY ---")
        click.echo(json.dumps(trace.vote_tally, indent=2))

    click.echo("--- END ---")


if __name__ == "__main__":  # pragma: no cover
    run_council()
