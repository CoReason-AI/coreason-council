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
import os
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import click

from coreason_council.core.aggregator import BaseAggregator, MockAggregator
from coreason_council.core.budget import SimpleBudgetManager
from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.llm_aggregator import LLMAggregator
from coreason_council.core.llm_client import OpenAILLMClient
from coreason_council.core.llm_proposer import LLMProposer
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.proposer import BaseProposer
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.core.types import Persona
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
@click.option("--max-budget", default=100, help="Maximum budget (in operations) before downgrading topology.")
@click.option("--show-trace", is_flag=True, default=False, help="Display the full debate transcript.")
@click.option(
    "--llm", is_flag=True, default=False, help="Use Real LLM (OpenAI) instead of Mock agents. Requires OPENAI_API_KEY."
)
@async_command
async def run_council(
    query: str, max_rounds: int, entropy_threshold: float, max_budget: int, show_trace: bool, llm: bool
) -> None:
    """
    Run a Council session for a given QUERY.
    """
    logger.info(f"Initializing Council for query: '{query}' (Mode: {'LLM' if llm else 'Mock'})")

    # Setup Components based on Mode
    aggregator: BaseAggregator
    proposer_factory: Callable[[Persona], BaseProposer] | None

    if llm:
        if not os.getenv("OPENAI_API_KEY"):
            raise click.ClickException("OPENAI_API_KEY environment variable is required when using --llm.")

        # Shared Client
        llm_client = OpenAILLMClient()

        # Factories
        def _llm_factory(p: Persona) -> BaseProposer:
            return LLMProposer(llm_client)

        proposer_factory = _llm_factory
        aggregator = LLMAggregator(llm_client)
    else:
        proposer_factory = None  # Defaults to Mock inside PanelSelector
        aggregator = MockAggregator()

    # 1. Select Panel
    # Inject the appropriate factory
    panel_selector = PanelSelector(proposer_factory=proposer_factory)
    proposers, personas = panel_selector.select_panel(query)
    click.echo(f"Selected Panel: {[p.name for p in personas]}")

    # 2. Initialize Components
    # Using JaccardDissenter for deterministic entropy
    dissenter = JaccardDissenter()
    # Using SimpleBudgetManager
    budget_manager = SimpleBudgetManager(max_budget=max_budget)

    # 3. Initialize Speaker
    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=dissenter,
        aggregator=aggregator,
        budget_manager=budget_manager,
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
