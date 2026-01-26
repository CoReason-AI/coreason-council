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
import json
from typing import Annotated, Callable

import typer
from typer import Argument, Option

from coreason_council.core.aggregator import BaseAggregator, MockAggregator
from coreason_council.core.budget import SimpleBudgetManager
from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.llm_aggregator import LLMAggregator
from coreason_council.core.llm_client import GatewayLLMClient
from coreason_council.core.llm_proposer import LLMProposer
from coreason_council.core.models.persona import Persona
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.proposer import BaseProposer
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.utils.logger import logger

app = typer.Typer()


async def _run_council(
    query: str,
    max_rounds: int,
    entropy_threshold: float,
    max_budget: int,
    show_trace: bool,
    llm: bool,
) -> None:
    logger.info(f"Initializing Council for query: '{query}' (Mode: {'LLM' if llm else 'Mock'})")

    # Setup Components based on Mode
    aggregator: BaseAggregator
    proposer_factory: Callable[[Persona], BaseProposer] | None

    if llm:
        # Shared Client
        llm_client = GatewayLLMClient()

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
    typer.echo(f"Selected Panel: {[p.name for p in personas]}")

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
    typer.echo("Session started... (Check logs for details)")
    verdict, trace = await speaker.resolve_query(query)

    # 5. Output Results
    typer.echo("\n--- FINAL VERDICT ---")
    typer.echo(f"Content: {verdict.content}")
    typer.echo(f"Confidence: {verdict.confidence_score}")
    typer.echo(f"Supporting Evidence: {verdict.supporting_evidence}")
    if verdict.alternatives:
        typer.echo("\n--- ALTERNATIVES (Deadlock) ---")
        for alt in verdict.alternatives:
            typer.echo(f"Option: {alt.label} - Supported by {len(alt.supporters)} proposers")

    typer.echo(f"\nSession ID: {trace.session_id}")

    # 6. Optional Trace Display
    if show_trace:
        typer.echo("\n--- DEBATE TRANSCRIPT ---")
        for entry in trace.transcripts:
            # Simple formatting: [Time] Actor (Action): Content
            typer.echo(f"[{entry.timestamp.strftime('%H:%M:%S')}] {entry.actor} ({entry.action}):")
            typer.echo(f"  {entry.content}")
            typer.echo("-" * 40)

        typer.echo("\n--- VOTE TALLY ---")
        typer.echo(json.dumps(trace.vote_tally, indent=2))

    typer.echo("--- END ---")


@app.command()
def run_council(
    query: Annotated[str, Argument(help="The query to be discussed.")],
    max_rounds: Annotated[int, Option(help="Maximum number of debate rounds.")] = 3,
    entropy_threshold: Annotated[float, Option(help="Entropy threshold for consensus.")] = 0.1,
    max_budget: Annotated[int, Option(help="Maximum budget (in operations) before downgrading topology.")] = 100,
    show_trace: Annotated[bool, Option(help="Display the full debate transcript.")] = False,
    llm: Annotated[bool, Option(help="Use Real LLM (Service Gateway) instead of Mock agents.")] = False,
) -> None:
    """
    Run a Council session for a given QUERY.
    """
    asyncio.run(_run_council(query, max_rounds, entropy_threshold, max_budget, show_trace, llm))


if __name__ == "__main__":  # pragma: no cover
    app()
