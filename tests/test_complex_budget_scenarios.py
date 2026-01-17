# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import pytest

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.budget import SimpleBudgetManager
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.persona import Persona
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker

# Original imports were: Persona


class TestComplexBudgetScenarios:
    @pytest.fixture
    def large_panel_setup(
        self,
    ) -> tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]:
        # Create a "Board of Directors" - 5 members
        names = ["CEO", "CFO", "CTO", "COO", "CMO"]
        proposers = [MockProposer(f"p_{name.lower()}") for name in names]
        personas = [Persona(name=name, system_prompt=f"You are the {name}") for name in names]
        dissenter = MockDissenter()
        aggregator = MockAggregator()
        return proposers, personas, dissenter, aggregator

    @pytest.mark.asyncio
    async def test_expensive_panel_forced_downgrade(
        self, large_panel_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]
    ) -> None:
        """
        Scenario: A large panel (5 members) tries to hold a 3-round debate.
        Cost Analysis:
        - Round 1: 5 ops
        - Round 2: 5^2 = 25 ops
        - Round 3: 5^2 = 25 ops
        Total: 55 ops.

        Budget: 30 ops.
        Outcome: Should downgrade to 1 round (Cost 5).
        """
        proposers, personas, dissenter, aggregator = large_panel_setup
        budget_manager = SimpleBudgetManager(max_budget=30)

        # High entropy to force debate IF allowed
        dissenter.default_entropy_score = 0.9

        speaker = ChamberSpeaker(
            proposers=proposers,
            personas=personas,
            dissenter=dissenter,
            aggregator=aggregator,
            budget_manager=budget_manager,
            max_rounds=3,
        )

        verdict, trace = await speaker.resolve_query("Strategic direction?")

        # Assertions
        # 1. Check that we didn't run critiques (Topology downgraded to STAR/Single Round)
        critiques = [e for e in trace.transcripts if "critique" in e.action]
        assert len(critiques) == 0, "Critiques should not occur if downgraded to 1 round."

        # 2. Check that we still got a result
        assert verdict.content is not None

        # 3. Check logs/trace for evidence? (Optional, relying on behavior mostly)
        # Verify the Vote Tally exists (it's populated at end of resolve_query)
        assert trace.vote_tally is not None

    @pytest.mark.asyncio
    async def test_expensive_panel_budget_ok(
        self, large_panel_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]
    ) -> None:
        """
        Scenario: Same large panel (55 ops cost), but budget is sufficient (60 ops).
        Outcome: Should run full 3 rounds (or until deadlock).
        """
        proposers, personas, dissenter, aggregator = large_panel_setup
        budget_manager = SimpleBudgetManager(max_budget=60)
        dissenter.default_entropy_score = 0.9

        speaker = ChamberSpeaker(
            proposers=proposers,
            personas=personas,
            dissenter=dissenter,
            aggregator=aggregator,
            budget_manager=budget_manager,
            max_rounds=3,
        )

        # Mock revisions to prevent infinite loops if logic depended on it (it doesn't, just max_rounds)
        # But MockProposer is fast.

        verdict, trace = await speaker.resolve_query("Strategic direction?")

        # Should have critiques
        critiques = [e for e in trace.transcripts if "critique" in e.action]
        assert len(critiques) > 0, "Critiques SHOULD occur if budget allows."
