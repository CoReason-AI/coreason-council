# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.budget import SimpleBudgetManager
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.persona import Persona
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker

# Original imports were: Persona


class TestSpeakerBudgetIntegration:
    @pytest.fixture
    def basic_setup(
        self,
    ) -> tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]:
        proposers = [MockProposer("p1"), MockProposer("p2"), MockProposer("p3")]
        personas = [
            Persona(name="P1", system_prompt=""),
            Persona(name="P2", system_prompt=""),
            Persona(name="P3", system_prompt=""),
        ]
        dissenter = MockDissenter()
        aggregator = MockAggregator()
        return proposers, personas, dissenter, aggregator

    @pytest.mark.asyncio
    async def test_speaker_respects_budget_downgrade(
        self, basic_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]
    ) -> None:
        proposers, personas, dissenter, aggregator = basic_setup

        # Budget = 10. Cost for 3 rounds/3 proposers = 21. Cost for 1 round = 3.
        # Should downgrade to 1 round.
        budget_manager = SimpleBudgetManager(max_budget=10)

        speaker = ChamberSpeaker(
            proposers=proposers,
            personas=personas,
            dissenter=dissenter,
            aggregator=aggregator,
            budget_manager=budget_manager,
            max_rounds=3,
        )

        # We need to verify that resolve_query uses 1 round.
        # We can inspect the returned trace to see if it has critiques.
        # Or check logs.
        # Or check calls to dissenter/aggregator.

        # To simulate high entropy (so debate WOULD happen if rounds > 1), set dissenter entropy to 1.0
        dissenter.default_entropy_score = 1.0

        verdict, trace = await speaker.resolve_query("test query")

        # If downgraded to 1 round, loop should break immediately after entropy check because 1 >= 1.
        # So no critiques should be in trace.
        critique_entries = [e for e in trace.transcripts if "critique" in e.action]
        assert len(critique_entries) == 0

        # Verify trace indicates Deadlock (since entropy was high) or Consensus?
        # If rounds=1 and entropy > threshold, it breaks and declares Deadlock.
        # Wait, let's check Speaker logic:
        # Loop starts. Round 1.
        # check entropy. High.
        # check current_round >= current_max_rounds. 1 >= 1 -> True.
        # Declares Deadlock.
        # Aggregator called with is_deadlock=True.

        assert len(verdict.alternatives) > 0  # Should be deadlock verdict

    @pytest.mark.asyncio
    async def test_speaker_no_budget_manager(
        self, basic_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]
    ) -> None:
        proposers, personas, dissenter, aggregator = basic_setup

        # No budget manager
        speaker = ChamberSpeaker(
            proposers=proposers,
            personas=personas,
            dissenter=dissenter,
            aggregator=aggregator,
            budget_manager=None,
            max_rounds=3,
        )

        # High entropy -> should run debate
        dissenter.default_entropy_score = 1.0

        # Mock proposer critique to return fast
        # Note: We cannot simply assign AsyncMock to a method on an instance if mypy is strict.
        # We cast to Any to bypass [method-assign] errors.
        from unittest.mock import Mock

        for p in proposers:
            # We cast to Any to avoid mypy complaining about method assignment
            cast(Any, p).critique_proposal = AsyncMock(
                return_value=Mock(
                    reviewer_id="x", target_proposer_id="y", content="c", flaws_identified=[], agreement_score=0.1
                )
            )
            cast(Any, p).revise_proposal = AsyncMock(
                return_value=Mock(proposer_id="x", content="revised", confidence=0.9)
            )

        # We can't easily mock inner loop calls without extensive patching,
        # but we can rely on trace having critiques.

        # NOTE: running full 3 rounds might be slow if mocks have delays. Ensure MockProposer has no delays.

        verdict, trace = await speaker.resolve_query("test query")

        critique_entries = [e for e in trace.transcripts if "critique" in e.action]
        # Should have critiques for round 1 and round 2.
        # Max rounds 3 means: Round 1 (High) -> Critique -> Round 2 (High) -> Critique -> Round 3 (High) -> Break.
        assert len(critique_entries) > 0
