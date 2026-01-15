# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from unittest.mock import AsyncMock

import pytest
from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.persona import Persona
from coreason_council.core.models.trace import TopologyType
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker


@pytest.fixture
def complex_personas() -> list[Persona]:
    return [
        Persona(name="A", system_prompt="A"),
        Persona(name="B", system_prompt="B"),
        Persona(name="C", system_prompt="C"),
    ]


@pytest.fixture
def mock_proposers() -> list[MockProposer]:
    return [
        MockProposer(proposer_id_prefix="p1"),
        MockProposer(proposer_id_prefix="p2"),
        MockProposer(proposer_id_prefix="p3"),
    ]


@pytest.mark.asyncio
async def test_complex_high_entropy_loop(complex_personas: list[Persona], mock_proposers: list[MockProposer]) -> None:
    """
    Complex Scenario: "The Persistent Disagreement"

    Setup:
    - 3 Personas.
    - Dissenter returns High Entropy for 2 rounds, then Low Entropy.
    - Max Rounds = 5.

    Goal:
    - Verify the loop runs exactly 3 times (Round 1, Round 2, Round 3->Consensus).
    - Verify the trace logs correct topology transitions.
    - Verify 'revise_round_X' and 'critique_round_X' entries exist for rounds 1 and 2.
    """
    mock_aggregator = MockAggregator()
    mock_dissenter = MockDissenter()

    # Side Effect:
    # Round 1 Check: 0.8 (High)
    # Round 2 Check: 0.6 (High)
    # Round 3 Check: 0.05 (Low -> Consensus)
    setattr(mock_dissenter, "calculate_entropy", AsyncMock(side_effect=[0.8, 0.6, 0.05]))

    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=complex_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
        max_rounds=5,
    )

    verdict, trace = await speaker.resolve_query("Difficult Query")

    # 1. Verify Loop Iterations
    # Entropy call count = 3
    assert mock_dissenter.calculate_entropy.call_count == 3

    # 2. Verify Final State
    assert trace.entropy_score == 0.05
    assert trace.final_verdict is not None
    assert not trace.final_verdict.alternatives  # Consensus reached, no alternatives

    # 3. Verify Trace Transcript Structure
    actions = [t.action for t in trace.transcripts]

    # Initial Proposals
    assert actions.count("propose") == 3

    # Round 1 Critiques & Revisions (3 proposers * 2 peers = 6 critiques)
    assert actions.count("critique_round_1") == 6
    assert actions.count("revise_round_1") == 3

    # Round 2 Critiques & Revisions
    assert actions.count("critique_round_2") == 6
    assert actions.count("revise_round_2") == 3

    # Round 3: Entropy check passed, so NO critique/revision for Round 3
    assert actions.count("critique_round_3") == 0
    assert actions.count("revise_round_3") == 0

    # Final Verdict
    assert "verdict" in actions

    # 4. Verify Topology Log
    # Should end in ROUND_TABLE because it entered the loop
    assert trace.topology == TopologyType.ROUND_TABLE


@pytest.mark.asyncio
async def test_max_rounds_boundary_deadlock(
    complex_personas: list[Persona], mock_proposers: list[MockProposer]
) -> None:
    """
    Complex Scenario: "Immediate Deadlock"

    Setup:
    - Max Rounds = 1.
    - Dissenter returns High Entropy immediately.

    Goal:
    - Verify it enters the loop, checks entropy, fails threshold.
    - Checks max_rounds condition (1 >= 1).
    - Breaks loop and declares deadlock WITHOUT running critiques.
    """
    mock_aggregator = MockAggregator()
    mock_dissenter = MockDissenter()
    setattr(mock_dissenter, "calculate_entropy", AsyncMock(return_value=0.9))

    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=complex_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
        max_rounds=1,
    )

    verdict, trace = await speaker.resolve_query("Impossible Query")

    # 1. Check Deadlock
    # MockAggregator creates alternatives if is_deadlock=True
    assert len(verdict.alternatives) > 0
    assert verdict.confidence_score < 0.5

    # 2. Verify NO Critiques were generated
    # Since Round 1 >= Max Rounds 1, it should break BEFORE critique phase.
    actions = [t.action for t in trace.transcripts]
    assert "critique_round_1" not in actions
    assert "revise_round_1" not in actions
