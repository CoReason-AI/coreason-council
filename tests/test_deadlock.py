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
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.core.types import Critique, Persona, ProposerOutput


@pytest.fixture
def mock_personas() -> list[Persona]:
    return [
        Persona(name="Alpha", system_prompt="You are Alpha", capabilities=["logic"]),
        Persona(name="Beta", system_prompt="You are Beta", capabilities=["creativity"]),
    ]


@pytest.fixture
def mock_proposers(mock_personas: list[Persona]) -> list[MockProposer]:
    return [
        MockProposer(proposer_id_prefix="p1", return_content="Alpha says A"),
        MockProposer(proposer_id_prefix="p2", return_content="Beta says B"),
    ]


@pytest.fixture
def mock_dissenter() -> MockDissenter:
    return MockDissenter()


@pytest.fixture
def mock_aggregator() -> MockAggregator:
    return MockAggregator()


@pytest.mark.asyncio
async def test_deadlock_resolution(
    mock_personas: list[Persona],
    mock_proposers: list[MockProposer],
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test Story C: The Deadlock.
    High entropy persists for max_rounds, triggering a Minority Report deadlock verdict.
    """
    # 1. Setup Dissenter to always return High Entropy
    mock_dissenter.default_entropy_score = 0.5  # High entropy

    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
        max_rounds=2,  # Short loop for testing
    )

    verdict, trace = await speaker.resolve_query("What is the meaning of life?")

    # Verify Deadlock Verdict Content
    assert verdict is not None
    assert "MINORITY REPORT" in verdict.content
    assert "Deadlock detected" in verdict.content

    # Verify Strict Deadlock Protocol (Structured Options)
    assert verdict.alternatives, "Deadlock verdict must contain alternatives."
    assert len(verdict.alternatives) == 2

    # Check Option A
    option_a = verdict.alternatives[0]
    assert option_a.label == "Option A"
    assert "p1-Alpha" in option_a.supporters  # Even index 0

    # Check Option B
    option_b = verdict.alternatives[1]
    assert option_b.label == "Option B"
    assert "p2-Beta" in option_b.supporters  # Odd index 1

    # Verify Confidence is low
    assert verdict.confidence_score <= 0.1

    # Trace Analysis:
    # Round 1: Critique + Revise
    # Round 2: Reached Max Round -> Deadlock (No Critique/Revise for Round 2)

    critique_r1_count = sum(1 for i in trace.transcripts if i.action == "critique_round_1")
    revise_r1_count = sum(1 for i in trace.transcripts if i.action == "revise_round_1")
    critique_r2_count = sum(1 for i in trace.transcripts if i.action == "critique_round_2")

    assert critique_r1_count == 2
    assert revise_r1_count == 2
    assert critique_r2_count == 0


@pytest.mark.asyncio
async def test_consensus_after_revision(
    mock_personas: list[Persona],
    mock_proposers: list[MockProposer],
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test scenario where entropy drops after one round of debate.
    """
    mock_dissenter = MockDissenter()
    # High entropy first, then low
    mock_dissenter.calculate_entropy = AsyncMock(side_effect=[0.5, 0.05])  # type: ignore

    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
        max_rounds=5,
    )

    verdict, trace = await speaker.resolve_query("Debate this.")

    assert "MINORITY REPORT" not in verdict.content
    assert not verdict.alternatives  # Should be empty on consensus

    # Verify interactions
    revise_r1_count = sum(1 for i in trace.transcripts if i.action == "revise_round_1")
    assert revise_r1_count == 2

    revise_r2_count = sum(1 for i in trace.transcripts if i.action == "revise_round_2")
    assert revise_r2_count == 0


@pytest.mark.asyncio
async def test_immediate_consensus(
    mock_personas: list[Persona],
    mock_proposers: list[MockProposer],
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test Story A behavior is preserved (Low Entropy -> Immediate Result).
    """
    mock_dissenter.default_entropy_score = 0.0

    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
    )

    verdict, trace = await speaker.resolve_query("Simple question.")

    assert "MINORITY REPORT" not in verdict.content
    assert not verdict.alternatives  # Should be empty on consensus

    # Verify no debate occurred
    critique_count = sum(1 for i in trace.transcripts if "critique" in i.action)
    assert critique_count == 0


@pytest.mark.asyncio
async def test_revise_proposal_failure(
    mock_personas: list[Persona],
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test failure in revise_proposal to ensure exception propagation and cover MockProposer logic.
    """
    # 1. Proposer that fails on revision
    # MockProposer uses the same failure_exception for all methods.

    class RevisionFailingMockProposer(MockProposer):
        async def revise_proposal(
            self, original_proposal: ProposerOutput, critiques: list[Critique], persona: Persona
        ) -> ProposerOutput:
            raise RuntimeError("Revision Failed")

    proposer = RevisionFailingMockProposer(proposer_id_prefix="fail")
    persona = Persona(name="Fail", system_prompt="Fail")

    # We need at least 2 proposers to trigger critique/revise flow (entropy calculation needs > 1 usually,
    # but MockDissenter returns constant regardless of count if > 1)

    proposers = [proposer, MockProposer(proposer_id_prefix="ok")]
    personas = [persona, Persona(name="OK", system_prompt="OK")]

    # Force debate
    dissenter = MockDissenter(default_entropy_score=0.9)

    speaker = ChamberSpeaker(
        proposers=proposers, personas=personas, dissenter=dissenter, aggregator=mock_aggregator, entropy_threshold=0.1
    )

    with pytest.raises(RuntimeError, match="Revision Failed"):
        await speaker.resolve_query("Trigger Revision Failure")


@pytest.mark.asyncio
async def test_mock_proposer_coverage() -> None:
    """
    Directly test MockProposer.revise_proposal with delay and failure to hit coverage lines.
    """
    # Test Delay
    proposer = MockProposer(delay_seconds=0.01)
    output = ProposerOutput(proposer_id="test", content="content", confidence=1.0)
    persona = Persona(name="Test", system_prompt="Test")

    # Revision
    revised = await proposer.revise_proposal(output, [], persona)
    assert "Revised based on 0 critiques" in revised.content

    # Test Failure
    proposer_fail = MockProposer(failure_exception=RuntimeError("Fail"))
    with pytest.raises(RuntimeError, match="Fail"):
        await proposer_fail.revise_proposal(output, [], persona)
