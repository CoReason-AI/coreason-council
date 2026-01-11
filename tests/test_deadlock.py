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
async def test_deadlock_odd_number_of_proposers(
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Edge Case 1: Deadlock with an odd number of proposers (3).
    Verifies that the mock aggregator correctly splits the group.
    """
    mock_dissenter.default_entropy_score = 0.9
    personas = [
        Persona(name="A", system_prompt="A"),
        Persona(name="B", system_prompt="B"),
        Persona(name="C", system_prompt="C"),
    ]
    proposers = [MockProposer(proposer_id_prefix=f"p{i}") for i in range(3)]

    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        max_rounds=1,
    )

    verdict, _ = await speaker.resolve_query("Query")

    assert verdict.alternatives
    assert len(verdict.alternatives) == 2
    # Even indices: 0, 2 (2 supporters)
    assert len(verdict.alternatives[0].supporters) == 2
    assert "p0-A" in verdict.alternatives[0].supporters
    assert "p2-C" in verdict.alternatives[0].supporters
    # Odd indices: 1 (1 supporter)
    assert len(verdict.alternatives[1].supporters) == 1
    assert "p1-B" in verdict.alternatives[1].supporters


@pytest.mark.asyncio
async def test_deadlock_verdict_serialization(
    mock_personas: list[Persona],
    mock_proposers: list[MockProposer],
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Edge Case 2: Serialization of Verdict with Alternatives.
    Ensures that CouncilTrace can dump the structured deadlock verdict to JSON/dict.
    """
    mock_dissenter.default_entropy_score = 0.9
    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        max_rounds=1,
    )

    _, trace = await speaker.resolve_query("Query")

    # Attempt to dump to dict (Pydantic v2)
    trace_dict = trace.model_dump()
    assert trace_dict["final_verdict"] is not None
    assert trace_dict["final_verdict"]["alternatives"] is not None
    assert len(trace_dict["final_verdict"]["alternatives"]) == 2
    assert trace_dict["final_verdict"]["alternatives"][0]["label"] == "Option A"


@pytest.mark.asyncio
async def test_complex_long_running_deadlock(
    mock_dissenter: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Complex Scenario: A long-running debate (5 rounds) with many proposers (4) that fails to reach consensus.
    """
    mock_dissenter.default_entropy_score = 0.9  # Persistent high entropy
    personas = [Persona(name=f"P{i}", system_prompt="...") for i in range(4)]
    proposers = [MockProposer(proposer_id_prefix=f"m{i}") for i in range(4)]

    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=mock_dissenter,
        aggregator=mock_aggregator,
        max_rounds=5,
    )

    verdict, trace = await speaker.resolve_query("Hard Query")

    # Verify Deadlock
    assert "MINORITY REPORT" in verdict.content
    assert len(verdict.alternatives) == 2
    assert len(verdict.alternatives[0].supporters) == 2  # 4 split into 2 vs 2

    # Verify rounds
    # Should have 4 critiques (Round 1 to 4) and 4 revisions (Round 1 to 4)
    # Round 5 starts, sees high entropy, checks max_rounds (>=5), and breaks.
    # So max_rounds=5 means we stop AT round 5 start, meaning 4 previous rounds completed?
    # Logic:
    # Round 1 start.
    # Check entropy. High.
    # Check current_round (1) >= max (5)? False.
    # Critique Round 1. Revise Round 1.
    # Round 2 start...
    # ...
    # Round 4 Critique/Revise.
    # Round 5 start.
    # Check entropy. High.
    # Check current_round (5) >= max (5)? True -> Break.
    # So we expect 4 sets of critique/revise actions.

    revise_r4_count = sum(1 for i in trace.transcripts if i.action == "revise_round_4")
    assert revise_r4_count == 4  # 4 proposers
    revise_r5_count = sum(1 for i in trace.transcripts if i.action == "revise_round_5")
    assert revise_r5_count == 0


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
