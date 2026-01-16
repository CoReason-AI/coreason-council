# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import time

import pytest

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.persona import Persona
from coreason_council.core.models.verdict import Verdict, VerdictOption
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker

# Original imports were: Critique, Persona, ProposerOutput, Verdict


@pytest.fixture
def mock_personas_3() -> list[Persona]:
    return [
        Persona(name="A", system_prompt="A"),
        Persona(name="B", system_prompt="B"),
        Persona(name="C", system_prompt="C"),
    ]


@pytest.fixture
def mock_dissenter() -> MockDissenter:
    return MockDissenter(default_entropy_score=0.0)


@pytest.fixture
def mock_aggregator() -> MockAggregator:
    return MockAggregator()


@pytest.mark.asyncio
async def test_speaker_concurrency(
    mock_personas_3: list[Persona], mock_dissenter: MockDissenter, mock_aggregator: MockAggregator
) -> None:
    """
    Verify that Proposers are executed concurrently.
    We set up 3 proposers, each taking 0.1s.
    Total time should be around 0.1s, significantly less than 0.3s.
    """
    delay = 0.1
    proposers = [
        MockProposer(delay_seconds=delay, proposer_id_prefix="p1"),
        MockProposer(delay_seconds=delay, proposer_id_prefix="p2"),
        MockProposer(delay_seconds=delay, proposer_id_prefix="p3"),
    ]

    speaker = ChamberSpeaker(proposers, mock_personas_3, mock_dissenter, mock_aggregator)

    start_time = time.time()
    await speaker.resolve_query("test query")
    end_time = time.time()

    duration = end_time - start_time

    # Allow some buffer for overhead, but it should definitely be faster than sequential
    # Sequential would be >= 0.3s. Concurrent should be ~0.1s + overhead.
    assert duration < (delay * 3)
    assert duration >= delay


@pytest.mark.asyncio
async def test_proposer_partial_failure(
    mock_personas_3: list[Persona], mock_dissenter: MockDissenter, mock_aggregator: MockAggregator
) -> None:
    """
    Verify that if one Proposer fails, the Speaker raises the exception (Fail Fast).
    """
    proposers = [
        MockProposer(return_content="OK", proposer_id_prefix="p1"),
        MockProposer(failure_exception=ValueError("Agent Failure"), proposer_id_prefix="p2"),
        MockProposer(return_content="OK", proposer_id_prefix="p3"),
    ]

    speaker = ChamberSpeaker(proposers, mock_personas_3, mock_dissenter, mock_aggregator)

    with pytest.raises(ValueError, match="Agent Failure"):
        await speaker.resolve_query("test query")


@pytest.mark.asyncio
async def test_entropy_boundary_inclusive(mock_personas_3: list[Persona], mock_aggregator: MockAggregator) -> None:
    """
    Verify that if entropy EQUALS the threshold, we still aggregate (inclusive check).
    """
    # Setup dissenter to return exactly 0.5
    dissenter = MockDissenter(default_entropy_score=0.5)

    proposers = [
        MockProposer(return_content="OK"),
        MockProposer(return_content="OK"),
        MockProposer(return_content="OK"),
    ]

    # Set threshold to 0.5
    speaker = ChamberSpeaker(proposers, mock_personas_3, dissenter, mock_aggregator, entropy_threshold=0.5)

    verdict, trace = await speaker.resolve_query("test query")

    assert isinstance(verdict, Verdict)
    assert trace.entropy_score == 0.5
    assert trace.final_verdict is not None


@pytest.mark.asyncio
async def test_aggregator_failure(mock_personas_3: list[Persona], mock_dissenter: MockDissenter) -> None:
    """
    Verify that if Aggregator fails, the exception propagates.
    """

    class FailingAggregator(MockAggregator):
        async def aggregate(
            self, proposals: list[ProposerOutput], critiques: list[Critique], is_deadlock: bool = False
        ) -> Verdict:
            raise RuntimeError("Aggregator Crashed")

    proposers = [
        MockProposer(return_content="OK"),
        MockProposer(return_content="OK"),
        MockProposer(return_content="OK"),
    ]

    speaker = ChamberSpeaker(proposers, mock_personas_3, mock_dissenter, FailingAggregator())

    with pytest.raises(RuntimeError, match="Aggregator Crashed"):
        await speaker.resolve_query("test query")


@pytest.mark.asyncio
async def test_vote_tally_deadlock_edge_cases(mock_personas_3: list[Persona], mock_dissenter: MockDissenter) -> None:
    """
    Test edge cases for vote_tally population in Deadlock scenarios.
    1. Empty Supporters list.
    2. Duplicate Supporters (should be de-duplicated).
    """

    # Custom Aggregator that produces messy deadlock data
    class MessyDeadlockAggregator(MockAggregator):
        async def aggregate(
            self, proposals: list[ProposerOutput], critiques: list[Critique], is_deadlock: bool = False
        ) -> Verdict:
            return Verdict(
                content="Deadlock",
                confidence_score=0.1,
                alternatives=[
                    # Case 1: No supporters
                    VerdictOption(label="Option Empty", content="No support", supporters=[]),
                    # Case 2: Duplicate supporters
                    VerdictOption(label="Option Dupes", content="Dupe support", supporters=["p1", "p1", "p2"]),
                ],
            )

    proposers = [
        MockProposer(return_content="A"),
        MockProposer(return_content="B"),
        MockProposer(return_content="C"),
    ]

    # Ensure deadlock triggers
    mock_dissenter.default_entropy_score = 0.9

    speaker = ChamberSpeaker(
        proposers, mock_personas_3, mock_dissenter, MessyDeadlockAggregator(), entropy_threshold=0.1, max_rounds=1
    )

    verdict, trace = await speaker.resolve_query("test query")

    assert trace.vote_tally is not None
    # Verify count is 0 for empty list
    assert trace.vote_tally["Option Empty"] == 0
    # Verify count is 2 for ['p1', 'p1', 'p2'] (p1 deduped)
    assert trace.vote_tally["Option Dupes"] == 2
