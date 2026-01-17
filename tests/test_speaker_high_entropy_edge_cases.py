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
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.persona import Persona
from coreason_council.core.models.trace import TopologyType
from coreason_council.core.models.verdict import Verdict
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker

# Original imports were: Persona, TopologyType, Verdict


@pytest.fixture
def mock_dissenter_high_entropy() -> MockDissenter:
    return MockDissenter(default_entropy_score=0.9)


@pytest.fixture
def mock_aggregator() -> MockAggregator:
    return MockAggregator(default_content="Final Answer")


@pytest.mark.asyncio
async def test_resolve_query_high_entropy_single_proposer(
    mock_dissenter_high_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test edge case: Single proposer.
    Even if the Dissenter is configured to return high entropy (0.9),
    the `calculate_entropy` logic must return 0.0 for a single proposal.
    Therefore, the system should follow the Low Entropy (STAR) path.
    """
    proposer = MockProposer(return_content="Lone Wolf", proposer_id_prefix="p1")
    persona = Persona(name="LonePersona", system_prompt="You are alone")

    speaker = ChamberSpeaker(
        proposers=[proposer],
        personas=[persona],
        dissenter=mock_dissenter_high_entropy,
        aggregator=mock_aggregator,
        entropy_threshold=0.5,
    )

    verdict, trace = await speaker.resolve_query("Am I alone?")

    # Verify Entropy Logic override
    assert trace.entropy_score == 0.0

    # Verify Topology remains STAR (Low Entropy path)
    assert trace.topology == TopologyType.STAR

    # Verify NO critiques generated
    # Check for any action starting with "critique"
    critique_actions = [e for e in trace.transcripts if "critique" in e.action]
    assert len(critique_actions) == 0

    # Verify Verdict exists
    assert isinstance(verdict, Verdict)

    # Check that the MockAggregator referenced the Proposer ID
    # ID format: {prefix}-{persona.name} -> p1-LonePersona
    assert "p1-LonePersona" in verdict.content


@pytest.mark.asyncio
async def test_resolve_query_critique_failure(
    mock_dissenter_high_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test edge case: One proposer fails during critique generation.
    The session should raise the exception (fail-fast behavior).
    """
    # Proposer 1 works fine
    p1 = MockProposer(return_content="P1 Content", proposer_id_prefix="p1")
    # Proposer 2 will fail when asked to critique
    error_msg = "Critique Logic Failed"
    p2 = MockProposer(return_content="P2 Content", proposer_id_prefix="p2", failure_exception=RuntimeError(error_msg))

    personas = [
        Persona(name="P1", system_prompt="Sys1"),
        Persona(name="P2", system_prompt="Sys2"),
    ]

    speaker = ChamberSpeaker(
        proposers=[p1, p2],
        personas=personas,
        dissenter=mock_dissenter_high_entropy,
        aggregator=mock_aggregator,
        entropy_threshold=0.5,
    )

    # p1 will try to critique p2 (OK)
    # p2 will try to critique p1 (FAIL)
    # asyncio.gather should raise the exception
    with pytest.raises(RuntimeError, match=error_msg):
        await speaker.resolve_query("Trigger Failure")


@pytest.mark.asyncio
async def test_resolve_query_many_proposers_critique_count(
    mock_dissenter_high_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test edge case: Combinatorial explosion of critiques.
    With 4 proposers, we expect 4 * 3 = 12 critiques.
    """
    count = 4
    proposers = [MockProposer(return_content=f"Content {i}", proposer_id_prefix=f"p{i}") for i in range(count)]
    personas = [Persona(name=f"Persona {i}", system_prompt=f"Sys {i}") for i in range(count)]

    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=mock_dissenter_high_entropy,
        aggregator=mock_aggregator,
        entropy_threshold=0.5,
    )

    _, trace = await speaker.resolve_query("Big meeting")

    critique_actions = [e for e in trace.transcripts if e.action == "critique_round_1"]
    expected_critiques = count * (count - 1)

    assert len(critique_actions) == expected_critiques
    assert trace.topology == TopologyType.ROUND_TABLE
