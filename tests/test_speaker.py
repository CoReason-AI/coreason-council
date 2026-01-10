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
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.core.types import CouncilTrace, Persona, Verdict


@pytest.fixture
def mock_proposers() -> list[MockProposer]:
    return [
        MockProposer(return_content="Content A", proposer_id_prefix="p1"),
        MockProposer(return_content="Content B", proposer_id_prefix="p2"),
    ]


@pytest.fixture
def mock_personas() -> list[Persona]:
    return [
        Persona(name="Persona A", system_prompt="You are A", capabilities=["logic"]),
        Persona(name="Persona B", system_prompt="You are B", capabilities=["creative"]),
    ]


@pytest.fixture
def mock_dissenter_low_entropy() -> MockDissenter:
    # Returns 0.0 entropy by default (Story A)
    return MockDissenter(default_entropy_score=0.0)


@pytest.fixture
def mock_dissenter_high_entropy() -> MockDissenter:
    # Returns 0.9 entropy (Story B trigger)
    return MockDissenter(default_entropy_score=0.9)


@pytest.fixture
def mock_aggregator() -> MockAggregator:
    return MockAggregator(default_content="Final Answer")


@pytest.mark.asyncio
async def test_chamber_speaker_init_validation(
    mock_proposers: list[MockProposer],
    mock_personas: list[Persona],
    mock_dissenter_low_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """Test all validation checks in __init__."""

    # 1. Valid Init
    speaker = ChamberSpeaker(mock_proposers, mock_personas, mock_dissenter_low_entropy, mock_aggregator)
    assert speaker

    # 2. Empty Proposers
    with pytest.raises(ValueError, match="requires at least one Proposer"):
        ChamberSpeaker(
            proposers=[], personas=mock_personas, dissenter=mock_dissenter_low_entropy, aggregator=mock_aggregator
        )

    # 3. Empty Personas
    with pytest.raises(ValueError, match="requires at least one Persona"):
        ChamberSpeaker(
            proposers=mock_proposers, personas=[], dissenter=mock_dissenter_low_entropy, aggregator=mock_aggregator
        )

    # 4. Mismatch Length
    with pytest.raises(ValueError, match="Count mismatch"):
        ChamberSpeaker(
            proposers=mock_proposers,
            personas=[mock_personas[0]],  # 2 vs 1
            dissenter=mock_dissenter_low_entropy,
            aggregator=mock_aggregator,
        )

    # 5. Missing Dissenter
    with pytest.raises(ValueError, match="requires a Dissenter"):
        ChamberSpeaker(
            proposers=mock_proposers,
            personas=mock_personas,
            dissenter=None,  # type: ignore
            aggregator=mock_aggregator,
        )

    # 6. Missing Aggregator
    with pytest.raises(ValueError, match="requires an Aggregator"):
        ChamberSpeaker(
            proposers=mock_proposers,
            personas=mock_personas,
            dissenter=mock_dissenter_low_entropy,
            aggregator=None,  # type: ignore
        )


@pytest.mark.asyncio
async def test_resolve_query_low_entropy_flow(
    mock_proposers: list[MockProposer],
    mock_personas: list[Persona],
    mock_dissenter_low_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test Story A: Low Cost / Low Entropy Flow.
    Should gather proposals, check entropy, and aggregate immediately.
    """
    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter_low_entropy,
        aggregator=mock_aggregator,
        entropy_threshold=0.1,
    )

    query = "Is this a simple question?"
    verdict, trace = await speaker.resolve_query(query)

    # 1. Verification of Return Types
    assert isinstance(verdict, Verdict)
    assert isinstance(trace, CouncilTrace)

    # 2. Verify Trace Content
    assert trace.session_id is not None
    assert trace.roster == ["Persona A", "Persona B"]
    assert trace.entropy_score == 0.0
    assert trace.final_verdict == verdict

    # Verify transcripts (Propose A, Propose B, Verdict)
    actions = [entry.action for entry in trace.transcripts]
    assert actions.count("propose") == 2
    assert "verdict" in actions

    # 3. Verify Aggregator Logic
    # The MockAggregator includes input IDs in the content
    assert "Final Answer" in verdict.content
    # Check that it referenced the inputs (p1-Persona A, p2-Persona B)
    # The MockProposer constructs ID as "{prefix}-{persona.name}"
    assert "p1-Persona A" in verdict.content
    assert "p2-Persona B" in verdict.content


@pytest.mark.asyncio
async def test_resolve_query_high_entropy_boundary(
    mock_proposers: list[MockProposer],
    mock_personas: list[Persona],
    mock_dissenter_high_entropy: MockDissenter,
    mock_aggregator: MockAggregator,
) -> None:
    """
    Test Boundary: High Entropy should trigger NotImplementedError (for this atomic unit).
    """
    speaker = ChamberSpeaker(
        proposers=mock_proposers,
        personas=mock_personas,
        dissenter=mock_dissenter_high_entropy,
        aggregator=mock_aggregator,
        entropy_threshold=0.5,
    )

    query = "Is this a complex debated question?"

    # The mock returns 0.9, which is > 0.5
    with pytest.raises(NotImplementedError, match="Debate loop"):
        await speaker.resolve_query(query)
