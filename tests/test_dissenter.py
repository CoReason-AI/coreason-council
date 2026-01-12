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
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.types import Persona, ProposerOutput


@pytest.fixture
def mock_proposal() -> ProposerOutput:
    return ProposerOutput(
        proposer_id="test-proposer-1",
        content="This is a test proposal.",
        confidence=0.9,
        metadata={"test": True},
    )


@pytest.fixture
def critic_persona() -> Persona:
    return Persona(
        name="The Skeptic",
        system_prompt="You are a skeptic.",
        capabilities=["critique", "falsification"],
    )


@pytest.mark.asyncio
async def test_mock_dissenter_critique(mock_proposal: ProposerOutput, critic_persona: Persona) -> None:
    """Test that MockDissenter returns a valid Critique object."""
    dissenter = MockDissenter(default_agreement_score=0.3, default_flaws=["Flaw A"])

    critique = await dissenter.critique(mock_proposal, critic_persona)

    assert critique.reviewer_id == critic_persona.name
    assert critique.target_proposer_id == mock_proposal.proposer_id
    assert critique.agreement_score == 0.3
    assert critique.flaws_identified == ["Flaw A"]
    assert "Critique by The Skeptic" in critique.content


@pytest.mark.asyncio
async def test_mock_dissenter_calculate_entropy(mock_proposal: ProposerOutput) -> None:
    """Test that MockDissenter returns the configured entropy score."""
    dissenter = MockDissenter(default_entropy_score=0.8)

    proposals = [mock_proposal, mock_proposal]  # Mocking a list of proposals
    entropy = await dissenter.calculate_entropy(proposals)

    assert entropy == 0.8


@pytest.mark.asyncio
async def test_mock_dissenter_delay(mock_proposal: ProposerOutput, critic_persona: Persona) -> None:
    """Test that MockDissenter respects the delay parameter."""
    dissenter = MockDissenter(delay_seconds=0.01)

    # We just ensure it doesn't crash and returns expected types
    critique = await dissenter.critique(mock_proposal, critic_persona)
    assert critique.agreement_score == 0.5  # Default

    # Pass two proposals to trigger the default entropy return (skipping the <=1 check)
    entropy = await dissenter.calculate_entropy([mock_proposal, mock_proposal])
    assert entropy == 0.1  # Default
