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

import pytest
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.models.interaction import ProposerOutput
from coreason_council.core.models.persona import Persona

# Original imports were: Persona, ProposerOutput


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
async def test_entropy_edge_cases(mock_proposal: ProposerOutput) -> None:
    """Test that entropy calculation handles empty and single-item lists correctly."""
    dissenter = MockDissenter(default_entropy_score=0.8)

    # Empty list
    entropy_empty = await dissenter.calculate_entropy([])
    assert entropy_empty == 0.0

    # Single item
    entropy_single = await dissenter.calculate_entropy([mock_proposal])
    assert entropy_single == 0.0

    # Two items (Mocked disagreement)
    entropy_multiple = await dissenter.calculate_entropy([mock_proposal, mock_proposal])
    assert entropy_multiple == 0.8


@pytest.mark.asyncio
async def test_concurrent_critiques(mock_proposal: ProposerOutput, critic_persona: Persona) -> None:
    """
    Test simulating a 'Round Table' where multiple critiques occur simultaneously.
    Verifies that the async mock implementation handles concurrency without blocking.
    """
    dissenter = MockDissenter(delay_seconds=0.1)  # Small delay to force overlap

    # Create 50 concurrent tasks
    tasks = [dissenter.critique(mock_proposal, critic_persona) for _ in range(50)]

    start_time = asyncio.get_running_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_running_loop().time()

    # Verification
    assert len(results) == 50
    for critique in results:
        assert critique.reviewer_id == "The Skeptic"

    # Ensure it didn't run sequentially (50 * 0.1 = 5 seconds)
    # It should run in roughly 0.1s + overhead
    duration = end_time - start_time
    assert duration < 1.0, f"Concurrency test took too long: {duration}s"


@pytest.mark.asyncio
async def test_large_payload_critique(critic_persona: Persona) -> None:
    """Test that the Dissenter handles very large proposal content."""
    dissenter = MockDissenter()

    # Create a 1MB string
    large_content = "x" * 1024 * 1024
    large_proposal = ProposerOutput(proposer_id="big-proposer", content=large_content, confidence=0.5)

    critique = await dissenter.critique(large_proposal, critic_persona)

    assert critique.target_proposer_id == "big-proposer"
    assert "Critique by The Skeptic" in critique.content


@pytest.mark.asyncio
async def test_boundary_values(mock_proposal: ProposerOutput, critic_persona: Persona) -> None:
    """Test strict boundary values for agreement and entropy."""

    # 1. Zero Agreement / Zero Entropy
    dissenter_low = MockDissenter(default_agreement_score=0.0, default_entropy_score=0.0)
    critique_low = await dissenter_low.critique(mock_proposal, critic_persona)
    entropy_low = await dissenter_low.calculate_entropy([mock_proposal, mock_proposal])

    assert critique_low.agreement_score == 0.0
    assert entropy_low == 0.0

    # 2. Max Agreement / Max Entropy
    dissenter_high = MockDissenter(default_agreement_score=1.0, default_entropy_score=1.0)
    critique_high = await dissenter_high.critique(mock_proposal, critic_persona)
    entropy_high = await dissenter_high.calculate_entropy([mock_proposal, mock_proposal])

    assert critique_high.agreement_score == 1.0
    assert entropy_high == 1.0
