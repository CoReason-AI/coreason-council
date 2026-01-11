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
from coreason_council.core.proposer import MockProposer
from coreason_council.core.types import Critique, Persona, ProposerOutput


@pytest.fixture
def sample_persona() -> Persona:
    return Persona(name="TestPersona", system_prompt="SysPrompt")


@pytest.fixture
def sample_proposal() -> ProposerOutput:
    return ProposerOutput(proposer_id="p1", content="content", confidence=0.9)


@pytest.mark.asyncio
async def test_mock_proposer(sample_persona: Persona) -> None:
    # Test delay
    proposer = MockProposer(delay_seconds=0.1)
    start = time.time()
    await proposer.propose("q", sample_persona)
    assert time.time() - start >= 0.1

    # Test failure
    error = ValueError("Boom")
    proposer_fail = MockProposer(failure_exception=error)
    with pytest.raises(ValueError, match="Boom"):
        await proposer_fail.propose("q", sample_persona)


@pytest.mark.asyncio
async def test_mock_dissenter(sample_proposal: ProposerOutput, sample_persona: Persona) -> None:
    dissenter = MockDissenter(delay_seconds=0.1)

    # Test critique
    start = time.time()
    critique = await dissenter.critique(sample_proposal, sample_persona)
    assert time.time() - start >= 0.1
    assert isinstance(critique, Critique)

    # Test entropy (<= 1 proposal)
    assert await dissenter.calculate_entropy([sample_proposal]) == 0.0

    # Test entropy (> 1 proposal)
    start = time.time()
    entropy = await dissenter.calculate_entropy([sample_proposal, sample_proposal])
    assert time.time() - start >= 0.1
    assert entropy == dissenter.default_entropy_score


@pytest.mark.asyncio
async def test_mock_aggregator(sample_proposal: ProposerOutput) -> None:
    aggregator = MockAggregator(delay_seconds=0.1)

    start = time.time()
    verdict = await aggregator.aggregate([sample_proposal], [])
    assert time.time() - start >= 0.1
    assert "Mock Verdict" in verdict.content
