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

from coreason_council.core.proposer import MockProposer
from coreason_council.core.types import Persona, ProposerOutput


@pytest.fixture
def mock_persona() -> Persona:
    return Persona(
        name="test-persona",
        system_prompt="You are a test assistant.",
        capabilities=["reasoning", "mocking"],
    )


@pytest.mark.asyncio
async def test_mock_proposer_initialization() -> None:
    proposer = MockProposer(
        return_content="Test Content",
        return_confidence=0.85,
        proposer_id_prefix="unit-test",
    )
    assert proposer.return_content == "Test Content"
    assert proposer.return_confidence == 0.85
    assert proposer.proposer_id_prefix == "unit-test"


@pytest.mark.asyncio
async def test_mock_proposer_propose(mock_persona: Persona) -> None:
    proposer = MockProposer(
        return_content="Base Answer",
        return_confidence=0.95,
        proposer_id_prefix="tester",
    )

    query = "What is the meaning of life?"
    output = await proposer.propose(query, mock_persona)

    assert isinstance(output, ProposerOutput)
    assert output.confidence == 0.95
    assert output.proposer_id == "tester-test-persona"
    assert "Base Answer" in output.content
    assert query in output.content
    assert mock_persona.name in output.content
    assert output.metadata["mock"] is True
    assert output.metadata["persona_capabilities"] == ["reasoning", "mocking"]


@pytest.mark.asyncio
async def test_mock_proposer_delay(mock_persona: Persona) -> None:
    import time

    delay = 0.1
    proposer = MockProposer(delay_seconds=delay)

    start_time = time.time()
    await proposer.propose("test", mock_persona)
    end_time = time.time()

    assert (end_time - start_time) >= delay


@pytest.mark.asyncio
async def test_mock_proposer_default_values(mock_persona: Persona) -> None:
    proposer = MockProposer()
    output = await proposer.propose("query", mock_persona)

    assert output.confidence == 0.9
    assert output.proposer_id == "mock-proposer-test-persona"
