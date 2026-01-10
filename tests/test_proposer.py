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


@pytest.mark.asyncio
async def test_mock_proposer_failure(mock_persona: Persona) -> None:
    """Test that the MockProposer correctly simulates failures."""
    failure_msg = "Simulated API Failure"
    proposer = MockProposer(failure_exception=RuntimeError(failure_msg))

    with pytest.raises(RuntimeError) as exc_info:
        await proposer.propose("query", mock_persona)

    assert str(exc_info.value) == failure_msg


@pytest.mark.asyncio
async def test_mock_proposer_concurrency(mock_persona: Persona) -> None:
    """
    Test concurrency handling by spawning multiple simultaneous requests.
    This ensures the 'async' nature works as expected without blocking.
    """
    delay = 0.1
    count = 10
    proposer = MockProposer(delay_seconds=delay)

    async def task(idx: int) -> ProposerOutput:
        return await proposer.propose(f"query-{idx}", mock_persona)

    import time

    start_time = time.time()

    # Launch 10 tasks concurrently
    tasks = [task(i) for i in range(count)]
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    total_time = end_time - start_time

    # If they ran sequentially, it would take count * delay.
    # If concurrent, it should take roughly delay (plus overhead).
    # We assert it's much faster than sequential.
    assert len(results) == count
    assert total_time < (delay * count * 0.5)
    assert total_time >= delay


@pytest.mark.asyncio
async def test_mock_proposer_edge_inputs(mock_persona: Persona) -> None:
    """Test various edge case inputs for the query."""
    proposer = MockProposer()

    # Empty string
    res_empty = await proposer.propose("", mock_persona)
    assert res_empty.content is not None

    # Unicode/Emoji
    res_emoji = await proposer.propose("Hello 🌍! 🧐", mock_persona)
    assert "Hello 🌍! 🧐" in res_emoji.content

    # Large payload
    large_query = "A" * 10000
    res_large = await proposer.propose(large_query, mock_persona)
    assert len(res_large.content) > 10000
