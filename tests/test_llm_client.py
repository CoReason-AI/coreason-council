# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import json

import pytest
from pydantic import BaseModel

from coreason_council.core.llm_client import LLMRequest, MockLLMClient


class MockSchema(BaseModel):
    score: float
    reasoning: str


@pytest.mark.asyncio
async def test_mock_llm_client_basic() -> None:
    """Test basic text completion with MockLLMClient."""
    client = MockLLMClient(return_content="Hello World")
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    response = await client.get_completion(request)

    assert response.content == "Hello World"
    assert response.usage["total_tokens"] == 20
    assert response.provider_metadata["mock"] is True


@pytest.mark.asyncio
async def test_mock_llm_client_structured_output_model() -> None:
    """Test structured JSON output simulation with Pydantic Model."""
    mock_data = MockSchema(score=0.9, reasoning="Good")
    client = MockLLMClient(return_json=mock_data)

    request = LLMRequest(
        messages=[{"role": "user", "content": "Rate this"}],
        response_schema=MockSchema,
    )

    response = await client.get_completion(request)

    # Parse content to verify JSON structure
    data = json.loads(response.content)
    assert data["score"] == 0.9
    assert data["reasoning"] == "Good"

    # Verify raw_content is preserved
    assert response.raw_content == mock_data
    assert response.raw_content.score == 0.9


@pytest.mark.asyncio
async def test_mock_llm_client_structured_output_dict() -> None:
    """Test structured JSON output simulation with Dict."""
    mock_data = {"score": 0.5, "reasoning": "Average"}
    client = MockLLMClient(return_json=mock_data)

    request = LLMRequest(
        messages=[{"role": "user", "content": "Rate this"}],
        response_schema={"type": "object"},
    )

    response = await client.get_completion(request)

    # Parse content to verify JSON structure
    data = json.loads(response.content)
    assert data["score"] == 0.5
    assert data["reasoning"] == "Average"

    # Verify raw_content
    assert response.raw_content == mock_data


@pytest.mark.asyncio
async def test_mock_llm_client_failure() -> None:
    """Test failure simulation."""
    client = MockLLMClient(failure_exception=ValueError("API Error"))
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ValueError, match="API Error"):
        await client.get_completion(request)


@pytest.mark.asyncio
async def test_mock_llm_client_delay() -> None:
    """Test delay simulation."""
    # Small delay to ensure it doesn't block but logic holds
    client = MockLLMClient(delay_seconds=0.01)
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    response = await client.get_completion(request)
    assert response.content == "Mock LLM Response"
