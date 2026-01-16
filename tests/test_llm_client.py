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


class InnerModel(BaseModel):
    id: int
    tag: str


class NestedModel(BaseModel):
    title: str
    items: list[InnerModel]


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


@pytest.mark.asyncio
async def test_mock_llm_client_empty_messages() -> None:
    """Test handling of empty messages list (Edge Case)."""
    client = MockLLMClient()
    request = LLMRequest(messages=[])  # Empty

    response = await client.get_completion(request)
    assert response.content == "Mock LLM Response"


@pytest.mark.asyncio
async def test_mock_llm_client_json_mode_fallback() -> None:
    """
    Test behavior when structured output is requested but return_json is None.
    Simulates LLM failing to follow JSON mode or returning plain text.
    """
    client = MockLLMClient(return_content="I refuse to output JSON.", return_json=None)

    request = LLMRequest(
        messages=[{"role": "user", "content": "JSON please"}],
        response_schema=MockSchema,
    )

    response = await client.get_completion(request)

    # Should fall back to standard content
    assert response.content == "I refuse to output JSON."
    assert response.raw_content is None


@pytest.mark.asyncio
async def test_mock_llm_client_nested_schema() -> None:
    """Test structured output with complex nested models."""
    mock_data = NestedModel(
        title="Complex",
        items=[InnerModel(id=1, tag="A"), InnerModel(id=2, tag="B")],
    )
    client = MockLLMClient(return_json=mock_data)

    request = LLMRequest(
        messages=[{"role": "user", "content": "Complex data"}],
        response_schema=NestedModel,
    )

    response = await client.get_completion(request)

    # Verify JSON structure
    data = json.loads(response.content)
    assert data["title"] == "Complex"
    assert len(data["items"]) == 2
    assert data["items"][0]["tag"] == "A"

    # Verify raw object
    assert response.raw_content == mock_data
    assert response.raw_content.items[1].id == 2


@pytest.mark.asyncio
async def test_mock_llm_client_zero_usage() -> None:
    """Test token usage edge cases (mock logic hardcodes 10/10/20, but we can verify it exists)."""
    # Note: MockLLMClient currently hardcodes usage. To test custom usage, we'd need to update MockLLMClient
    # to accept usage param. For now, we verify the contract holds.
    client = MockLLMClient()
    request = LLMRequest(messages=[])
    response = await client.get_completion(request)

    assert response.usage["total_tokens"] >= 0
    assert "prompt_tokens" in response.usage
