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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from coreason_council.core.llm_client import GatewayLLMClient, LLMRequest


class MockSchema(BaseModel):
    verdict: str
    confidence: float


@pytest.mark.asyncio
async def test_gateway_client_completion() -> None:
    client = GatewayLLMClient(gateway_url="http://test-gw", access_token="test-token")
    request = LLMRequest(messages=[{"role": "user", "content": "Hello"}])

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "choices": [{"message": {"content": "World"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 10},
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.post.return_value = mock_response

        response = await client.get_completion(request)

        assert response.content == "World"
        assert response.usage["total_tokens"] == 10

        # Verify call
        mock_instance.post.assert_called_once()
        args, kwargs = mock_instance.post.call_args
        assert args[0] == "http://test-gw/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-token"
        assert kwargs["json"]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_gateway_client_structured_output() -> None:
    client = GatewayLLMClient(gateway_url="http://test-gw")
    request = LLMRequest(
        messages=[{"role": "user", "content": "Decide"}], response_schema=MockSchema
    )

    expected_json = {"verdict": "yes", "confidence": 0.9}
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "test-id",
        "choices": [{"message": {"content": json.dumps(expected_json)}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 15},
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.post.return_value = mock_response

        response = await client.get_completion(request)

        assert response.raw_content is not None
        assert isinstance(response.raw_content, MockSchema)
        assert response.raw_content.verdict == "yes"
        assert response.raw_content.confidence == 0.9

        # Verify schema injection
        args, kwargs = mock_instance.post.call_args
        payload = kwargs["json"]
        assert payload["response_format"] == {"type": "json_object"}
        # Check system prompt injection
        messages = payload["messages"]
        assert messages[0]["role"] == "system"
        assert "IMPORTANT: You must respond with valid JSON" in messages[0]["content"]

import httpx

@pytest.mark.asyncio
async def test_gateway_client_http_error() -> None:
    client = GatewayLLMClient()
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance

        mock_instance.post.side_effect = httpx.HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=MagicMock(status_code=400)
        )

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_completion(request)


@pytest.mark.asyncio
async def test_gateway_client_request_error() -> None:
    client = GatewayLLMClient()
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance

        mock_instance.post.side_effect = httpx.RequestError(
            "Connection Refused", request=MagicMock(url="url")
        )

        with pytest.raises(httpx.RequestError):
            await client.get_completion(request)


@pytest.mark.asyncio
async def test_gateway_client_invalid_json_response() -> None:
    client = GatewayLLMClient()
    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}])

    mock_response = MagicMock()
    # Missing 'choices'
    mock_response.json.return_value = {"error": "foo"}
    mock_response.raise_for_status.return_value = None

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.post.return_value = mock_response

        with pytest.raises(ValueError, match="Invalid response format"):
            await client.get_completion(request)


@pytest.mark.asyncio
async def test_gateway_client_structured_output_parsing_failure() -> None:
    client = GatewayLLMClient()
    request = LLMRequest(
        messages=[{"role": "user", "content": "Hi"}], response_schema=MockSchema
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Not JSON"}, "finish_reason": "stop"}]
    }

    with patch("httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        MockClient.return_value.__aenter__.return_value = mock_instance
        mock_instance.post.return_value = mock_response

        with pytest.raises(ValueError, match="LLM failed to return valid JSON"):
            await client.get_completion(request)
