# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel

from coreason_council.core.llm_client import LLMRequest, LLMResponse, OpenAILLMClient


class MockStructuredOutput(BaseModel):
    reasoning: str
    verdict: str


class NestedSchema(BaseModel):
    sub_field: str


class ComplexSchema(BaseModel):
    id: int
    details: NestedSchema
    tags: list[str]


@pytest.fixture
def openai_client() -> Any:
    # Ensure no real API key is needed for instantiation if mocked, but we pass one to avoid validation errors if any
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
        return OpenAILLMClient()


@pytest.mark.asyncio
async def test_openai_client_standard_completion(openai_client: OpenAILLMClient) -> None:
    # Mock the underlying client.chat.completions.create
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-123"
    mock_response.choices = [MagicMock(message=MagicMock(content="Mocked response content"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_create.return_value = mock_response

    # Cast to Any to allow method assignment for mocking.
    # Because 'instructor' wraps the client, we mock the method on openai_client.client.chat.completions
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(messages=[{"role": "user", "content": "Hello"}], temperature=0.5)

    response = await openai_client.get_completion(request)

    assert isinstance(response, LLMResponse)
    assert response.content == "Mocked response content"
    assert response.usage["total_tokens"] == 15
    assert response.provider_metadata["id"] == "chatcmpl-123"

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.5
    # messages might be cast to ChatCompletionMessageParam, so we check content
    assert call_kwargs["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_openai_client_system_prompt_handling(openai_client: OpenAILLMClient) -> None:
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ok"), finish_reason="stop")]
    mock_create.return_value = mock_response
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(messages=[{"role": "user", "content": "Hi"}], system_prompt="You are a helpful assistant.")

    await openai_client.get_completion(request)

    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["content"] == "Hi"


@pytest.mark.asyncio
async def test_openai_client_structured_output(openai_client: OpenAILLMClient) -> None:
    # Mock create_with_completion (Instructor's method)
    mock_create_wc = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-struct"

    mock_parsed_obj = MockStructuredOutput(reasoning="Because X", verdict="True")
    mock_response.choices = [MagicMock(message=MagicMock(), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10, total_tokens=30)

    # create_with_completion returns (model, completion)
    mock_create_wc.return_value = (mock_parsed_obj, mock_response)

    cast(Any, openai_client.client).chat.completions.create_with_completion = mock_create_wc

    request = LLMRequest(messages=[{"role": "user", "content": "Analyze this"}], response_schema=MockStructuredOutput)

    response = await openai_client.get_completion(request)

    assert response.raw_content == mock_parsed_obj
    # Content should be the serialized JSON of the Pydantic object
    assert "Because X" in response.content
    assert "True" in response.content

    mock_create_wc.assert_called_once()
    call_kwargs = mock_create_wc.call_args.kwargs
    assert call_kwargs["response_model"] == MockStructuredOutput


@pytest.mark.asyncio
async def test_openai_client_error_handling(openai_client: OpenAILLMClient) -> None:
    mock_create = AsyncMock()
    mock_create.side_effect = Exception("API Error")
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(messages=[{"role": "user", "content": "fail"}])

    with pytest.raises(Exception) as excinfo:
        await openai_client.get_completion(request)

    assert "API Error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_openai_client_api_errors(openai_client: OpenAILLMClient) -> None:
    # Test specific OpenAI errors
    mock_create = AsyncMock()

    # Rate Limit
    mock_create.side_effect = RateLimitError(
        message="Rate limit exceeded", response=MagicMock(), body={"message": "Rate limit exceeded"}
    )
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(messages=[{"role": "user", "content": "fast"}])

    with pytest.raises(RateLimitError) as excinfo:
        await openai_client.get_completion(request)
    assert "Rate limit exceeded" in str(excinfo.value)

    # Auth Error
    mock_create.side_effect = AuthenticationError(
        message="Invalid API Key", response=MagicMock(), body={"message": "Invalid API Key"}
    )

    with pytest.raises(AuthenticationError) as excinfo_auth:
        await openai_client.get_completion(request)
    assert "Invalid API Key" in str(excinfo_auth.value)


@pytest.mark.asyncio
async def test_openai_client_complex_nested_schema(openai_client: OpenAILLMClient) -> None:
    mock_create_wc = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-complex"

    mock_data = ComplexSchema(id=123, details=NestedSchema(sub_field="sub"), tags=["a", "b"])

    mock_response.choices = [MagicMock(message=MagicMock(), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    mock_create_wc.return_value = (mock_data, mock_response)

    cast(Any, openai_client.client).chat.completions.create_with_completion = mock_create_wc

    request = LLMRequest(messages=[{"role": "user", "content": "Complex data"}], response_schema=ComplexSchema)

    response = await openai_client.get_completion(request)

    assert response.raw_content == mock_data
    # Verify serialization contains nested data
    assert "sub" in response.content
    assert "123" in response.content

    mock_create_wc.assert_called_once()
    call_kwargs = mock_create_wc.call_args.kwargs
    assert call_kwargs["response_model"] == ComplexSchema


@pytest.mark.asyncio
async def test_openai_client_token_usage_missing(openai_client: OpenAILLMClient) -> None:
    # Verify behavior when usage is None (optional in OpenAI API)
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="No usage stats"), finish_reason="stop")]
    mock_response.usage = None  # Simulate missing usage
    mock_create.return_value = mock_response
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(messages=[{"role": "user", "content": "test"}])

    response = await openai_client.get_completion(request)

    assert response.usage == {}
    assert response.content == "No usage stats"
