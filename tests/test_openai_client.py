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

    # Cast to Any to allow method assignment for mocking
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
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


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
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1] == {"role": "user", "content": "Hi"}


@pytest.mark.asyncio
async def test_openai_client_structured_output(openai_client: OpenAILLMClient) -> None:
    # Mock beta.chat.completions.parse
    mock_parse = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-struct"

    mock_parsed_obj = MockStructuredOutput(reasoning="Because X", verdict="True")
    mock_response.choices = [MagicMock(message=MagicMock(parsed=mock_parsed_obj), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=20, completion_tokens=10, total_tokens=30)
    mock_parse.return_value = mock_response

    cast(Any, openai_client.client).beta.chat.completions.parse = mock_parse

    request = LLMRequest(messages=[{"role": "user", "content": "Analyze this"}], response_schema=MockStructuredOutput)

    response = await openai_client.get_completion(request)

    assert response.raw_content == mock_parsed_obj
    # Content should be the serialized JSON of the Pydantic object
    assert "Because X" in response.content
    assert "True" in response.content

    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs["response_format"] == MockStructuredOutput


@pytest.mark.asyncio
async def test_openai_client_structured_output_refusal(openai_client: OpenAILLMClient) -> None:
    # Mock beta.chat.completions.parse with refusal
    mock_parse = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-refused"

    # Simulate refusal: parsed is None, refusal is populated
    mock_response.choices = [
        MagicMock(message=MagicMock(parsed=None, refusal="I cannot answer this."), finish_reason="stop")
    ]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    mock_parse.return_value = mock_response

    cast(Any, openai_client.client).beta.chat.completions.parse = mock_parse

    request = LLMRequest(
        messages=[{"role": "user", "content": "Do something bad"}], response_schema=MockStructuredOutput
    )

    response = await openai_client.get_completion(request)

    assert response.raw_content is None
    assert response.content == "I cannot answer this."
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_openai_client_json_mode(openai_client: OpenAILLMClient) -> None:
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"foo": "bar"}'), finish_reason="stop")]
    mock_create.return_value = mock_response
    cast(Any, openai_client.client).chat.completions.create = mock_create

    request = LLMRequest(
        messages=[{"role": "user", "content": "json please"}],
        response_schema={"type": "object"},  # Dict schema implies checking for json_mode metadata or custom
        metadata={"json_mode": True},
    )

    await openai_client.get_completion(request)

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


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
    mock_parse = AsyncMock()
    mock_response = MagicMock()
    mock_response.id = "chatcmpl-complex"

    mock_data = ComplexSchema(id=123, details=NestedSchema(sub_field="sub"), tags=["a", "b"])

    mock_response.choices = [MagicMock(message=MagicMock(parsed=mock_data), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    mock_parse.return_value = mock_response

    cast(Any, openai_client.client).beta.chat.completions.parse = mock_parse

    request = LLMRequest(messages=[{"role": "user", "content": "Complex data"}], response_schema=ComplexSchema)

    response = await openai_client.get_completion(request)

    assert response.raw_content == mock_data
    # Verify serialization contains nested data
    assert "sub" in response.content
    assert "123" in response.content

    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args.kwargs
    assert call_kwargs["response_format"] == ComplexSchema


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


@pytest.mark.asyncio
async def test_openai_client_complex_history(openai_client: OpenAILLMClient) -> None:
    # Verify mult-turn conversation
    mock_create = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Response"), finish_reason="stop")]
    mock_create.return_value = mock_response
    cast(Any, openai_client.client).chat.completions.create = mock_create

    messages = [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]
    request = LLMRequest(messages=messages)

    await openai_client.get_completion(request)

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    sent_messages = call_kwargs["messages"]
    assert len(sent_messages) == 4
    assert sent_messages[0]["content"] == "Sys"
    assert sent_messages[3]["content"] == "Q2"
