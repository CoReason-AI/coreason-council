import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from coreason_council.core.llm_client import LLMRequest, LLMResponse, OpenAILLMClient


class MockStructuredOutput(BaseModel):
    reasoning: str
    verdict: str


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
