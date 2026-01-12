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
from abc import ABC, abstractmethod
from typing import Any, Optional, cast

from pydantic import BaseModel, Field

from coreason_council.utils.logger import logger


class LLMRequest(BaseModel):
    """
    Standardized request object for LLM interactions.
    Supports structured output requests via 'response_schema'.
    """

    messages: list[dict[str, str]]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    # Optional Pydantic model class or dict schema for structured JSON output
    response_schema: Optional[Any] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """
    Standardized response object from LLM interactions.
    """

    content: str
    raw_content: Any = None  # The raw parsed object if structured output was requested
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    Enforces a unified interface for heterogeneous backends (OpenAI, Anthropic, Local).
    """

    @abstractmethod
    async def get_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generates a completion for the given request.

        Args:
            request: The LLMRequest object containing messages and parameters.

        Returns:
            LLMResponse object containing the text content and metadata.
        """
        pass  # pragma: no cover


class MockLLMClient(BaseLLMClient):
    """
    Mock implementation of LLM Client for testing.
    """

    def __init__(
        self,
        return_content: str = "Mock LLM Response",
        return_json: Any = None,
        delay_seconds: float = 0.0,
        failure_exception: Optional[Exception] = None,
    ) -> None:
        self.return_content = return_content
        self.return_json = return_json
        self.delay_seconds = delay_seconds
        self.failure_exception = failure_exception

    async def get_completion(self, request: LLMRequest) -> LLMResponse:
        logger.debug(f"MockLLMClient processing request with {len(request.messages)} messages.")

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if self.failure_exception:
            raise self.failure_exception

        # Simulate structured output if requested and available
        if request.response_schema and self.return_json:
            import json

            # If return_json is a dict/model, serialize it to string for 'content'
            if isinstance(self.return_json, BaseModel):
                content_str = self.return_json.model_dump_json()
            else:
                content_str = json.dumps(self.return_json)

            return LLMResponse(
                content=content_str,
                raw_content=self.return_json,
                usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                finish_reason="stop",
                provider_metadata={"mock": True},
            )

        return LLMResponse(
            content=self.return_content,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            finish_reason="stop",
            provider_metadata={"mock": True},
        )


class OpenAILLMClient(BaseLLMClient):
    """
    OpenAI implementation of LLM Client.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI client.
        Relies on OPENAI_API_KEY environment variable if api_key is not provided.
        """
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key)

    async def get_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generates a completion using OpenAI API.
        Handles both standard chat completion and structured output (beta.parse).
        """
        from openai.types.chat import ChatCompletionMessageParam

        logger.debug(f"OpenAILLMClient processing request with {len(request.messages)} messages.")

        # Prepare messages
        # We need to cast the dictionaries to ChatCompletionMessageParam because OpenAI's types are TypedDicts
        # and LLMRequest.messages is just list[dict[str, str]].
        messages_list: list[ChatCompletionMessageParam] = []

        if request.system_prompt:
            messages_list.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            # Assuming msg has 'role' and 'content' keys as strings.
            # We trust the caller or validation elsewhere, here we just cast to satisfy mypy.
            messages_list.append(cast(ChatCompletionMessageParam, msg))

        # Default model if not specified in metadata
        model = str(request.metadata.get("model", "gpt-4o"))

        try:
            if (
                request.response_schema
                and isinstance(request.response_schema, type)
                and issubclass(request.response_schema, BaseModel)
            ):
                # Use Structured Outputs (beta.parse)
                logger.debug(f"Using beta.chat.completions.parse with schema {request.response_schema.__name__}")
                completion_parsed = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=messages_list,
                    response_format=request.response_schema,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                parsed_content = completion_parsed.choices[0].message.parsed
                finish_reason = completion_parsed.choices[0].finish_reason

                # We need to serialize the parsed content back to string for consistency
                # If parsed_content is None (refusal), handle it
                if parsed_content is None:
                    # Fallback to refusal content if available
                    content_str = getattr(completion_parsed.choices[0].message, "refusal", "") or ""
                else:
                    content_str = parsed_content.model_dump_json()

                raw_content = parsed_content
                usage_obj = completion_parsed.usage
                completion_id = completion_parsed.id

            else:
                # Standard completion
                # Check if generic JSON mode is requested via metadata or if response_schema is a dict
                response_format = None
                if isinstance(request.response_schema, dict):
                    # It's a dict schema, might be json_schema or simple json_object
                    if request.metadata.get("json_mode", False):
                        response_format = {"type": "json_object"}

                # For standard create, we need to cast response_format properly if strictly typed,
                # but let's see if Any works or if we need specific type.
                # OpenAI expects ResponseFormat or dict.

                completion = await self.client.chat.completions.create(
                    model=model,
                    messages=messages_list,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=cast(Any, response_format),
                )
                content_str = completion.choices[0].message.content or ""
                raw_content = None
                finish_reason = completion.choices[0].finish_reason
                usage_obj = completion.usage
                completion_id = completion.id

            # Extract usage
            usage = {}
            if usage_obj:
                usage = {
                    "prompt_tokens": usage_obj.prompt_tokens,
                    "completion_tokens": usage_obj.completion_tokens,
                    "total_tokens": usage_obj.total_tokens,
                }

            return LLMResponse(
                content=content_str,
                raw_content=raw_content,
                usage=usage,
                finish_reason=str(finish_reason) if finish_reason else None,
                provider_metadata={"model": model, "id": completion_id},
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e
