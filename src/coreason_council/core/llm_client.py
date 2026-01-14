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

import instructor
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from coreason_council.settings import settings
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
    OpenAI implementation of LLM Client using Instructor.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initializes the OpenAI client patched with Instructor.
        Relies on settings.openai_api_key if api_key is not provided.
        """
        key = api_key or settings.openai_api_key
        # We assume OPENAI_API_KEY is available in env or settings
        self.client = instructor.from_openai(AsyncOpenAI(api_key=key))

    async def get_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Generates a completion using OpenAI API via Instructor.
        """
        logger.debug(f"OpenAILLMClient processing request with {len(request.messages)} messages.")

        # Prepare messages
        messages: list[ChatCompletionMessageParam] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            # msg is dict[str, str], we cast to ChatCompletionMessageParam to satisfy type checker
            # Assuming structure is correct (role, content)
            messages.append(cast(ChatCompletionMessageParam, msg))

        # Default model if not specified in metadata
        model = str(request.metadata.get("model", "gpt-4o"))

        try:
            if (
                request.response_schema
                and isinstance(request.response_schema, type)
                and issubclass(request.response_schema, BaseModel)
            ):
                # Use Instructor for structured output
                logger.debug(f"Using instructor.chat.completions.create with schema {request.response_schema.__name__}")

                # Instructor's create returns the parsed object directly
                response_model = request.response_schema

                # We use create_with_completion to get usage stats
                parsed_content, completion = await self.client.chat.completions.create_with_completion(
                    model=model,
                    messages=messages,
                    response_model=response_model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )

                content_str = parsed_content.model_dump_json()
                raw_content = parsed_content
                usage_obj = completion.usage
                finish_reason = completion.choices[0].finish_reason
                completion_id = completion.id

            else:
                # Standard completion
                # Actually, instructor wraps the client. If we want raw completion without validation,
                # we might need to access the original client or pass response_model=None if supported.
                # Instructor docs say:
                # client.chat.completions.create(..., response_model=None) returns regular response.

                completion = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_model=None,
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
