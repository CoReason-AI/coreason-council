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
from typing import Any, Optional

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
