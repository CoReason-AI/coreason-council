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
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
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


class GatewayLLMClient(BaseLLMClient):
    """
    LLM Client that talks to the Internal Gateway Service (Service H).
    """

    def __init__(self, gateway_url: Optional[str] = None, access_token: Optional[str] = None) -> None:
        self.gateway_url = gateway_url or settings.gateway_url
        self.access_token = access_token or settings.gateway_access_token

    async def get_completion(self, request: LLMRequest) -> LLMResponse:
        logger.debug(f"GatewayLLMClient calling {self.gateway_url} with {len(request.messages)} messages.")

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.extend(request.messages)

        model = str(request.metadata.get("model", "gpt-4o"))
        response_format = None

        # Handle structured output schema injection
        if (
            request.response_schema
            and isinstance(request.response_schema, type)
            and issubclass(request.response_schema, BaseModel)
        ):
            # Inject schema into system prompt or as response format if supported
            # We assume OpenAI-compatible "response_format={'type': 'json_object'}" and prompting
            schema = request.response_schema.model_json_schema()
            schema_str = json.dumps(schema, indent=2)

            # Append instructions to the last system message or create one
            instruction = (
                f"\n\nIMPORTANT: You must respond with valid JSON matching the following schema:\n{schema_str}"
            )

            # Find last system message or insert one
            found_system = False
            for msg in messages:
                if msg["role"] == "system":
                    msg["content"] += instruction
                    found_system = True
                    break
            if not found_system:
                messages.insert(0, {"role": "system", "content": instruction})

            response_format = {"type": "json_object"}

        payload = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Content-Type": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        async with httpx.AsyncClient() as client:
            try:
                # Append /chat/completions if not present, but settings default is /v1 which implies we need path
                url = self.gateway_url.rstrip("/") + "/chat/completions"
                response = await client.post(url, json=payload, headers=headers, timeout=60.0)
                response.raise_for_status()
                data = response.json()
            except httpx.RequestError as exc:
                logger.error(f"An error occurred while requesting {exc.request.url!r}.")
                raise
            except httpx.HTTPStatusError as exc:
                logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
                raise

        # Parse response
        try:
            choice = data["choices"][0]
            content_str = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage_data = data.get("usage", {})

            raw_content = None
            if request.response_schema and response_format:
                # Parse JSON content
                try:
                    parsed_json = json.loads(content_str)
                    raw_content = request.response_schema.model_validate(parsed_json)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    raise ValueError(f"LLM failed to return valid JSON matching schema: {e}") from e

            return LLMResponse(
                content=content_str,
                raw_content=raw_content,
                usage={
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
                finish_reason=finish_reason,
                provider_metadata={"model": model, "id": data.get("id")},
            )

        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected response format from Gateway: {e}")
            raise ValueError(f"Invalid response format from Gateway: {e}") from e
