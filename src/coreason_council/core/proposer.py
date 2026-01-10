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

from coreason_council.core.types import Persona, ProposerOutput
from coreason_council.utils.logger import logger


class BaseProposer(ABC):
    """
    Abstract base class for all Proposers (Voices).
    Enforces the interface for generating proposals based on a persona.
    """

    @abstractmethod
    async def propose(self, query: str, persona: Persona) -> ProposerOutput:
        """
        Generates a proposal for the given query using the specified persona.

        Args:
            query: The input question or topic.
            persona: The persona (system prompt/role) to adopt.

        Returns:
            ProposerOutput containing the response and confidence score.
        """
        pass  # pragma: no cover


class MockProposer(BaseProposer):
    """
    A mock implementation of a Proposer for testing and development.
    """

    def __init__(
        self,
        return_content: str = "Mock response",
        return_confidence: float = 0.9,
        proposer_id_prefix: str = "mock-proposer",
        delay_seconds: float = 0.0,
    ) -> None:
        self.return_content = return_content
        self.return_confidence = return_confidence
        self.proposer_id_prefix = proposer_id_prefix
        self.delay_seconds = delay_seconds

    async def propose(self, query: str, persona: Persona) -> ProposerOutput:
        logger.info(f"MockProposer processing query: '{query}' with persona: '{persona.name}'")

        # Simulate processing
        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        content = f"{self.return_content} (Query: {query}, Persona: {persona.name})"

        return ProposerOutput(
            proposer_id=f"{self.proposer_id_prefix}-{persona.name}",
            content=content,
            confidence=self.return_confidence,
            metadata={"mock": True, "persona_capabilities": persona.capabilities},
        )
