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

from coreason_council.core.types import Critique, Persona, ProposerOutput
from coreason_council.utils.logger import logger


class BaseDissenter(ABC):
    """
    Abstract base class for the Dissenter (The Critic).
    Responsible for falsification, critique generation, and entropy calculation.
    """

    @abstractmethod
    async def critique(
        self,
        target_proposal: ProposerOutput,
        persona: Persona,
    ) -> Critique:
        """
        Generates a critique for a specific proposal using the Dissenter persona.

        Args:
            target_proposal: The proposal to critique.
            persona: The persona (system prompt/role) acting as the critic (e.g., "The Skeptic").

        Returns:
            Critique object containing flaws and agreement score.
        """
        pass  # pragma: no cover

    @abstractmethod
    async def calculate_entropy(self, proposals: list[ProposerOutput]) -> float:
        """
        Calculates the semantic entropy (disagreement score) between multiple proposals.

        Args:
            proposals: A list of proposals to analyze.

        Returns:
            A float between 0.0 (total agreement) and 1.0 (total chaos/disagreement).
        """
        pass  # pragma: no cover


class MockDissenter(BaseDissenter):
    """
    A mock implementation of a Dissenter for testing and development.
    """

    def __init__(
        self,
        default_agreement_score: float = 0.5,
        default_entropy_score: float = 0.1,
        default_flaws: list[str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.default_agreement_score = default_agreement_score
        self.default_entropy_score = default_entropy_score
        self.default_flaws = default_flaws or ["Potential logical fallacy detected", "Citation needed"]
        self.delay_seconds = delay_seconds

    async def critique(
        self,
        target_proposal: ProposerOutput,
        persona: Persona,
    ) -> Critique:
        logger.info(f"MockDissenter critiquing proposal '{target_proposal.proposer_id}' as '{persona.name}'")

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        content = (
            f"Critique by {persona.name}: The proposal makes some valid points but lacks specific evidence. "
            f"(Target: {target_proposal.proposer_id})"
        )

        return Critique(
            reviewer_id=persona.name,
            target_proposer_id=target_proposal.proposer_id,
            content=content,
            flaws_identified=self.default_flaws,
            agreement_score=self.default_agreement_score,
        )

    async def calculate_entropy(self, proposals: list[ProposerOutput]) -> float:
        logger.info(f"MockDissenter calculating entropy for {len(proposals)} proposals.")

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        # In a real implementation, this would compare contents.
        # Here we just return the configured mock value.
        return self.default_entropy_score
