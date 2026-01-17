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
import re
from abc import ABC, abstractmethod
from itertools import combinations

from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.persona import Persona
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

        if len(proposals) <= 1:
            return 0.0

        # In a real implementation, this would compare contents.
        # Here we just return the configured mock value.
        return self.default_entropy_score


class JaccardDissenter(BaseDissenter):
    """
    A Dissenter implementation that calculates semantic entropy using Jaccard Similarity.
    Agreement = Jaccard Similarity (Intersection / Union)
    Entropy = 1.0 - Average Pairwise Agreement
    """

    def _tokenize(self, text: str) -> set[str]:
        """
        Simple tokenizer: lowercase and split by non-alphanumeric characters.
        """
        if not text:
            return set()
        # Find all sequences of alphanumeric characters
        tokens = re.findall(r"\w+", text.lower())
        return set(tokens)

    def _calculate_jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates Jaccard Similarity between two texts.
        Score = |Intersection| / |Union|
        """
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if not tokens_a and not tokens_b:
            return 1.0  # Both empty = identical

        if not tokens_a or not tokens_b:
            return 0.0  # One empty, one not = completely different

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union

    async def critique(
        self,
        target_proposal: ProposerOutput,
        persona: Persona,
    ) -> Critique:
        """
        Generates a basic critique.
        For JaccardDissenter, the critique logic is not the primary focus, so we return a placeholder.
        """
        logger.info(f"JaccardDissenter critiquing proposal '{target_proposal.proposer_id}'")

        # Basic analysis: Word count
        word_count = len(self._tokenize(target_proposal.content))

        content = (
            f"Critique by {persona.name}: Jaccard analysis complete. "
            f"Proposal contains approximately {word_count} unique tokens."
        )

        return Critique(
            reviewer_id=persona.name,
            target_proposer_id=target_proposal.proposer_id,
            content=content,
            flaws_identified=["Standard Jaccard Review - No semantic flaws detected via set ops."],
            agreement_score=0.5,  # Neutral default
        )

    async def calculate_entropy(self, proposals: list[ProposerOutput]) -> float:
        """
        Calculates entropy as 1.0 - Average Pairwise Jaccard Similarity.
        """
        count = len(proposals)
        logger.info(f"JaccardDissenter calculating entropy for {count} proposals.")

        if count <= 1:
            return 0.0

        similarity_sum = 0.0
        pair_count = 0

        # Calculate pairwise similarity for all unique combinations
        for p1, p2 in combinations(proposals, 2):
            sim = self._calculate_jaccard_similarity(p1.content, p2.content)
            similarity_sum += sim
            pair_count += 1

        average_similarity = similarity_sum / pair_count if pair_count > 0 else 1.0
        entropy = 1.0 - average_similarity

        logger.debug(f"Average Jaccard Similarity: {average_similarity}, Entropy: {entropy}")
        return entropy
