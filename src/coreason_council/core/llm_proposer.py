# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Annotated

from pydantic import BaseModel, Field

from coreason_council.core.llm_client import BaseLLMClient, LLMRequest
from coreason_council.core.proposer import BaseProposer
from coreason_council.core.types import Critique, Persona, ProposerOutput
from coreason_council.utils.logger import logger


class ProposalContent(BaseModel):
    """Internal schema for LLM structured generation of proposals."""

    content: str = Field(description="The detailed content of the proposal or answer.")
    confidence: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")]


class CritiqueContent(BaseModel):
    """Internal schema for LLM structured generation of critiques."""

    content: str = Field(description="The detailed critique of the proposal.")
    flaws_identified: list[str] = Field(description="A list of specific flaws or issues identified.")
    agreement_score: Annotated[float, Field(ge=0.0, le=1.0, description="Agreement score between 0.0 and 1.0.")]


class LLMProposer(BaseProposer):
    """
    A Proposer implementation that uses a real LLM via BaseLLMClient.
    """

    def __init__(self, llm_client: BaseLLMClient) -> None:
        """
        Initializes the LLMProposer.

        Args:
            llm_client: The LLM client to use for generating responses.
        """
        self.llm_client = llm_client

    async def propose(self, query: str, persona: Persona) -> ProposerOutput:
        """
        Generates a proposal using the LLM.
        """
        logger.info(f"LLMProposer generating proposal for query: '{query}' as '{persona.name}'")

        system_prompt = (
            f"{persona.system_prompt}\n"
            "You are a member of a consensus council. Your goal is to provide a distinct "
            "and well-reasoned answer to the user's query based on your persona."
        )

        user_prompt = f"Query: {query}\n\nPlease provide your answer and a self-assessed confidence score."

        request = LLMRequest(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            response_schema=ProposalContent,
            metadata={"persona": persona.name, "task": "propose"},
        )

        response = await self.llm_client.get_completion(request)

        # Handle potential failure to parse (though OpenAILLMClient should handle it or raise)
        if response.raw_content and isinstance(response.raw_content, ProposalContent):
            result = response.raw_content
        else:
            # Fallback if raw_content isn't populated (e.g. non-structured client or error)
            # For now, we assume strict adherence to the client contract which says raw_content
            # is populated if response_schema was used.
            # But let's be safe for typing.
            raise ValueError("LLM failed to return structured ProposalContent.")

        return ProposerOutput(
            proposer_id=f"llm-{persona.name.lower()}",
            content=result.content,
            confidence=result.confidence,
            metadata={
                "persona": persona.name,
                "usage": response.usage,
                "model": response.provider_metadata.get("model"),
            },
        )

    async def critique_proposal(self, target_proposal: ProposerOutput, persona: Persona) -> Critique:
        """
        Generates a critique using the LLM.
        """
        logger.info(f"LLMProposer '{persona.name}' critiquing '{target_proposal.proposer_id}'")

        system_prompt = (
            f"{persona.system_prompt}\n"
            "You are a critical reviewer in a consensus council. Your goal is to critique the provided proposal "
            "identifying logical fallacies, missing evidence, or bias."
        )

        user_prompt = (
            f"Please critique the following proposal.\n\n"
            f"Target Proposer ID: {target_proposal.proposer_id}\n"
            f"Proposal Content: {target_proposal.content}\n\n"
            "Provide a detailed critique, a list of specific flaws, and an agreement score (0.0 to 1.0)."
        )

        request = LLMRequest(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            response_schema=CritiqueContent,
            metadata={"persona": persona.name, "task": "critique"},
        )

        response = await self.llm_client.get_completion(request)

        if response.raw_content and isinstance(response.raw_content, CritiqueContent):
            result = response.raw_content
        else:
            raise ValueError("LLM failed to return structured CritiqueContent.")

        return Critique(
            reviewer_id=persona.name,
            target_proposer_id=target_proposal.proposer_id,
            content=result.content,
            flaws_identified=result.flaws_identified,
            agreement_score=result.agreement_score,
        )

    async def revise_proposal(
        self, original_proposal: ProposerOutput, critiques: list[Critique], persona: Persona
    ) -> ProposerOutput:
        """
        Revises a proposal using the LLM based on critiques.
        """
        logger.info(f"LLMProposer '{persona.name}' revising proposal based on {len(critiques)} critiques.")

        if not critiques:
            # No critiques, return original
            logger.info("No critiques provided, returning original proposal.")
            return original_proposal

        system_prompt = (
            f"{persona.system_prompt}\n"
            "You are a member of a consensus council. You have received critiques on your previous answer. "
            "Your goal is to revise your answer to address valid feedback while maintaining your perspective "
            "if the critique is unconvincing."
        )

        critiques_text = "\n\n".join(
            [f"Critique from {c.reviewer_id}:\n{c.content}\nFlaws: {c.flaws_identified}" for c in critiques]
        )

        user_prompt = (
            f"Original Proposal:\n{original_proposal.content}\n\n"
            f"Received Critiques:\n{critiques_text}\n\n"
            "Please provide a revised answer and a new confidence score."
        )

        request = LLMRequest(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            response_schema=ProposalContent,
            metadata={"persona": persona.name, "task": "revise"},
        )

        response = await self.llm_client.get_completion(request)

        if response.raw_content and isinstance(response.raw_content, ProposalContent):
            result = response.raw_content
        else:
            raise ValueError("LLM failed to return structured ProposalContent.")

        return ProposerOutput(
            proposer_id=original_proposal.proposer_id,
            content=result.content,
            confidence=result.confidence,
            metadata={
                "persona": persona.name,
                "usage": response.usage,
                "model": response.provider_metadata.get("model"),
                "revision_of": original_proposal.proposer_id,
                "critique_count": len(critiques),
            },
        )
