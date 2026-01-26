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

from coreason_council.core.aggregator import BaseAggregator
from coreason_council.core.llm_client import BaseLLMClient, LLMRequest
from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.verdict import Verdict, VerdictOption
from coreason_council.utils.logger import logger


class VerdictOptionContent(BaseModel):
    """Internal schema for LLM structured generation of verdict options."""

    label: str = Field(description="A short label for this option (e.g. 'Option A').")
    content: str = Field(description="The detailed description of this option.")
    supporters: list[str] = Field(description="List of proposer IDs supporting this option.")


class VerdictContent(BaseModel):
    """Internal schema for LLM structured generation of the final verdict."""

    content: str = Field(description="The final verdict content or synthesis.")
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")]
    supporting_evidence: list[str] = Field(
        default_factory=list, description="List of specific evidence points supporting the verdict."
    )
    dissenting_opinions: list[str] = Field(default_factory=list, description="List of dissenting opinions or concerns.")
    alternatives: list[VerdictOptionContent] = Field(
        default_factory=list, description="List of alternative options if no consensus was reached (Deadlock)."
    )


class LLMAggregator(BaseAggregator):
    """
    An Aggregator implementation that uses a real LLM via BaseLLMClient.
    """

    def __init__(self, llm_client: BaseLLMClient, model: str = "gpt-4o") -> None:
        """
        Initializes the LLMAggregator.

        Args:
            llm_client: The LLM client to use for generating verdicts.
            model: The LLM model identifier to use (e.g. 'gpt-4o').
        """
        self.llm_client = llm_client
        self.model = model

    async def aggregate(
        self,
        proposals: list[ProposerOutput],
        critiques: list[Critique],
        is_deadlock: bool = False,
    ) -> Verdict:
        """
        Synthesizes a final verdict using the LLM.
        """
        logger.info(
            f"LLMAggregator aggregating {len(proposals)} proposals and {len(critiques)} critiques "
            f"(Deadlock: {is_deadlock})"
        )

        system_prompt = (
            "You are 'The Judge', a synthesizer engine in a high-stakes consensus council. "
            "Your goal is to evaluate multiple proposals and peer critiques to produce a final, "
            "high-confidence verdict. "
            "You must be objective, identifying the strongest arguments and discarding hallucinations or weak logic."
        )

        # Build context string
        proposals_text = "\n\n".join(
            [f"Proposal from {p.proposer_id}:\n{p.content}\nConfidence: {p.confidence}" for p in proposals]
        )

        critiques_text = "None"
        if critiques:
            critiques_text = "\n\n".join(
                [
                    f"Critique by {c.reviewer_id} targeting {c.target_proposer_id}:\n"
                    f"{c.content}\nFlaws: {c.flaws_identified}"
                    for c in critiques
                ]
            )

        if is_deadlock:
            # Instruction for Minority Report
            instruction = (
                "The council has failed to reach consensus (Deadlock). "
                "You must generate a 'Minority Report' that clearly outlines the competing options. "
                "Populate the 'alternatives' field with distinct viewpoints (Option A, Option B, etc.) "
                "and assign the respective supporters (proposer_ids) to each option. "
                "The main 'content' should summarize the conflict. "
                "Assign a low confidence score (e.g., < 0.5)."
            )
        else:
            # Instruction for Consensus Synthesis
            instruction = (
                "Synthesize a single, coherent final answer ('content') that incorporates the best elements "
                "of the proposals and addresses valid critiques. "
                "Cite specific evidence in 'supporting_evidence'. "
                "Note any remaining concerns in 'dissenting_opinions'. "
                "Leave 'alternatives' empty as this is a consensus verdict."
            )

        user_prompt = f"Proposals:\n{proposals_text}\n\nCritiques:\n{critiques_text}\n\nInstructions:\n{instruction}"

        request = LLMRequest(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=system_prompt,
            response_schema=VerdictContent,
            metadata={"task": "aggregate", "is_deadlock": is_deadlock, "model": self.model},
        )

        response = await self.llm_client.get_completion(request)

        if response.raw_content and isinstance(response.raw_content, VerdictContent):
            result = response.raw_content
        else:
            raise ValueError("LLM failed to return structured VerdictContent.")

        # Map internal VerdictOptionContent to domain VerdictOption
        domain_alternatives = [
            VerdictOption(label=alt.label, content=alt.content, supporters=alt.supporters)
            for alt in result.alternatives
        ]

        return Verdict(
            content=result.content,
            confidence_score=result.confidence_score,
            supporting_evidence=result.supporting_evidence,
            dissenting_opinions=result.dissenting_opinions,
            alternatives=domain_alternatives,
        )
