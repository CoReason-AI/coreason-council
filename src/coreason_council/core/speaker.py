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
import uuid
from typing import Sequence

from coreason_council.core.aggregator import BaseAggregator
from coreason_council.core.dissenter import BaseDissenter
from coreason_council.core.proposer import BaseProposer
from coreason_council.core.types import CouncilTrace, Critique, Persona, TopologyType, Verdict
from coreason_council.utils.logger import logger


class ChamberSpeaker:
    """
    The Chamber Speaker (The Orchestrator).
    Manages the lifecycle of a Council Session, orchestrates Proposers,
    Dissenter, and Aggregator to reach a consensus.
    """

    def __init__(
        self,
        proposers: Sequence[BaseProposer],
        personas: Sequence[Persona],
        dissenter: BaseDissenter,
        aggregator: BaseAggregator,
        entropy_threshold: float = 0.1,
    ) -> None:
        """
        Initializes the Chamber Speaker with the necessary components.

        Args:
            proposers: A sequence of Proposer instances (The Voices).
            personas: A sequence of Personas corresponding to the proposers.
            dissenter: The Dissenter instance (The Critic).
            aggregator: The Aggregator instance (The Judge).
            entropy_threshold: The threshold below which consensus is accepted immediately.
        """
        if not proposers:
            raise ValueError("The Council requires at least one Proposer.")

        if not personas:
            raise ValueError("The Council requires at least one Persona.")

        if len(proposers) != len(personas):
            raise ValueError(
                f"Count mismatch: {len(proposers)} proposers vs {len(personas)} personas. "
                "Each proposer must have an assigned persona."
            )

        if dissenter is None:
            raise ValueError("The Council requires a Dissenter.")

        if aggregator is None:
            raise ValueError("The Council requires an Aggregator.")

        # Create a defensive copy to ensure immutability from the outside
        self.proposers = list(proposers)
        self.personas = list(personas)
        self.dissenter = dissenter
        self.aggregator = aggregator
        self.entropy_threshold = entropy_threshold

        logger.info(
            f"ChamberSpeaker initialized with {len(self.proposers)} proposers "
            f"and entropy threshold {self.entropy_threshold}."
        )

    async def resolve_query(self, query: str) -> tuple[Verdict, CouncilTrace]:
        """
        Orchestrates the resolution of a query through the Council.

        Args:
            query: The input question or problem statement.

        Returns:
            A tuple containing the final Verdict and the full CouncilTrace.
        """
        session_id = str(uuid.uuid4())
        roster_names = [p.name for p in self.personas]

        trace = CouncilTrace(
            session_id=session_id,
            roster=roster_names,
            topology=TopologyType.STAR,  # Default for single-round/low-entropy flow
        )

        logger.info(f"Session {session_id}: Speaker received query: '{query}'")

        # --- Phase 1: Proposals (Parallel Isolation) ---
        logger.debug(f"Session {session_id}: Requesting proposals from {len(self.proposers)} agents.")

        # Prepare tasks
        tasks = []
        for proposer, persona in zip(self.proposers, self.personas, strict=True):
            tasks.append(proposer.propose(query, persona))

        # Execute concurrently
        proposals = await asyncio.gather(*tasks)

        # Log interactions
        for proposal, persona in zip(proposals, self.personas, strict=True):
            trace.log_interaction(
                actor=persona.name,
                action="propose",
                content=proposal.content,
            )

        # --- Phase 2: Entropy Check ---
        entropy = await self.dissenter.calculate_entropy(proposals)
        trace.entropy_score = entropy
        logger.info(f"Session {session_id}: Calculated entropy score: {entropy}")

        # --- Phase 3: Decision & Aggregation ---
        critiques: list[Critique] = []

        if entropy <= self.entropy_threshold:
            logger.info(f"Session {session_id}: Low entropy detected. Proceeding to immediate aggregation.")
        else:
            logger.warning(
                f"Session {session_id}: High entropy ({entropy} > {self.entropy_threshold}) detected. "
                "Initiating Peer Critique Round."
            )
            # Switch topology log to indicate debate occurred
            trace.topology = TopologyType.ROUND_TABLE

            # Peer Critique Logic: Everyone critiques everyone else
            critique_tasks = []
            for i, (_proposer_target, proposal) in enumerate(zip(self.proposers, proposals, strict=True)):
                for j, (proposer_critic, persona_critic) in enumerate(zip(self.proposers, self.personas, strict=True)):
                    if i == j:
                        continue  # Do not critique self

                    # Create task for Proposer J to critique Proposal I
                    critique_tasks.append(proposer_critic.critique_proposal(proposal, persona_critic))

            if critique_tasks:
                logger.debug(f"Session {session_id}: Launching {len(critique_tasks)} critique tasks.")
                critiques = await asyncio.gather(*critique_tasks)

                # Log critiques
                for c in critiques:
                    trace.log_interaction(actor=c.reviewer_id, action="critique", content=c.content)

        # Final Aggregation (happens for both Low and High entropy now)
        verdict = await self.aggregator.aggregate(proposals, critiques=critiques)
        trace.final_verdict = verdict
        trace.log_interaction(
            actor="Aggregator",
            action="verdict",
            content=verdict.content,
        )
        return verdict, trace
