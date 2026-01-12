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
        max_rounds: int = 3,
    ) -> None:
        """
        Initializes the Chamber Speaker with the necessary components.

        Args:
            proposers: A sequence of Proposer instances (The Voices).
            personas: A sequence of Personas corresponding to the proposers.
            dissenter: The Dissenter instance (The Critic).
            aggregator: The Aggregator instance (The Judge).
            entropy_threshold: The threshold below which consensus is accepted immediately.
            max_rounds: The maximum number of debate rounds before triggering a deadlock.
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
        self.max_rounds = max_rounds

        logger.info(
            f"ChamberSpeaker initialized with {len(self.proposers)} proposers, "
            f"entropy threshold {self.entropy_threshold}, and max rounds {self.max_rounds}."
        )

    async def resolve_query(self, query: str, max_rounds: int | None = None) -> tuple[Verdict, CouncilTrace]:
        """
        Orchestrates the resolution of a query through the Council.

        Args:
            query: The input question or problem statement.
            max_rounds: Optional override for the maximum number of debate rounds.

        Returns:
            A tuple containing the final Verdict and the full CouncilTrace.
        """
        session_id = str(uuid.uuid4())
        roster_names = [p.name for p in self.personas]
        current_max_rounds = max_rounds if max_rounds is not None else self.max_rounds

        trace = CouncilTrace(
            session_id=session_id,
            roster=roster_names,
            topology=TopologyType.STAR,  # Default for single-round/low-entropy flow
        )

        logger.info(f"Session {session_id}: Speaker received query: '{query}'")

        # --- Phase 1: Initial Proposals (Parallel Isolation) ---
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

        # Loop variables
        current_round = 1
        critiques: list[Critique] = []
        is_deadlock = False

        # --- Phase 2: Divergence-Convergence Loop ---
        while True:
            # Check Entropy
            entropy = await self.dissenter.calculate_entropy(proposals)
            trace.entropy_score = entropy  # Update trace with latest entropy
            logger.info(f"Session {session_id} (Round {current_round}): Calculated entropy score: {entropy}")

            if entropy <= self.entropy_threshold:
                logger.info(
                    f"Session {session_id}: Low entropy ({entropy} <= {self.entropy_threshold}) detected. "
                    "Consensus reached."
                )
                break

            if current_round >= current_max_rounds:
                logger.warning(
                    f"Session {session_id}: Max rounds ({current_max_rounds}) reached with high entropy "
                    f"({entropy} > {self.entropy_threshold}). Declaring Deadlock."
                )
                is_deadlock = True
                break

            logger.warning(
                f"Session {session_id}: High entropy ({entropy} > {self.entropy_threshold}) detected. "
                f"Initiating Round {current_round} Peer Critique."
            )
            trace.topology = TopologyType.ROUND_TABLE  # Switch topology log

            # 1. Peer Critique Logic: Everyone critiques everyone else
            critique_tasks = []
            for i, (_proposer_target, proposal) in enumerate(zip(self.proposers, proposals, strict=True)):
                for j, (proposer_critic, persona_critic) in enumerate(zip(self.proposers, self.personas, strict=True)):
                    if i == j:
                        continue  # Do not critique self

                    critique_tasks.append(proposer_critic.critique_proposal(proposal, persona_critic))

            # Store critiques for this round (clearing previous ones as we want fresh context,
            # or we could accumulate them. The prompt implies revising based on critiques.
            # Usually we want the Aggregator to see the *final* state or latest relevant critiques.
            # We'll refresh the critiques list for the current round.)
            critiques = await asyncio.gather(*critique_tasks)

            # Log critiques
            for c in critiques:
                trace.log_interaction(actor=c.reviewer_id, action=f"critique_round_{current_round}", content=c.content)

            # 2. Proposal Revision Logic
            revision_tasks = []
            # Map critiques to their target proposers to pass relevant feedback
            # critiques structure: [Critique(reviewer=B, target=A), Critique(reviewer=C, target=A), ...]
            for proposer, proposal, persona in zip(self.proposers, proposals, self.personas, strict=True):
                # Filter critiques targeting this proposer
                my_critiques = [c for c in critiques if c.target_proposer_id == proposal.proposer_id]
                revision_tasks.append(proposer.revise_proposal(proposal, my_critiques, persona))

            proposals = await asyncio.gather(*revision_tasks)

            # Log revisions
            for proposal, persona in zip(proposals, self.personas, strict=True):
                trace.log_interaction(
                    actor=persona.name,
                    action=f"revise_round_{current_round}",
                    content=proposal.content,
                )

            current_round += 1

        # --- Phase 3: Final Aggregation ---
        # We pass the final set of proposals and the latest set of critiques (if any exist)
        # If we broke early due to low entropy, critiques might be empty (Story A).
        verdict = await self.aggregator.aggregate(proposals, critiques=critiques, is_deadlock=is_deadlock)
        trace.final_verdict = verdict
        trace.log_interaction(
            actor="Aggregator",
            action="verdict",
            content=verdict.content,
        )

        # Populate vote tally
        if verdict.alternatives:
            # Deadlock scenario: Count supporters for each alternative
            trace.vote_tally = {alt.label: len(set(alt.supporters)) for alt in verdict.alternatives}
        else:
            # Consensus scenario: Assume all participating proposers support the consensus
            trace.vote_tally = {"Consensus": len(self.proposers)}

        return verdict, trace
