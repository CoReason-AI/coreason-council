# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Sequence

from coreason_council.core.aggregator import BaseAggregator
from coreason_council.core.dissenter import BaseDissenter
from coreason_council.core.proposer import BaseProposer
from coreason_council.core.types import CouncilTrace, Verdict
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
        dissenter: BaseDissenter,
        aggregator: BaseAggregator,
    ) -> None:
        """
        Initializes the Chamber Speaker with the necessary components.

        Args:
            proposers: A sequence of Proposer instances (The Voices).
            dissenter: The Dissenter instance (The Critic).
            aggregator: The Aggregator instance (The Judge).
        """
        if not proposers:
            raise ValueError("The Council requires at least one Proposer.")

        self.proposers = proposers
        self.dissenter = dissenter
        self.aggregator = aggregator
        logger.info(f"ChamberSpeaker initialized with {len(proposers)} proposers.")

    async def resolve_query(self, query: str) -> tuple[Verdict, CouncilTrace]:
        """
        Orchestrates the resolution of a query through the Council.

        Args:
            query: The input question or problem statement.

        Returns:
            A tuple containing the final Verdict and the full CouncilTrace.
        """
        # This is a placeholder for the orchestration logic.
        # It will be implemented in subsequent atomic units.
        logger.info(f"Speaker received query: '{query}'")
        raise NotImplementedError("Orchestration logic not yet implemented.")
