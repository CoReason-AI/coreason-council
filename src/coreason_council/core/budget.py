# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from abc import ABC, abstractmethod


class BudgetExceededError(Exception):
    """Raised when the cost estimate exceeds the available budget even after downgrading."""

    pass


class BaseBudgetManager(ABC):
    """
    Abstract base class for the Budget Manager (The Economist Hook).
    Responsible for estimating costs and enforcing budget constraints.
    """

    @abstractmethod
    def check_budget(self, n_proposers: int, max_rounds: int) -> int:
        """
        Estimates the cost and checks if it fits within the budget.
        If the budget is exceeded, it may suggest a downgraded configuration.

        Args:
            n_proposers: Number of active proposers (N).
            max_rounds: Requested maximum number of rounds (M).

        Returns:
            The approved max_rounds (which may be lower than requested).

        Raises:
            BudgetExceededError: If even the minimal configuration exceeds the budget.
        """
        pass  # pragma: no cover


class SimpleBudgetManager(BaseBudgetManager):
    """
    A simple implementation that counts 'operations' as a proxy for cost.
    Operation Cost Model:
    - Round 1 (Propose): N operations.
    - Round k > 1 (Debate): N^2 operations (N critiques N-1 peers + N revisions).
    Total Cost = N + (max_rounds - 1) * N^2
    """

    def __init__(self, max_budget: int = 100) -> None:
        self.max_budget = max_budget

    def calculate_cost(self, n_proposers: int, max_rounds: int) -> int:
        """Calculates the theoretical cost in operations."""
        if max_rounds <= 0:
            return 0

        # Round 1 cost
        cost = n_proposers

        if max_rounds > 1:
            # Debate rounds cost: (M-1) * (Critiques + Revisions)
            # Critiques = N * (N - 1)
            # Revisions = N
            # Total per round = N^2 - N + N = N^2
            debate_rounds = max_rounds - 1
            cost += debate_rounds * (n_proposers**2)

        return cost

    def check_budget(self, n_proposers: int, max_rounds: int) -> int:
        # Check requested configuration
        estimated_cost = self.calculate_cost(n_proposers, max_rounds)

        if estimated_cost <= self.max_budget:
            return max_rounds

        # Budget exceeded, try downgrading to single round (Simple Majority Vote)
        downgraded_rounds = 1
        min_cost = self.calculate_cost(n_proposers, downgraded_rounds)

        if min_cost <= self.max_budget:
            return downgraded_rounds

        # Even minimal configuration fails
        raise BudgetExceededError(f"Budget exceeded. Available: {self.max_budget}, Required (Min): {min_cost}")
