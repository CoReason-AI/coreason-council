# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import pytest

from coreason_council.core.budget import BudgetExceededError, SimpleBudgetManager


def test_calculate_cost() -> None:
    manager = SimpleBudgetManager(max_budget=100)

    # Case 1: 1 Round, 3 Proposers
    # Cost = 3
    assert manager.calculate_cost(3, 1) == 3

    # Case 2: 3 Rounds, 3 Proposers
    # Round 1: 3
    # Round 2: 3^2 = 9
    # Round 3: 3^2 = 9
    # Total = 3 + 18 = 21
    assert manager.calculate_cost(3, 3) == 21

    # Case 3: 2 Rounds, 5 Proposers
    # Round 1: 5
    # Round 2: 5^2 = 25
    # Total = 30
    assert manager.calculate_cost(5, 2) == 30


def test_check_budget_success() -> None:
    manager = SimpleBudgetManager(max_budget=50)

    # Cost is 21 <= 50
    assert manager.check_budget(3, 3) == 3


def test_check_budget_downgrade() -> None:
    manager = SimpleBudgetManager(max_budget=10)

    # Cost for 3 rounds, 3 proposers is 21 > 10
    # Downgrade to 1 round: cost 3 <= 10
    assert manager.check_budget(3, 3) == 1


def test_check_budget_failure() -> None:
    manager = SimpleBudgetManager(max_budget=2)

    # Cost for 1 round, 3 proposers is 3 > 2
    with pytest.raises(BudgetExceededError):
        manager.check_budget(3, 1)

    # Check with downgrade attempt
    # Cost for 3 rounds is 21, downgrade to 1 is 3. Still > 2.
    with pytest.raises(BudgetExceededError):
        manager.check_budget(3, 3)
