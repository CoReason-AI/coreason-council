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


def test_zero_budget() -> None:
    """Test that a zero budget rejects even the minimal configuration."""
    manager = SimpleBudgetManager(max_budget=0)
    # Minimal cost: 1 proposer, 1 round = 1 operation.
    with pytest.raises(BudgetExceededError):
        manager.check_budget(1, 1)


def test_negative_budget() -> None:
    """Test that a negative budget is handled correctly (rejects all)."""
    manager = SimpleBudgetManager(max_budget=-10)
    with pytest.raises(BudgetExceededError):
        manager.check_budget(1, 1)


def test_exact_budget_match() -> None:
    """Test boundary condition where cost exactly equals budget."""
    # N=2, R=2.
    # Cost = N + (R-1)*N^2 = 2 + (1)*4 = 6.
    manager = SimpleBudgetManager(max_budget=6)
    # Should accept R=2
    assert manager.check_budget(2, 2) == 2


def test_budget_one_short() -> None:
    """Test boundary condition where cost is 1 greater than budget."""
    # Cost = 6. Budget = 5.
    manager = SimpleBudgetManager(max_budget=5)
    # Should downgrade to R=1 (Cost = 2)
    assert manager.check_budget(2, 2) == 1


def test_large_scale_inputs() -> None:
    """Test calculation with larger inputs to ensure no overflow/logic errors."""
    manager = SimpleBudgetManager(max_budget=1_000_000)
    # N=50, R=5.
    # Cost = 50 + 4 * 2500 = 50 + 10,000 = 10,050.
    assert manager.check_budget(50, 5) == 5

    # N=1000, R=2.
    # Cost = 1000 + 1 * 1,000,000 = 1,001,000.
    # Budget is 1,000,000. So it fails R=2.
    # Downgrades to R=1 (Cost=1000).
    assert manager.check_budget(1000, 2) == 1


def test_single_round_always_safe_if_budget_allows() -> None:
    """If max_rounds=1 is requested, it should pass if budget covers N."""
    manager = SimpleBudgetManager(max_budget=10)
    # N=10, R=1 -> Cost=10. Fits.
    assert manager.check_budget(10, 1) == 1
