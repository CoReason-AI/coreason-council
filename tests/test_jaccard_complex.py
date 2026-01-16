# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import time

import pytest

from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.models.interaction import ProposerOutput

# Original imports were: ProposerOutput


@pytest.fixture
def jaccard_dissenter() -> JaccardDissenter:
    return JaccardDissenter()


def create_output(content: str, pid: str) -> ProposerOutput:
    return ProposerOutput(proposer_id=pid, content=content, confidence=1.0)


@pytest.mark.asyncio
async def test_edge_case_symbolic_inputs(jaccard_dissenter: JaccardDissenter) -> None:
    """
    Current implementation uses \\w+ regex, which ignores punctuation and symbols.
    Therefore, '!!!' and '???' result in empty token sets.
    Empty set vs Empty set = Jaccard 1.0 (Identical).
    Entropy = 0.0.
    """
    p1 = create_output("!!!", "p1")
    p2 = create_output("???", "p2")

    # Verify tokenization first
    assert jaccard_dissenter._tokenize("!!!") == set()

    # Verify entropy
    entropy = await jaccard_dissenter.calculate_entropy([p1, p2])
    assert entropy == 0.0


@pytest.mark.asyncio
async def test_edge_case_formatting_normalization(jaccard_dissenter: JaccardDissenter) -> None:
    """
    Verify that casing, whitespace, and punctuation are normalized.
    """
    p1 = create_output("Hello World", "p1")
    p2 = create_output("  hello   world!  ", "p2")
    p3 = create_output("HELLO\nWORLD.", "p3")

    # All should be treated as {"hello", "world"}
    # Pairwise sim should be 1.0 for all.
    # Entropy should be 0.0.

    entropy = await jaccard_dissenter.calculate_entropy([p1, p2, p3])
    assert entropy == 0.0


@pytest.mark.asyncio
async def test_complex_bipolar_consensus(jaccard_dissenter: JaccardDissenter) -> None:
    """
    Scenario: 2 Proposers say 'Yes', 2 Proposers say 'No'.
    Group A: [p1, p2] (identical)
    Group B: [p3, p4] (identical)
    Group A vs Group B: Disjoint.

    Pairs:
    (p1,p2): Sim 1.0
    (p3,p4): Sim 1.0
    (p1,p3), (p1,p4), (p2,p3), (p2,p4): Sim 0.0 (4 pairs)

    Sum Sim = 2.0
    Total Pairs = 4C2 = 6
    Avg Sim = 2/6 = 0.333...
    Entropy = 1 - 0.333... = 0.666...
    """
    p1 = create_output("Yes", "p1")
    p2 = create_output("Yes", "p2")
    p3 = create_output("No", "p3")
    p4 = create_output("No", "p4")

    entropy = await jaccard_dissenter.calculate_entropy([p1, p2, p3, p4])
    expected_sim = 2 / 6
    expected_entropy = 1.0 - expected_sim

    assert abs(entropy - expected_entropy) < 0.001


@pytest.mark.asyncio
async def test_complex_chain_scenario(jaccard_dissenter: JaccardDissenter) -> None:
    """
    Scenario: 'Chain' of overlapping ideas.
    p1: "A"
    p2: "A B"
    p3: "B C"
    p4: "C"

    Pairs & Similarities:
    1. (p1, p2) "A" vs "A B": Int=1, Union=2 -> 0.5
    2. (p1, p3) "A" vs "B C": Int=0 -> 0.0
    3. (p1, p4) "A" vs "C": Int=0 -> 0.0
    4. (p2, p3) "A B" vs "B C": Int=1 ("B"), Union=3 ("A,B,C") -> 1/3
    5. (p2, p4) "A B" vs "C": Int=0 -> 0.0
    6. (p3, p4) "B C" vs "C": Int=1, Union=2 -> 0.5

    Sum Sim = 0.5 + 0 + 0 + 0.333 + 0 + 0.5 = 1.333...
    Avg Sim = 1.333 / 6 = 0.222...
    Entropy = 1 - 0.222... = 0.777...
    """
    p1 = create_output("apple", "p1")
    p2 = create_output("apple banana", "p2")
    p3 = create_output("banana cherry", "p3")
    p4 = create_output("cherry", "p4")

    entropy = await jaccard_dissenter.calculate_entropy([p1, p2, p3, p4])

    sum_sim = 0.5 + 0.0 + 0.0 + (1 / 3) + 0.0 + 0.5
    expected_entropy = 1.0 - (sum_sim / 6)

    assert abs(entropy - expected_entropy) < 0.001


@pytest.mark.asyncio
async def test_performance_large_inputs(jaccard_dissenter: JaccardDissenter) -> None:
    """
    Sanity check for larger inputs.
    We generate 1000 UNIQUE words to ensure the set size is large.
    """
    # Create 1000 unique words
    unique_words = [f"word{i}" for i in range(1000)]
    base_text = " ".join(unique_words)

    p1 = create_output(base_text + " uniqueA", "p1")
    p2 = create_output(base_text + " uniqueB", "p2")

    # Set 1 size: 1001
    # Set 2 size: 1001
    # Intersection: 1000
    # Union: 1002
    # Sim: 1000/1002 ~ 0.998

    start_time = time.time()
    entropy = await jaccard_dissenter.calculate_entropy([p1, p2])
    duration = time.time() - start_time

    # Should be very fast (< 0.1s)
    assert duration < 1.0

    # Entropy should be very low
    # 1 - 0.998 = 0.002
    assert entropy < 0.01
