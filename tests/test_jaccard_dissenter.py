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

from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.types import Persona, ProposerOutput


@pytest.fixture
def jaccard_dissenter() -> JaccardDissenter:
    return JaccardDissenter()


@pytest.fixture
def basic_persona() -> Persona:
    return Persona(name="Tester", system_prompt="Test Prompt")


def create_output(content: str, pid: str = "p1") -> ProposerOutput:
    return ProposerOutput(proposer_id=pid, content=content, confidence=1.0)


@pytest.mark.asyncio
async def test_tokenize_logic(jaccard_dissenter: JaccardDissenter) -> None:
    text = "Hello, World! This is a test."
    tokens = jaccard_dissenter._tokenize(text)
    assert tokens == {"hello", "world", "this", "is", "a", "test"}


@pytest.mark.asyncio
async def test_tokenize_empty(jaccard_dissenter: JaccardDissenter) -> None:
    assert jaccard_dissenter._tokenize("") == set()
    # Ignoring type check for None input if strict type checking is on,
    # but the implementation handles it if passed dynamically.
    # For test purposes we cast or ignore if mypy complains about passing None to str argument.
    # However, implementation expects str. If we want to test robustness:
    # assert jaccard_dissenter._tokenize(None) == set()  # type: ignore


@pytest.mark.asyncio
async def test_jaccard_similarity_identical(jaccard_dissenter: JaccardDissenter) -> None:
    t1 = "The quick brown fox"
    t2 = "The quick brown fox"
    assert jaccard_dissenter._calculate_jaccard_similarity(t1, t2) == 1.0


@pytest.mark.asyncio
async def test_jaccard_similarity_disjoint(jaccard_dissenter: JaccardDissenter) -> None:
    t1 = "apple banana"
    t2 = "cherry date"
    assert jaccard_dissenter._calculate_jaccard_similarity(t1, t2) == 0.0


@pytest.mark.asyncio
async def test_jaccard_similarity_partial(jaccard_dissenter: JaccardDissenter) -> None:
    t1 = "apple banana cherry"
    t2 = "banana cherry date"
    # Intersection: banana, cherry (2)
    # Union: apple, banana, cherry, date (4)
    # Score: 0.5
    assert jaccard_dissenter._calculate_jaccard_similarity(t1, t2) == 0.5


@pytest.mark.asyncio
async def test_jaccard_similarity_one_empty(jaccard_dissenter: JaccardDissenter) -> None:
    # This hits the line where one is empty and the other is not -> returns 0.0
    t1 = "some content"
    t2 = ""
    assert jaccard_dissenter._calculate_jaccard_similarity(t1, t2) == 0.0
    assert jaccard_dissenter._calculate_jaccard_similarity(t2, t1) == 0.0


@pytest.mark.asyncio
async def test_entropy_identical_proposals(jaccard_dissenter: JaccardDissenter) -> None:
    p1 = create_output("consensus answer", "p1")
    p2 = create_output("consensus answer", "p2")
    p3 = create_output("consensus answer", "p3")

    entropy = await jaccard_dissenter.calculate_entropy([p1, p2, p3])
    assert entropy == 0.0


@pytest.mark.asyncio
async def test_entropy_disjoint_proposals(jaccard_dissenter: JaccardDissenter) -> None:
    p1 = create_output("A", "p1")
    p2 = create_output("B", "p2")

    # Sim: 0.0 -> Entropy: 1.0
    entropy = await jaccard_dissenter.calculate_entropy([p1, p2])
    assert entropy == 1.0


@pytest.mark.asyncio
async def test_entropy_mixed_proposals(jaccard_dissenter: JaccardDissenter) -> None:
    p1 = create_output("A B", "p1")
    p2 = create_output("B C", "p2")
    p3 = create_output("C A", "p3")

    # Pairs:
    # p1-p2: {B} / {A,B,C} = 1/3
    # p2-p3: {C} / {A,B,C} = 1/3
    # p1-p3: {A} / {A,B,C} = 1/3
    # Avg Sim = 1/3
    # Entropy = 1 - 1/3 = 2/3 = 0.666...

    entropy = await jaccard_dissenter.calculate_entropy([p1, p2, p3])
    assert abs(entropy - (2 / 3)) < 0.001


@pytest.mark.asyncio
async def test_entropy_single_proposal(jaccard_dissenter: JaccardDissenter) -> None:
    p1 = create_output("solo", "p1")
    assert await jaccard_dissenter.calculate_entropy([p1]) == 0.0


@pytest.mark.asyncio
async def test_entropy_empty_proposals(jaccard_dissenter: JaccardDissenter) -> None:
    assert await jaccard_dissenter.calculate_entropy([]) == 0.0


@pytest.mark.asyncio
async def test_critique_passthrough(jaccard_dissenter: JaccardDissenter, basic_persona: Persona) -> None:
    p1 = create_output("test content", "p1")
    critique = await jaccard_dissenter.critique(p1, basic_persona)

    assert critique.reviewer_id == "Tester"
    assert critique.target_proposer_id == "p1"
    assert "2 unique tokens" in critique.content
