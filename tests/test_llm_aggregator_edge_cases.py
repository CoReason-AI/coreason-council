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

from coreason_council.core.llm_aggregator import LLMAggregator, VerdictContent, VerdictOptionContent
from coreason_council.core.llm_client import MockLLMClient
from coreason_council.core.types import Critique, ProposerOutput, Verdict


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def llm_aggregator(mock_llm_client: MockLLMClient) -> LLMAggregator:
    return LLMAggregator(mock_llm_client)


@pytest.mark.asyncio
async def test_deadlock_with_no_alternatives_returned(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    """
    Edge Case: The system requests a Minority Report (Deadlock), but the LLM
    fails to follow instructions and returns an empty alternatives list.
    The Aggregator should faithfully return the empty list in the Verdict.
    """
    mock_response = VerdictContent(
        content="Failed to split options.",
        confidence_score=0.1,
        supporting_evidence=[],
        dissenting_opinions=[],
        alternatives=[],  # Empty despite deadlock
    )
    mock_llm_client.return_json = mock_response

    verdict = await llm_aggregator.aggregate([], [], is_deadlock=True)

    assert isinstance(verdict, Verdict)
    assert verdict.content == "Failed to split options."
    assert verdict.alternatives == []
    # This behavior allows the caller/speaker to handle the 'empty alternatives deadlock' case
    # if necessary, rather than the aggregator crashing or inventing data.


@pytest.mark.asyncio
async def test_consensus_with_hallucinated_alternatives(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    """
    Edge Case: The system requests a Consensus (No Deadlock), but the LLM
    hallucinates and provides 'alternatives' anyway.
    The Aggregator currently maps them. This test validates that behavior ensures
    Glass Box visibility (we see what the LLM outputted).
    """
    mock_response = VerdictContent(
        content="Consensus Reached",
        confidence_score=0.9,
        supporting_evidence=[],
        dissenting_opinions=[],
        alternatives=[
            VerdictOptionContent(label="Hallucination A", content="A", supporters=["ghost"]),
        ],
    )
    mock_llm_client.return_json = mock_response

    verdict = await llm_aggregator.aggregate([], [], is_deadlock=False)

    assert isinstance(verdict, Verdict)
    assert verdict.content == "Consensus Reached"
    assert len(verdict.alternatives) == 1
    assert verdict.alternatives[0].label == "Hallucination A"


@pytest.mark.asyncio
async def test_large_input_handling(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    """
    Complex Scenario: Handling very large proposal texts.
    Verifies that prompt construction doesn't crash on large strings.
    """
    large_content = "Word " * 10000  # Large string
    proposals = [
        ProposerOutput(proposer_id="p1", content=large_content, confidence=0.9),
    ]

    # Mock a valid return
    mock_llm_client.return_json = VerdictContent(content="Summary", confidence_score=0.9, alternatives=[])

    verdict = await llm_aggregator.aggregate(proposals, [])

    assert verdict.content == "Summary"
    # Implicitly passes if no MemoryError or RecursionError occurs during f-string construction


@pytest.mark.asyncio
async def test_special_characters_in_prompt_construction(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    """
    Edge Case: Inputs contain special characters, emojis, and newlines.
    """
    proposals = [ProposerOutput(proposer_id="p1", content="Line 1\nLine 2\tTabbed\nEmoji 🚀", confidence=0.5)]
    critiques = [
        Critique(
            reviewer_id="crit-ø",
            target_proposer_id="p1",
            content="Critique with «quotes» and \n newlines.",
            flaws_identified=["Flaw #1"],
            agreement_score=0.2,
        )
    ]

    mock_llm_client.return_json = VerdictContent(content="Handled 🚀", confidence_score=0.8, alternatives=[])

    verdict = await llm_aggregator.aggregate(proposals, critiques)
    assert verdict.content == "Handled 🚀"


@pytest.mark.asyncio
async def test_extreme_confidence_scores(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    """
    Edge Case: Proposals with 0.0 or 1.0 confidence.
    """
    proposals = [
        ProposerOutput(proposer_id="p1", content="Zero", confidence=0.0),
        ProposerOutput(proposer_id="p2", content="One", confidence=1.0),
    ]

    mock_llm_client.return_json = VerdictContent(content="Result", confidence_score=0.5, alternatives=[])

    # Just ensuring validation passes and prompt construction works
    verdict = await llm_aggregator.aggregate(proposals, [])
    assert verdict.confidence_score == 0.5
