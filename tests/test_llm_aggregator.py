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
from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.verdict import Verdict

# Original imports were: Critique, ProposerOutput, Verdict


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def llm_aggregator(mock_llm_client: MockLLMClient) -> LLMAggregator:
    return LLMAggregator(mock_llm_client)


@pytest.fixture
def sample_proposals() -> list[ProposerOutput]:
    return [
        ProposerOutput(proposer_id="p1", content="Proposal 1", confidence=0.9),
        ProposerOutput(proposer_id="p2", content="Proposal 2", confidence=0.8),
    ]


@pytest.fixture
def sample_critiques() -> list[Critique]:
    return [
        Critique(
            reviewer_id="p2",
            target_proposer_id="p1",
            content="Critique 1",
            flaws_identified=["Flaw A"],
            agreement_score=0.5,
        )
    ]


@pytest.mark.asyncio
async def test_aggregate_consensus(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
    sample_proposals: list[ProposerOutput],
    sample_critiques: list[Critique],
) -> None:
    # Setup mock return for consensus
    mock_response = VerdictContent(
        content="Final Consensus",
        confidence_score=0.95,
        supporting_evidence=["Evidence A"],
        dissenting_opinions=["Concern B"],
        alternatives=[],
    )
    mock_llm_client.return_json = mock_response

    verdict = await llm_aggregator.aggregate(sample_proposals, sample_critiques, is_deadlock=False)

    assert isinstance(verdict, Verdict)
    assert verdict.content == "Final Consensus"
    assert verdict.confidence_score == 0.95
    assert verdict.supporting_evidence == ["Evidence A"]
    assert verdict.dissenting_opinions == ["Concern B"]
    assert verdict.alternatives == []


@pytest.mark.asyncio
async def test_aggregate_deadlock(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
    sample_proposals: list[ProposerOutput],
    sample_critiques: list[Critique],
) -> None:
    # Setup mock return for deadlock
    mock_response = VerdictContent(
        content="Deadlock Summary",
        confidence_score=0.4,
        supporting_evidence=[],
        dissenting_opinions=[],
        alternatives=[
            VerdictOptionContent(label="Option A", content="Desc A", supporters=["p1"]),
            VerdictOptionContent(label="Option B", content="Desc B", supporters=["p2"]),
        ],
    )
    mock_llm_client.return_json = mock_response

    verdict = await llm_aggregator.aggregate(sample_proposals, sample_critiques, is_deadlock=True)

    assert isinstance(verdict, Verdict)
    assert verdict.content == "Deadlock Summary"
    assert verdict.confidence_score == 0.4
    assert len(verdict.alternatives) == 2
    assert verdict.alternatives[0].label == "Option A"
    assert verdict.alternatives[0].supporters == ["p1"]


@pytest.mark.asyncio
async def test_aggregate_empty_inputs(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
) -> None:
    # Setup mock return
    mock_response = VerdictContent(
        content="Empty Verdict",
        confidence_score=0.0,
        supporting_evidence=[],
        dissenting_opinions=[],
        alternatives=[],
    )
    mock_llm_client.return_json = mock_response

    verdict = await llm_aggregator.aggregate([], [])

    assert verdict.content == "Empty Verdict"
    assert verdict.confidence_score == 0.0


@pytest.mark.asyncio
async def test_aggregate_invalid_response(
    mock_llm_client: MockLLMClient,
    llm_aggregator: LLMAggregator,
    sample_proposals: list[ProposerOutput],
) -> None:
    # Mock client returns None/mismatch for raw_content when response_schema is set
    # but strictly MockLLMClient returns what we set in return_json.
    # If we set return_json to None (default), MockLLMClient logic might just return string content
    # and not populate raw_content if we don't set it.
    mock_llm_client.return_json = None

    with pytest.raises(ValueError, match="LLM failed to return structured VerdictContent"):
        await llm_aggregator.aggregate(sample_proposals, [])
