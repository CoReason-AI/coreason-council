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

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.types import Critique, ProposerOutput, Verdict


@pytest.fixture
def mock_aggregator() -> MockAggregator:
    return MockAggregator(
        default_content="Test Verdict",
        default_confidence=0.8,
        default_supporting_evidence=["Evidence 1"],
        default_dissenting_opinions=["Dissent 1"],
    )


@pytest.fixture
def sample_proposals() -> list[ProposerOutput]:
    return [
        ProposerOutput(
            proposer_id="prop-1",
            content="Proposal 1 content",
            confidence=0.9,
        ),
        ProposerOutput(
            proposer_id="prop-2",
            content="Proposal 2 content",
            confidence=0.85,
        ),
    ]


@pytest.fixture
def sample_critiques() -> list[Critique]:
    return [
        Critique(
            reviewer_id="critic-1",
            target_proposer_id="prop-1",
            content="Critique 1",
            flaws_identified=["Flaw A"],
            agreement_score=0.7,
        )
    ]


@pytest.mark.asyncio
async def test_mock_aggregator_initialization(mock_aggregator: MockAggregator) -> None:
    assert mock_aggregator.default_content == "Test Verdict"
    assert mock_aggregator.default_confidence == 0.8
    assert mock_aggregator.default_supporting_evidence == ["Evidence 1"]


@pytest.mark.asyncio
async def test_mock_aggregator_aggregate(
    mock_aggregator: MockAggregator,
    sample_proposals: list[ProposerOutput],
    sample_critiques: list[Critique],
) -> None:
    verdict = await mock_aggregator.aggregate(sample_proposals, sample_critiques)

    assert isinstance(verdict, Verdict)
    assert "Test Verdict" in verdict.content
    assert "prop-1" in verdict.content
    assert "prop-2" in verdict.content
    assert verdict.confidence_score == 0.8
    assert verdict.supporting_evidence == ["Evidence 1"]
    assert verdict.dissenting_opinions == ["Dissent 1"]


@pytest.mark.asyncio
async def test_mock_aggregator_empty_inputs(mock_aggregator: MockAggregator) -> None:
    verdict = await mock_aggregator.aggregate([], [])

    assert isinstance(verdict, Verdict)
    assert verdict.content == "Test Verdict (Based on inputs from: )"
    assert verdict.confidence_score == 0.8


@pytest.mark.asyncio
async def test_mock_aggregator_delay() -> None:
    aggregator = MockAggregator(delay_seconds=0.1)

    # We can't easily assert exact timing in a robust way, but we can ensure it completes
    # and doesn't crash.
    verdict = await aggregator.aggregate([], [])
    assert isinstance(verdict, Verdict)
