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
from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.verdict import Verdict

# Original imports were: Critique, ProposerOutput, Verdict


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
    # New check for critique inclusion
    assert "critic-1" in verdict.content
    assert verdict.confidence_score == 0.8
    assert verdict.supporting_evidence == ["Evidence 1"]
    assert verdict.dissenting_opinions == ["Dissent 1"]


@pytest.mark.asyncio
async def test_mock_aggregator_empty_inputs(mock_aggregator: MockAggregator) -> None:
    verdict = await mock_aggregator.aggregate([], [])

    assert isinstance(verdict, Verdict)
    # Check proper formatting for empty lists
    assert "Based on inputs from: ; Critiqued by: )" in verdict.content
    assert verdict.confidence_score == 0.8


@pytest.mark.asyncio
async def test_mock_aggregator_delay() -> None:
    aggregator = MockAggregator(delay_seconds=0.1)

    # We can't easily assert exact timing in a robust way, but we can ensure it completes
    # and doesn't crash.
    verdict = await aggregator.aggregate([], [])
    assert isinstance(verdict, Verdict)


@pytest.mark.asyncio
async def test_aggregator_large_input_volume(mock_aggregator: MockAggregator) -> None:
    """Test aggregation with a large number of proposals and critiques."""
    proposals = [ProposerOutput(proposer_id=f"prop-{i}", content=f"Content {i}", confidence=0.9) for i in range(100)]
    critiques = [
        Critique(
            reviewer_id=f"critic-{i}",
            target_proposer_id=f"prop-{i}",
            content=f"Critique {i}",
            flaws_identified=[],
            agreement_score=0.5,
        )
        for i in range(50)
    ]

    verdict = await mock_aggregator.aggregate(proposals, critiques)

    assert isinstance(verdict, Verdict)
    assert "prop-0" in verdict.content
    assert "prop-99" in verdict.content
    assert "critic-0" in verdict.content
    assert "critic-49" in verdict.content


@pytest.mark.asyncio
async def test_aggregator_special_characters(mock_aggregator: MockAggregator) -> None:
    """Test handling of IDs with special characters."""
    special_proposals = [
        ProposerOutput(
            proposer_id="prop-@#$%^&*",
            content="Special Chars",
            confidence=0.9,
        ),
        ProposerOutput(
            proposer_id="prop-🚀-emoji",
            content="Emoji",
            confidence=0.9,
        ),
    ]

    special_critiques = [
        Critique(
            reviewer_id="critic-über-cool",
            target_proposer_id="prop-@#$%^&*",
            content="Unicode",
            flaws_identified=[],
            agreement_score=0.5,
        )
    ]

    verdict = await mock_aggregator.aggregate(special_proposals, special_critiques)

    assert "prop-@#$%^&*" in verdict.content
    assert "prop-🚀-emoji" in verdict.content
    assert "critic-über-cool" in verdict.content


@pytest.mark.asyncio
async def test_aggregator_complex_metadata(mock_aggregator: MockAggregator) -> None:
    """Test proposals with complex, deeply nested metadata."""
    complex_metadata = {
        "source": "database",
        "nested": {"level1": {"level2": [1, 2, 3]}},
        "none_value": None,
        "bool_value": True,
    }

    proposals = [
        ProposerOutput(
            proposer_id="prop-complex",
            content="Content",
            confidence=0.9,
            metadata=complex_metadata,
        )
    ]

    verdict = await mock_aggregator.aggregate(proposals, [])

    # The aggregator doesn't use metadata in output, but we ensure it doesn't crash
    assert "prop-complex" in verdict.content
    assert verdict.confidence_score == 0.8
