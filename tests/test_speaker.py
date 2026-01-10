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
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker


def test_speaker_initialization() -> None:
    """Test that ChamberSpeaker initializes correctly with valid components."""
    proposers = [MockProposer(), MockProposer()]
    dissenter = MockDissenter()
    aggregator = MockAggregator()

    speaker = ChamberSpeaker(
        proposers=proposers,
        dissenter=dissenter,
        aggregator=aggregator,
    )

    assert speaker.proposers == proposers
    assert speaker.dissenter == dissenter
    assert speaker.aggregator == aggregator


def test_speaker_initialization_empty_proposers() -> None:
    """Test that ChamberSpeaker raises ValueError if no proposers are provided."""
    dissenter = MockDissenter()
    aggregator = MockAggregator()

    with pytest.raises(ValueError, match="The Council requires at least one Proposer."):
        ChamberSpeaker(
            proposers=[],
            dissenter=dissenter,
            aggregator=aggregator,
        )


def test_speaker_initialization_none_args() -> None:
    """Test that ChamberSpeaker raises ValueError if critical components are None."""
    proposers = [MockProposer()]
    dissenter = MockDissenter()
    aggregator = MockAggregator()

    # Test None dissenter - use type ignore because we are testing runtime safety against bad callers
    with pytest.raises(ValueError, match="The Council requires a Dissenter."):
        ChamberSpeaker(
            proposers=proposers,
            dissenter=None,  # type: ignore
            aggregator=aggregator,
        )

    # Test None aggregator
    with pytest.raises(ValueError, match="The Council requires an Aggregator."):
        ChamberSpeaker(
            proposers=proposers,
            dissenter=dissenter,
            aggregator=None,  # type: ignore
        )


def test_speaker_proposers_immutability() -> None:
    """Test that modifying the input list of proposers does not affect the Speaker."""
    p1 = MockProposer()
    p2 = MockProposer()
    proposers_list = [p1]

    speaker = ChamberSpeaker(
        proposers=proposers_list,
        dissenter=MockDissenter(),
        aggregator=MockAggregator(),
    )

    # Modify the external list
    proposers_list.append(p2)

    # Speaker should still only have p1
    assert len(speaker.proposers) == 1
    assert speaker.proposers[0] == p1
    assert p2 not in speaker.proposers


def test_speaker_large_council_configuration() -> None:
    """Test initialization with a large number of proposers (scalability/stress test)."""
    # Create 50 distinct proposers
    proposers = [MockProposer(proposer_id_prefix=f"worker-{i}") for i in range(50)]
    dissenter = MockDissenter()
    aggregator = MockAggregator()

    speaker = ChamberSpeaker(
        proposers=proposers,
        dissenter=dissenter,
        aggregator=aggregator,
    )

    assert len(speaker.proposers) == 50
    # Verify strict object identity to ensure order and content are preserved
    assert speaker.proposers[0] is proposers[0]
    assert speaker.proposers[49] is proposers[49]


@pytest.mark.asyncio
async def test_resolve_query_not_implemented() -> None:
    """Test that resolve_query currently raises NotImplementedError."""
    speaker = ChamberSpeaker(
        proposers=[MockProposer()],
        dissenter=MockDissenter(),
        aggregator=MockAggregator(),
    )

    with pytest.raises(NotImplementedError):
        await speaker.resolve_query("Test query")
