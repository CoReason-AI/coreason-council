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
