# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coreason_council.core.council_service import CouncilService
from coreason_council.core.models.verdict import Verdict


@pytest.fixture
def mock_presets_data() -> dict[str, list[dict[str, str]]]:
    return {"medical": [{"name": "A", "system_prompt": "Prompt A"}]}


@patch("coreason_council.core.council_service.yaml.safe_load")
@patch("builtins.open")
@patch("coreason_council.core.council_service.Path.exists")
def test_load_presets(
    mock_exists: MagicMock,
    mock_open: MagicMock,
    mock_yaml: MagicMock,
    mock_presets_data: dict[str, list[dict[str, str]]],
) -> None:
    mock_exists.return_value = True
    mock_yaml.return_value = mock_presets_data

    service = CouncilService()
    assert "A" in service.presets
    assert service.presets["A"].system_prompt == "Prompt A"


@patch("coreason_council.core.council_service.Path.exists")
def test_load_presets_not_found(mock_exists: MagicMock) -> None:
    mock_exists.return_value = False
    service = CouncilService()
    assert service.presets == {}


@patch("coreason_council.core.council_service.yaml.safe_load")
@patch("builtins.open")
@patch("coreason_council.core.council_service.Path.exists")
def test_load_presets_exception(
    mock_exists: MagicMock, mock_open: MagicMock, mock_yaml: MagicMock
) -> None:
    mock_exists.return_value = True
    mock_yaml.side_effect = Exception("YAML Error")
    service = CouncilService()
    assert service.presets == {}


def test_get_persona() -> None:
    service = CouncilService()
    service.presets = {"A": MagicMock()}

    # Found
    p = service.get_persona("A")
    assert p == service.presets["A"]

    # Not found
    p = service.get_persona("B")
    assert p.name == "B"


@pytest.mark.asyncio
@patch("coreason_council.core.council_service.GatewayLLMClient")
@patch("coreason_council.core.council_service.LLMProposer")
@patch("coreason_council.core.council_service.LLMAggregator")
async def test_convene_session(
    MockAggregator: MagicMock, MockProposer: MagicMock, MockClient: MagicMock
) -> None:
    service = CouncilService()
    service.presets = {}

    mock_proposer_instance = MockProposer.return_value
    mock_proposer_instance.propose = AsyncMock()
    mock_proposer_instance.propose.return_value = MagicMock(
        content="Vote", confidence=0.8, proposer_id="p1"
    )

    mock_aggregator_instance = MockAggregator.return_value
    mock_aggregator_instance.aggregate = AsyncMock()
    mock_aggregator_instance.aggregate.return_value = Verdict(
        content="Verdict",
        confidence_score=0.9,
        supporting_evidence=[],
        dissenting_opinions=["Dissent"],
        alternatives=[],
    )

    result = await service.convene_session("Topic", ["A"], "gpt-4o")

    assert result["verdict"] == "Verdict"
    assert result["dissent"] == "Dissent"
    assert len(result["votes"]) == 1
    assert result["votes"][0]["content"] == "Vote"
