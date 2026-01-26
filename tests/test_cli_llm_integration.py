# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import os
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from coreason_council.main import app

runner = CliRunner()


def test_cli_llm_wiring() -> None:
    """Test that --llm flag correctly instantiates LLM components."""
    with patch("coreason_council.main.GatewayLLMClient") as MockClient:
        with patch("coreason_council.main.LLMProposer"):
            with patch("coreason_council.main.LLMAggregator") as MockAggregator:
                with patch("coreason_council.main.ChamberSpeaker") as MockSpeaker:
                    # Setup mocks to allow execution to proceed
                    mock_speaker_instance = MockSpeaker.return_value

                    # Make resolve_query awaitable
                    async def mock_resolve(*args: Any, **kwargs: Any) -> Any:
                        return (
                            type(
                                "Verdict",
                                (),
                                {
                                    "content": "V",
                                    "confidence_score": 1.0,
                                    "supporting_evidence": [],
                                    "alternatives": [],
                                },
                            )(),
                            type(
                                "Trace",
                                (),
                                {"session_id": "1", "transcripts": [], "vote_tally": {}},
                            )(),
                        )

                    mock_speaker_instance.resolve_query.side_effect = mock_resolve

                    result = runner.invoke(app, ["query", "--llm"])

                    assert result.exit_code == 0
                    MockClient.assert_called()
                    # LLMProposer should be used (via factory), but checking call count is hard here
                    MockAggregator.assert_called()


def test_cli_default_wiring() -> None:
    """Test default wiring (Mock mode)."""
    with patch("coreason_council.main.MockAggregator") as MockAggregator:
        with patch("coreason_council.main.ChamberSpeaker") as MockSpeaker:
            # Setup mocks
            mock_speaker_instance = MockSpeaker.return_value

            async def mock_resolve(*args: Any, **kwargs: Any) -> Any:
                return (
                    type(
                        "Verdict",
                        (),
                        {
                            "content": "V",
                            "confidence_score": 1.0,
                            "supporting_evidence": [],
                            "alternatives": [],
                        },
                    )(),
                    type(
                        "Trace",
                        (),
                        {"session_id": "1", "transcripts": [], "vote_tally": {}},
                    )(),
                )

            mock_speaker_instance.resolve_query.side_effect = mock_resolve

            result = runner.invoke(app, ["query"])

            assert result.exit_code == 0
            MockAggregator.assert_called()
