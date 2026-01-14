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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.llm_aggregator import LLMAggregator
from coreason_council.core.llm_proposer import LLMProposer
from coreason_council.core.proposer import MockProposer
from coreason_council.main import run_council


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_llm_missing_api_key(cli_runner: CliRunner) -> None:
    """Test that --llm fails if OPENAI_API_KEY is not set."""
    # Ensure env var is not set
    with patch.dict(os.environ, {}, clear=True):
        result = cli_runner.invoke(run_council, ["test query", "--llm"])
        assert result.exit_code != 0
        # Click usually prints exception messages to output
        assert "OPENAI_API_KEY environment variable is required" in result.output


def test_cli_llm_wiring(cli_runner: CliRunner) -> None:
    """Test that --llm correctly wires LLM components."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
        with (
            patch("coreason_council.main.OpenAILLMClient") as MockClientCls,
            patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
            patch("coreason_council.main.JaccardDissenter"),
        ):
            # Mock the trace return from resolve_query
            mock_trace = MagicMock()
            mock_trace.transcripts = []
            mock_trace.vote_tally = {}
            mock_trace.session_id = "test-session"

            mock_verdict = MagicMock()
            mock_verdict.content = "Test Verdict"
            mock_verdict.confidence_score = 0.9
            mock_verdict.alternatives = []

            # Setup async mock return properly
            # ChamberSpeaker instance is returned by constructor
            speaker_instance = MockSpeaker.return_value
            # resolve_query is async
            speaker_instance.resolve_query = AsyncMock(return_value=(mock_verdict, mock_trace))

            result = cli_runner.invoke(run_council, ["test query", "--llm"])

            # If exit_code != 0, print output for debugging
            if result.exit_code != 0:
                print(result.output)
                print(result.exception)

            assert result.exit_code == 0

            # Verify OpenAILLMClient was instantiated
            MockClientCls.assert_called_once()

            # Verify ChamberSpeaker init arguments
            call_args = MockSpeaker.call_args
            assert call_args is not None
            _, kwargs = call_args

            # Check Aggregator
            assert isinstance(kwargs["aggregator"], LLMAggregator)

            # Check Proposers
            proposers = kwargs["proposers"]
            assert len(proposers) > 0
            assert all(isinstance(p, LLMProposer) for p in proposers)


def test_cli_default_wiring(cli_runner: CliRunner) -> None:
    """Test that default (no --llm) uses Mock components."""
    with (
        patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
        patch("coreason_council.main.JaccardDissenter"),
    ):
        # Mock the trace return
        mock_trace = MagicMock()
        mock_trace.transcripts = []
        mock_trace.vote_tally = {}
        mock_trace.session_id = "test-session"

        mock_verdict = MagicMock()
        mock_verdict.content = "Test Verdict"
        mock_verdict.confidence_score = 0.9
        mock_verdict.alternatives = []

        # Setup async mock return properly
        speaker_instance = MockSpeaker.return_value
        speaker_instance.resolve_query = AsyncMock(return_value=(mock_verdict, mock_trace))

        result = cli_runner.invoke(run_council, ["test query"])

        if result.exit_code != 0:
            print(result.output)
            print(result.exception)

        assert result.exit_code == 0

        # Verify ChamberSpeaker init arguments
        call_args = MockSpeaker.call_args
        assert call_args is not None
        _, kwargs = call_args

        # Check Aggregator
        assert isinstance(kwargs["aggregator"], MockAggregator)

        # Check Proposers
        proposers = kwargs["proposers"]
        assert len(proposers) > 0
        assert all(isinstance(p, MockProposer) for p in proposers)
