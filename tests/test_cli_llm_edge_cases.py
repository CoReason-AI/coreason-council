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
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from openai import APIError

from coreason_council.core.types import CouncilTrace, Verdict, VerdictOption
from coreason_council.main import run_council


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def mock_env_key() -> Generator[None, None, None]:
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-key"}):
        yield


def test_cli_llm_api_failure(cli_runner: CliRunner, mock_env_key: None) -> None:
    """
    Test Case: API Failure Handling
    Verifies that unhandled API errors propagate and crash the CLI (or are handled if we implemented catching).
    Currently, we expect the CLI to exit with code 1 due to the unhandled exception.
    """
    with (
        patch("coreason_council.main.OpenAILLMClient"),
        patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
    ):
        # Simulate OpenAILLMClient being instantiated, but Speaker failing
        speaker_instance = MockSpeaker.return_value
        speaker_instance.resolve_query.side_effect = APIError(
            message="Service Unavailable", request=MagicMock(), body={}
        )

        result = cli_runner.invoke(run_council, ["test query", "--llm"])

        assert result.exit_code != 0
        assert isinstance(result.exception, APIError)


def test_cli_llm_empty_query(cli_runner: CliRunner, mock_env_key: None) -> None:
    """
    Test Case: Empty Query
    Verifies behavior when query is empty. The CLI logic should still attempt to proceed,
    likely defaulting to General panel, unless Click prevents empty args (it doesn't by default).
    """
    with (
        patch("coreason_council.main.OpenAILLMClient"),
        patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
        patch("coreason_council.main.JaccardDissenter"),
    ):
        mock_trace = MagicMock(spec=CouncilTrace)
        mock_trace.transcripts = []
        mock_trace.vote_tally = {"Consensus": 3}
        mock_trace.session_id = "test-session"

        mock_verdict = MagicMock(spec=Verdict)
        mock_verdict.content = "Empty response"
        mock_verdict.confidence_score = 0.0
        mock_verdict.supporting_evidence = []
        mock_verdict.alternatives = []

        speaker_instance = MockSpeaker.return_value
        speaker_instance.resolve_query = AsyncMock(return_value=(mock_verdict, mock_trace))

        result = cli_runner.invoke(run_council, ["", "--llm"])

        assert result.exit_code == 0
        assert "Content: Empty response" in result.output

        # Verify empty string was passed
        speaker_instance.resolve_query.assert_called_once_with("")


def test_cli_llm_deadlock_scenario(cli_runner: CliRunner, mock_env_key: None) -> None:
    """
    Test Case: Deadlock Scenario in LLM Mode
    Verifies that if resolve_query returns a Verdict with alternatives (deadlock),
    the CLI correctly prints the Alternatives section.
    """
    with (
        patch("coreason_council.main.OpenAILLMClient"),
        patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
        patch("coreason_council.main.JaccardDissenter"),
    ):
        mock_trace = MagicMock(spec=CouncilTrace)
        mock_trace.transcripts = []
        mock_trace.vote_tally = {"Option A": 1, "Option B": 1}
        mock_trace.session_id = "deadlock-session"

        mock_verdict = MagicMock(spec=Verdict)
        mock_verdict.content = "No consensus"
        mock_verdict.confidence_score = 0.1
        mock_verdict.supporting_evidence = []
        mock_verdict.alternatives = [
            VerdictOption(label="Option A", content="Use X", supporters=["gpt-4"]),
            VerdictOption(label="Option B", content="Use Y", supporters=["claude-3"]),
        ]

        speaker_instance = MockSpeaker.return_value
        speaker_instance.resolve_query = AsyncMock(return_value=(mock_verdict, mock_trace))

        result = cli_runner.invoke(run_council, ["tough question", "--llm"])

        assert result.exit_code == 0
        assert "--- ALTERNATIVES (Deadlock) ---" in result.output
        assert "Option: Option A - Supported by 1 proposers" in result.output
        assert "Option: Option B - Supported by 1 proposers" in result.output


def test_cli_llm_budget_trigger(cli_runner: CliRunner, mock_env_key: None) -> None:
    """
    Test Case: Budget Constraint
    Verifies that the BudgetManager is initialized and passed to Speaker.
    We can verify this by inspecting the call args to ChamberSpeaker.
    """
    with (
        patch("coreason_council.main.OpenAILLMClient"),
        patch("coreason_council.main.ChamberSpeaker") as MockSpeaker,
        patch("coreason_council.main.JaccardDissenter"),
        patch("coreason_council.main.SimpleBudgetManager") as MockBudget,
    ):
        # Setup mocks to allow successful run
        speaker_instance = MockSpeaker.return_value
        speaker_instance.resolve_query = AsyncMock(return_value=(MagicMock(alternatives=[]), MagicMock()))

        result = cli_runner.invoke(run_council, ["query", "--llm", "--max-budget", "50"])

        assert result.exit_code == 0

        # Verify BudgetManager initialized with correct max_budget
        MockBudget.assert_called_once_with(max_budget=50)

        # Verify Speaker received the budget manager
        _, kwargs = MockSpeaker.call_args
        assert kwargs["budget_manager"] == MockBudget.return_value
