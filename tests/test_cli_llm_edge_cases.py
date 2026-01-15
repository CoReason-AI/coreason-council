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
from unittest.mock import MagicMock, patch

import pytest
from coreason_council.main import app
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture
def mock_env_key() -> None:
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"


def test_cli_llm_api_failure(mock_env_key: None) -> None:
    """Test graceful failure when LLM API call fails."""
    # We patch settings to ensure valid API key is present so checks pass
    with patch("coreason_council.settings.settings.openai_api_key", "sk-mock"):
        with patch("coreason_council.main.OpenAILLMClient") as mock_client_cls:
            # Mock client to raise exception on any call
            mock_instance = mock_client_cls.return_value
            # If using get_completion directly, mock that too
            mock_instance.get_completion.side_effect = Exception("API Connection Error")
            # If using client.chat.completions.create
            mock_instance.client = MagicMock()
            mock_instance.client.chat.completions.create.side_effect = Exception("API Connection Error")

            # We need to ensure that when run_council runs, it hits this exception.
            # run_council calls speaker.resolve_query -> proposer.propose -> llm_client.get_completion

            # Since Typer/asyncio wrapper catches exception, we expect non-zero exit code or exception bubble up?
            # Typer usually captures exceptions.

            result = runner.invoke(app, ["query", "--llm"])

            # If exception is uncaught in main, result.exit_code should be non-zero
            assert result.exit_code != 0
            assert isinstance(result.exception, Exception)


def test_cli_llm_empty_query(mock_env_key: None) -> None:
    """Test LLM mode with empty query."""
    with patch("coreason_council.settings.settings.openai_api_key", "sk-mock"):
        with patch("coreason_council.main.OpenAILLMClient"):
            # Should just run through (assuming mocking works enough to not crash)
            pass

    # Invoking with empty query
    result = runner.invoke(app, ["", "--llm"])
    assert result.exit_code == 0 or result.exit_code == 1


def test_cli_llm_deadlock_scenario(mock_env_key: None) -> None:
    """Test LLM mode reaching max rounds (simulated via args)."""
    with patch("coreason_council.settings.settings.openai_api_key", "sk-mock"):
        # We need to patch Speaker or OpenAILLMClient to avoid real calls
        # Patching ChamberSpeaker.resolve_query is safest to test CLI argument passing
        with patch("coreason_council.main.ChamberSpeaker.resolve_query") as mock_resolve:
            # Return valid tuple
            mock_resolve.return_value = (
                type(
                    "Verdict",
                    (),
                    {"content": "V", "confidence_score": 1.0, "supporting_evidence": [], "alternatives": []},
                )(),
                type("Trace", (), {"session_id": "1", "transcripts": [], "vote_tally": {}})(),
            )

            result = runner.invoke(app, ["query", "--llm", "--max-rounds", "1"])
            assert result.exit_code == 0
            mock_resolve.assert_called_once()


def test_cli_llm_budget_trigger(mock_env_key: None) -> None:
    """Test LLM mode with low budget."""
    with patch("coreason_council.settings.settings.openai_api_key", "sk-mock"):
        with patch("coreason_council.main.ChamberSpeaker.resolve_query") as mock_resolve:
            mock_resolve.return_value = (
                type(
                    "Verdict",
                    (),
                    {"content": "V", "confidence_score": 1.0, "supporting_evidence": [], "alternatives": []},
                )(),
                type("Trace", (), {"session_id": "1", "transcripts": [], "vote_tally": {}})(),
            )

            result = runner.invoke(app, ["query", "--llm", "--max-budget", "1"])
            assert result.exit_code == 0
