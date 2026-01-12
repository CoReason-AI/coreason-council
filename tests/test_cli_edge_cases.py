# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from click.testing import CliRunner

from coreason_council.main import run_council


def test_cli_empty_query() -> None:
    """Test CLI behavior with an empty query string."""
    runner = CliRunner()
    result = runner.invoke(run_council, [""])

    # Should run (defaults to General panel) but might result in "effectively empty" entropy checks
    assert result.exit_code == 0
    assert "Selected Panel" in result.output
    assert "Session started" in result.output


def test_cli_zero_rounds() -> None:
    """Test CLI behavior with max-rounds set to 0."""
    runner = CliRunner()
    # Should likely deadlock immediately after initial proposals
    result = runner.invoke(run_council, ["Test Query", "--max-rounds", "0"])

    assert result.exit_code == 0
    assert "MINORITY REPORT: Deadlock detected" in result.output
    # Depending on implementation, confidence might be low
    assert "Confidence: 0.1" in result.output


def test_cli_negative_rounds() -> None:
    """Test CLI behavior with negative max-rounds."""
    runner = CliRunner()
    # Logically same as 0 rounds (1 >= -1 is True)
    result = runner.invoke(run_council, ["Test Query", "--max-rounds", "-5"])

    assert result.exit_code == 0
    assert "MINORITY REPORT: Deadlock detected" in result.output


def test_cli_high_entropy_threshold() -> None:
    """Test CLI with entropy threshold > 1.0 (Always Consensus)."""
    runner = CliRunner()
    # Even with JaccardDissenter, entropy is <= 1.0. So 1.5 should always trigger consensus immediately.
    result = runner.invoke(run_council, ["Complex Query", "--entropy-threshold", "1.5"])

    assert result.exit_code == 0
    assert "Consensus" not in result.output  # The word consensus is in logs, not necessarily stdout output
    # But it should NOT be a deadlock
    assert "MINORITY REPORT" not in result.output
    assert "--- ALTERNATIVES" not in result.output
    assert "Confidence: 0.95" in result.output  # MockAggregator default confidence


def test_cli_negative_entropy_threshold() -> None:
    """Test CLI with negative entropy threshold (Always High Entropy until max rounds)."""
    runner = CliRunner()
    # Entropy (0.0 to 1.0) will never be <= -0.1. So it should run until max rounds and deadlock.
    result = runner.invoke(run_council, ["Simple Query", "--entropy-threshold", "-0.1", "--max-rounds", "2"])

    assert result.exit_code == 0
    assert "MINORITY REPORT: Deadlock detected" in result.output
