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


def test_run_council_cli_basic() -> None:
    """Test the basic CLI execution with a simple query."""
    runner = CliRunner()
    # Use a high entropy threshold to ensure immediate consensus (Story A)
    # This guarantees the 'Mock Verdict' output without deadlock
    result = runner.invoke(run_council, ["Is 50mg of Aspirin safe?", "--entropy-threshold", "1.0"])

    assert result.exit_code == 0
    assert "Selected Panel" in result.output
    assert "Session started..." in result.output
    assert "--- FINAL VERDICT ---" in result.output
    assert "Content: Mock Verdict" in result.output


def test_run_council_cli_deadlock() -> None:
    """Test the CLI execution triggering deadlock."""
    runner = CliRunner()
    # Use strict entropy threshold to force deadlock with mocks
    result = runner.invoke(
        run_council, ["Complex query", "--max-rounds", "1", "--entropy-threshold", "0.0"]
    )

    assert result.exit_code == 0
    assert "--- FINAL VERDICT ---" in result.output
    assert "MINORITY REPORT: Deadlock detected" in result.output
    assert "--- ALTERNATIVES (Deadlock) ---" in result.output
