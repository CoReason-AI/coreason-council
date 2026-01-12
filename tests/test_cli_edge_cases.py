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
    """Test that an empty query defaults to the General panel and runs successfully."""
    runner = CliRunner()
    result = runner.invoke(run_council, [""])

    assert result.exit_code == 0
    # Should default to General Panel
    assert "Selected Panel: ['Generalist', 'Skeptic', 'Optimist']" in result.output
    assert "--- FINAL VERDICT ---" in result.output


def test_cli_panel_routing_medical() -> None:
    """Test that medical keywords trigger the Medical Panel."""
    runner = CliRunner()
    result = runner.invoke(run_council, ["What is the treatment for lung cancer?"])

    assert result.exit_code == 0
    assert "Selected Panel: ['Oncologist', 'Biostatistician', 'Regulatory']" in result.output


def test_cli_panel_routing_code() -> None:
    """Test that code keywords trigger the Code Panel."""
    runner = CliRunner()
    result = runner.invoke(run_council, ["Fix this python bug in the compiler"])

    assert result.exit_code == 0
    assert "Selected Panel: ['Architect', 'Security', 'QA']" in result.output


def test_cli_single_round_no_debate() -> None:
    """
    Test with --max-rounds 1.
    This should execute Phase 1 (Proposals) and then immediately break the loop
    because current_round (1) >= max_rounds (1).
    Effectively zero debate rounds.
    """
    runner = CliRunner()
    # Force high entropy so we would normally debate, but max-rounds 1 prevents it.
    result = runner.invoke(run_council, ["Complex query", "--max-rounds", "1", "--entropy-threshold", "-1.0"])

    assert result.exit_code == 0
    assert "Session started..." in result.output
    # Should trigger Deadlock because entropy is high (forced by -1.0 threshold vs positive entropy)
    # and we hit max rounds immediately.
    assert "Declaring Deadlock" in result.output or "MINORITY REPORT" in result.output
    assert "--- ALTERNATIVES (Deadlock) ---" in result.output


def test_cli_forced_high_entropy_loop() -> None:
    """
    Test with negative entropy threshold to force the loop to run until max rounds.
    With max-rounds 2, it should run 1 critique round.
    """
    runner = CliRunner()
    result = runner.invoke(run_council, ["Query", "--max-rounds", "2", "--entropy-threshold", "-0.1"])

    assert result.exit_code == 0
    # Logic:
    # Round 1: Entropy calculated. Entropy >= -0.1 (True).
    # Round 1 check: 1 >= 2 (False).
    # Critique Round 1 happens.
    # Round increments to 2.
    # Round 2: Entropy calculated.
    # Round 2 check: 2 >= 2 (True). Break. Deadlock.

    assert "MINORITY REPORT" in result.output
    # We can't easily assert the "number of rounds" from CLI output without verbose logs,
    # but successful execution implies the loop logic held.
