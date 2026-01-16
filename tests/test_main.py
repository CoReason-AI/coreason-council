# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.persona import Persona

# Original imports were: Critique, Persona, ProposerOutput
from coreason_council.main import app

runner = CliRunner()


def test_run_council_cli_basic() -> None:
    """Test basic CLI execution with defaults."""
    result = runner.invoke(app, ["Is this valid?"])
    assert result.exit_code == 0
    assert "Selected Panel" in result.output
    assert "Session started" in result.output
    assert "FINAL VERDICT" in result.output


def test_run_council_cli_deadlock() -> None:
    """Test CLI output format during a deadlock (simulated by high entropy)."""
    # We rely on Mock behavior, but we can't easily force deadlock purely via CLI args
    # unless we patch the Dissenter inside main.
    with patch("coreason_council.main.JaccardDissenter") as mock_dissenter_class:
        mock_dissenter = mock_dissenter_class.return_value

        # Force high entropy
        async def mock_calculate_entropy(proposals: list[Any]) -> float:
            return 0.9

        mock_dissenter.calculate_entropy.side_effect = mock_calculate_entropy

        # Mock critique to return valid object
        async def mock_critique(target: Any, persona: Any) -> Any:
            return Critique(
                reviewer_id="rev",
                target_proposer_id="target",
                content="critique",
                flaws_identified=[],
                agreement_score=0.1,
            )

        mock_dissenter.critique.side_effect = mock_critique

        # Run with short max rounds to force deadlock quickly
        result = runner.invoke(app, ["Debate this", "--max-rounds", "2", "--entropy-threshold", "0.5"])

        assert result.exit_code == 0
        assert "Session started" in result.output
        assert "ALTERNATIVES (Deadlock)" in result.output


def test_run_council_cli_with_trace() -> None:
    """Test that --show-trace flag outputs the transcript."""
    result = runner.invoke(app, ["Trace check", "--show-trace"])
    assert result.exit_code == 0
    assert "DEBATE TRANSCRIPT" in result.output
    assert "VOTE TALLY" in result.output


def test_cli_complex_convergence_trace() -> None:
    """
    Test a complex scenario where the debate runs for multiple rounds
    before converging, verifying the trace output captures the evolution.
    """
    # We patch the JaccardDissenter to simulate high entropy then low entropy
    with patch("coreason_council.main.JaccardDissenter") as mock_dissenter_class:
        mock_dissenter = mock_dissenter_class.return_value

        # State container for closure
        state = {"call_count": 0}

        # Helper to create awaitable at runtime (inside the loop)
        async def mock_calculate_entropy(proposals: list[ProposerOutput]) -> float:
            # Sequence: High -> High -> Low
            entropy_sequence = [0.8, 0.5, 0.05]
            val = entropy_sequence[min(state["call_count"], len(entropy_sequence) - 1)]
            state["call_count"] += 1
            return val

        mock_dissenter.calculate_entropy.side_effect = mock_calculate_entropy

        # Mock critique method to return a valid object
        async def mock_critique(target: ProposerOutput, persona: Persona) -> Critique:
            return Critique(
                reviewer_id=persona.name,
                target_proposer_id=target.proposer_id,
                content="Mock Critique",
                flaws_identified=["Flaw"],
                agreement_score=0.5,
            )

        mock_dissenter.critique.side_effect = mock_critique

        # Run with max rounds 5, threshold 0.1
        result = runner.invoke(app, ["Debate Query", "--max-rounds", "5", "--entropy-threshold", "0.1", "--show-trace"])

        assert result.exit_code == 0
        assert "DEBATE TRANSCRIPT" in result.output

        # Verify multiple rounds happened
        # We expect entropy checks.
        # Check logs if possible, or trace output
        assert "critique_round_1" in result.output
        # Should converge eventually
        assert "FINAL VERDICT" in result.output
        assert "Consensus" in result.output or "Supported by" in result.output  # Depending on vote tally logic
