# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from unittest.mock import patch

from click.testing import CliRunner

from coreason_council.core.types import Critique, Persona, ProposerOutput
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
    result = runner.invoke(run_council, ["Complex query", "--max-rounds", "1", "--entropy-threshold", "0.0"])

    assert result.exit_code == 0
    assert "--- FINAL VERDICT ---" in result.output
    assert "MINORITY REPORT: Deadlock detected" in result.output
    assert "--- ALTERNATIVES (Deadlock) ---" in result.output


def test_run_council_cli_with_trace() -> None:
    """Test the CLI execution with the --show-trace flag."""
    runner = CliRunner()
    result = runner.invoke(run_council, ["Explain Quantum", "--show-trace"])

    assert result.exit_code == 0
    assert "--- DEBATE TRANSCRIPT ---" in result.output
    assert "--- VOTE TALLY ---" in result.output
    # Check for presence of timestamps or actor names which indicate trace is printing
    assert "propose" in result.output or "verdict" in result.output


def test_cli_complex_convergence_trace() -> None:
    """
    Test a complex scenario where the debate runs for multiple rounds
    before converging, verifying the trace output captures the evolution.
    """
    runner = CliRunner()

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
        result = runner.invoke(
            run_council, ["Debate Query", "--max-rounds", "5", "--entropy-threshold", "0.1", "--show-trace"]
        )

        assert result.exit_code == 0, f"CLI Failed with: {result.exception}"

        # Verify trace structure
        output = result.output
        assert "--- DEBATE TRANSCRIPT ---" in output

        # Should see multiple rounds of critiques/revisions
        assert "critique_round_1" in output
        assert "revise_round_1" in output
        assert "critique_round_2" in output
        assert "revise_round_2" in output

        # Should NOT see round 3 critique (since entropy dropped)
        assert "critique_round_3" not in output

        # Should end in normal verdict (not deadlock)
        assert "MINORITY REPORT" not in output
        assert "Content: Mock Verdict" in output
