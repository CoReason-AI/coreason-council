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
from unittest.mock import AsyncMock

import pytest

from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.dissenter import MockDissenter
from coreason_council.core.proposer import MockProposer
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.core.types import Critique, Persona, ProposerOutput


@pytest.fixture
def basic_setup() -> tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator]:
    proposers = [
        MockProposer(proposer_id_prefix="p1", return_content="A"),
        MockProposer(proposer_id_prefix="p2", return_content="B"),
    ]
    personas = [
        Persona(name="P1", system_prompt="Sys1"),
        Persona(name="P2", system_prompt="Sys2"),
    ]
    dissenter = MockDissenter()
    aggregator = MockAggregator()
    return proposers, personas, dissenter, aggregator


@pytest.mark.asyncio
async def test_max_rounds_semantics(
    basic_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator],
) -> None:
    """
    Test exact behavior of max_rounds parameter.
    max_rounds=1 -> 0 debate rounds (Initial Proposal -> Entropy High -> Stop).
    max_rounds=2 -> 1 debate round (Initial -> Entropy High -> Debate -> Revise -> Entropy High -> Stop).
    """
    proposers, personas, dissenter, aggregator = basic_setup

    # Force High Entropy constantly
    dissenter.default_entropy_score = 0.9

    # Case A: max_rounds = 1
    speaker_1 = ChamberSpeaker(proposers, personas, dissenter, aggregator, entropy_threshold=0.1, max_rounds=1)
    _, trace_1 = await speaker_1.resolve_query("Query")

    # Should be 0 critique rounds
    critiques_1 = [e for e in trace_1.transcripts if "critique" in e.action]
    assert len(critiques_1) == 0
    assert trace_1.final_verdict is not None
    assert "Deadlock detected" in trace_1.final_verdict.content

    # Case B: max_rounds = 2
    speaker_2 = ChamberSpeaker(proposers, personas, dissenter, aggregator, entropy_threshold=0.1, max_rounds=2)
    _, trace_2 = await speaker_2.resolve_query("Query")

    # Should be 1 critique round (Round 1)
    # 2 proposers * 1 round = 2 critique actions
    critiques_2 = [e for e in trace_2.transcripts if "critique" in e.action]
    assert len(critiques_2) == 2
    assert trace_2.final_verdict is not None
    assert "Deadlock detected" in trace_2.final_verdict.content

    # Verify we had exactly Round 1 interactions
    assert any(e.action == "critique_round_1" for e in trace_2.transcripts)
    assert not any(e.action == "critique_round_2" for e in trace_2.transcripts)


@pytest.mark.asyncio
async def test_late_convergence(
    basic_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator],
) -> None:
    """
    Test that consensus reached at the very last allowed check results in Verdict, not Deadlock.
    max_rounds = 3.
    R1: High (Debate)
    R2: High (Debate)
    R3: Low (Consensus)
    """
    proposers, personas, dissenter, aggregator = basic_setup

    # Mock Entropy: High, High, Low
    # Calls:
    # 1. Initial Proposals (Round 1 check) -> High
    # 2. After R1 Debate (Round 2 check) -> High
    # 3. After R2 Debate (Round 3 check) -> Low
    dissenter.calculate_entropy = AsyncMock(side_effect=[0.9, 0.9, 0.05])  # type: ignore

    speaker = ChamberSpeaker(proposers, personas, dissenter, aggregator, entropy_threshold=0.1, max_rounds=3)
    verdict, trace = await speaker.resolve_query("Query")

    # Should NOT be deadlock
    assert "Deadlock detected" not in verdict.content

    # Should have had 2 rounds of debate
    critiques = [e for e in trace.transcripts if "critique" in e.action]
    # 2 proposers * 2 rounds = 4 critiques
    assert len(critiques) == 4

    # Verify specific rounds
    assert any(e.action == "critique_round_1" for e in trace.transcripts)
    assert any(e.action == "critique_round_2" for e in trace.transcripts)
    assert not any(e.action == "critique_round_3" for e in trace.transcripts)


@pytest.mark.asyncio
async def test_oscillating_disagreement(
    basic_setup: tuple[list[MockProposer], list[Persona], MockDissenter, MockAggregator],
) -> None:
    """
    Test a scenario where Proposers change their minds but never agree, ensuring Deadlock stops infinite loops.
    """
    _, personas, dissenter, aggregator = basic_setup

    # Custom Flip-Flopping Proposer
    class OscillatingProposer(MockProposer):
        def __init__(self, start_state: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.state = start_state

        async def revise_proposal(
            self, original_proposal: ProposerOutput, critiques: list[Critique], persona: Persona
        ) -> ProposerOutput:
            # Toggle State
            self.state = "B" if self.state == "A" else "A"
            new_content = f"{self.state}"
            return ProposerOutput(
                proposer_id=original_proposal.proposer_id,
                content=new_content,
                confidence=0.8,
            )

    proposers = [
        OscillatingProposer(start_state="A", proposer_id_prefix="p1", return_content="A"),
        OscillatingProposer(start_state="B", proposer_id_prefix="p2", return_content="B"),
    ]

    # Dissenter always sees High Entropy (since they are always A vs B or B vs A)
    dissenter.default_entropy_score = 0.9

    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=dissenter,
        aggregator=aggregator,
        entropy_threshold=0.1,
        max_rounds=4,
    )

    verdict, trace = await speaker.resolve_query("Oscillate")

    # Must end in Deadlock
    assert "Deadlock detected" in verdict.content

    # Should have run for 3 debate rounds (since max_rounds=4 means check R1, R2, R3, R4(Stop))
    # R1 check (High) -> Debate 1
    # R2 check (High) -> Debate 2
    # R3 check (High) -> Debate 3
    # R4 check (High) -> Stop

    critique_rounds = set()
    for e in trace.transcripts:
        if e.action.startswith("critique_round_"):
            critique_rounds.add(e.action)

    assert len(critique_rounds) == 3
    assert "critique_round_1" in critique_rounds
    assert "critique_round_2" in critique_rounds
    assert "critique_round_3" in critique_rounds
