# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.persona import Persona
from coreason_council.core.models.trace import CouncilTrace, TopologyType
from coreason_council.core.models.verdict import Verdict, VerdictOption


def test_council_trace_initialization() -> None:
    trace = CouncilTrace(session_id="test-session-123", roster=["Alice", "Bob"], topology=TopologyType.STAR)
    assert trace.session_id == "test-session-123"
    assert trace.roster == ["Alice", "Bob"]
    assert trace.topology == TopologyType.STAR
    assert trace.transcripts == []
    assert trace.entropy_score is None


def test_council_trace_serialization() -> None:
    trace = CouncilTrace(session_id="test-session-123", roster=["Alice", "Bob"], topology=TopologyType.STAR)
    trace.log_interaction("Alice", "proposal", "I think X")

    assert len(trace.transcripts) == 1
    assert trace.transcripts[0].actor == "Alice"
    assert isinstance(trace.transcripts[0].timestamp, datetime)
    assert trace.transcripts[0].timestamp.tzinfo == timezone.utc

    json_output = trace.model_dump_json()
    data = json.loads(json_output)

    assert data["session_id"] == "test-session-123"
    assert len(data["transcripts"]) == 1
    assert data["transcripts"][0]["actor"] == "Alice"
    # Pydantic serializes datetime to ISO format by default
    assert "timestamp" in data["transcripts"][0]


def test_proposer_output_validation() -> None:
    # Valid
    output = ProposerOutput(proposer_id="p1", content="Answer", confidence=0.8)
    assert output.confidence == 0.8

    # Invalid confidence (> 1.0)
    with pytest.raises(ValidationError):
        ProposerOutput(proposer_id="p1", content="Answer", confidence=1.5)

    # Invalid confidence (< 0.0)
    with pytest.raises(ValidationError):
        ProposerOutput(proposer_id="p1", content="Answer", confidence=-0.1)


def test_critique_structure() -> None:
    critique = Critique(
        reviewer_id="p2",
        target_proposer_id="p1",
        content="I disagree",
        flaws_identified=["Fallacy A"],
        agreement_score=0.2,
    )
    assert critique.reviewer_id == "p2"
    assert critique.flaws_identified == ["Fallacy A"]


def test_verdict_structure() -> None:
    verdict = Verdict(
        content="Final Answer", confidence_score=0.95, supporting_evidence=["Fact 1"], dissenting_opinions=[]
    )
    assert verdict.content == "Final Answer"
    assert verdict.confidence_score == 0.95
    assert verdict.alternatives == []  # Default empty


def test_verdict_with_alternatives() -> None:
    option_a = VerdictOption(label="Option A", content="Do X", supporters=["p1"])
    option_b = VerdictOption(label="Option B", content="Do Y", supporters=["p2"])

    verdict = Verdict(
        content="Split Decision",
        confidence_score=0.1,
        alternatives=[option_a, option_b],
    )

    assert len(verdict.alternatives) == 2
    assert verdict.alternatives[0].label == "Option A"
    assert verdict.alternatives[0].supporters == ["p1"]


def test_persona_defaults() -> None:
    p = Persona(name="Test", system_prompt="You are a test.")
    assert p.capabilities == []
