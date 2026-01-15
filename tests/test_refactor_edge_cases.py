# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import pytest
from coreason_council.core.models.interaction import ProposerOutput
from coreason_council.core.models.trace import CouncilTrace, TopologyType
from coreason_council.core.models.verdict import Verdict, VerdictOption
from pydantic import ValidationError


def test_trace_full_serialization_cycle() -> None:
    """
    Edge Case: Ensure the CouncilTrace (which aggregates all other models)
    can be serialized to JSON and deserialized back without data loss,
    especially ensuring nested models (Verdict, alternatives) are handled correctly.
    """
    # Create a complex trace object
    trace = CouncilTrace(
        session_id="session-complex-1", roster=["Alice", "Bob"], topology=TopologyType.ROUND_TABLE, entropy_score=0.45
    )

    # Add transcripts
    trace.log_interaction("Alice", "propose", "Proposal A")
    trace.log_interaction("Bob", "critique", "Critique of A")

    # Add a Verdict with Alternatives (Deadlock scenario)
    verdict = Verdict(
        content="Deadlock Reached",
        confidence_score=0.1,
        supporting_evidence=["Ev 1"],
        dissenting_opinions=["Diss 1"],
        alternatives=[
            VerdictOption(label="Opt A", content="Content A", supporters=["Alice"]),
            VerdictOption(label="Opt B", content="Content B", supporters=["Bob"]),
        ],
    )
    trace.final_verdict = verdict
    trace.vote_tally = {"Opt A": 1, "Opt B": 1}

    # Serialize
    json_str = trace.model_dump_json()

    # Deserialize
    rehydrated_trace = CouncilTrace.model_validate_json(json_str)

    # Verify Deep Equality
    assert rehydrated_trace.session_id == trace.session_id
    assert rehydrated_trace.topology == trace.topology
    assert len(rehydrated_trace.transcripts) == 2
    assert rehydrated_trace.transcripts[0].actor == "Alice"
    assert rehydrated_trace.final_verdict is not None
    assert rehydrated_trace.final_verdict.content == "Deadlock Reached"
    assert len(rehydrated_trace.final_verdict.alternatives) == 2
    assert rehydrated_trace.final_verdict.alternatives[0].supporters == ["Alice"]
    assert rehydrated_trace.vote_tally == {"Opt A": 1, "Opt B": 1}


def test_verdict_option_edge_cases() -> None:
    """
    Edge Case: VerdictOption validation.
    """
    # 1. Empty supporters list (Synthesized option with no explicit backers)
    opt = VerdictOption(label="Synthesis", content="New Idea", supporters=[])
    assert opt.supporters == []

    # 2. Empty Content
    opt_empty = VerdictOption(label="Label", content="", supporters=["A"])
    assert opt_empty.content == ""


def test_verdict_validation_edge_cases() -> None:
    """
    Edge Case: Verdict validation boundaries.
    """
    # Boundary: Confidence Score exactly 0.0 and 1.0
    v_zero = Verdict(content="X", confidence_score=0.0)
    assert v_zero.confidence_score == 0.0

    v_one = Verdict(content="X", confidence_score=1.0)
    assert v_one.confidence_score == 1.0

    # Invalid: Confidence > 1.0
    with pytest.raises(ValidationError):
        Verdict(content="X", confidence_score=1.01)


def test_proposer_output_metadata_serialization() -> None:
    """
    Edge Case: Ensure arbitrary metadata in ProposerOutput handles
    nested types correctly during serialization.
    """
    output = ProposerOutput(
        proposer_id="p1",
        content="content",
        confidence=0.5,
        metadata={"nested": {"key": "value"}, "list": [1, 2, 3], "none_val": None},
    )

    json_str = output.model_dump_json()
    restored = ProposerOutput.model_validate_json(json_str)

    assert restored.metadata["nested"]["key"] == "value"
    assert restored.metadata["list"] == [1, 2, 3]
    assert restored.metadata["none_val"] is None
