# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council


from coreason_council.core.types import Verdict, VerdictOption


def test_verdict_option_edge_cases() -> None:
    # Edge Case: Empty supporters list
    # It is syntactically valid for an option to have no supporters
    # (e.g., a synthesized option nobody explicitly voted for yet)
    opt = VerdictOption(label="Option Zero", content="No support", supporters=[])
    assert opt.supporters == []

    # Edge Case: Empty strings
    opt_empty = VerdictOption(label="", content="", supporters=[""])
    assert opt_empty.label == ""
    assert opt_empty.supporters == [""]


def test_verdict_serialization_complex() -> None:
    """
    Test that a Verdict with alternatives serializes and deserializes correctly.
    This ensures the 'Glass Box' requirement works for the new data structure.
    """
    opt1 = VerdictOption(label="A", content="Plan A", supporters=["Alice", "Bob"])
    opt2 = VerdictOption(label="B", content="Plan B", supporters=["Charlie"])

    verdict = Verdict(
        content="Deadlock",
        confidence_score=0.1,
        alternatives=[opt1, opt2],
        supporting_evidence=["Ev1"],
        dissenting_opinions=["Diss1"],
    )

    # Serialize
    json_str = verdict.model_dump_json()

    # Deserialize
    verdict_out = Verdict.model_validate_json(json_str)

    # Verify integrity
    assert verdict_out.content == "Deadlock"
    assert len(verdict_out.alternatives) == 2
    assert verdict_out.alternatives[0].label == "A"
    assert verdict_out.alternatives[0].supporters == ["Alice", "Bob"]
    assert verdict_out.alternatives[1].content == "Plan B"


def test_complex_verdict_scenario() -> None:
    """
    Complex scenario: Large number of alternatives with special characters.
    """
    alternatives = []
    for i in range(10):
        alternatives.append(
            VerdictOption(
                label=f"Option #{i} (Special: &%@!)",
                content=f'Content for option {i}\nWith newlines and "quotes"',
                supporters=[f"voter_{x}" for x in range(i)],  # Variable number of supporters
            )
        )

    verdict = Verdict(content="Complex\nMulti-line\nVerdict", confidence_score=0.0, alternatives=alternatives)

    assert len(verdict.alternatives) == 10
    assert verdict.alternatives[9].label == "Option #9 (Special: &%@!)"
    assert len(verdict.alternatives[9].supporters) == 9
    assert "\n" in verdict.alternatives[5].content


def test_high_confidence_with_alternatives() -> None:
    """
    Test that the model allows high confidence even if alternatives are present.
    This is a data model test, not a business logic test.
    The Aggregator *logic* ensures low confidence, but the *type* should be flexible.
    """
    verdict = Verdict(
        content="Consensus but with alternatives recorded",
        confidence_score=0.99,
        alternatives=[VerdictOption(label="Minority", content="Ignored", supporters=["Dave"])],
    )
    assert verdict.confidence_score == 0.99
    assert len(verdict.alternatives) == 1
