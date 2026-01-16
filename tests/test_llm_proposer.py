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

from coreason_council.core.llm_client import MockLLMClient
from coreason_council.core.llm_proposer import CritiqueContent, LLMProposer, ProposalContent
from coreason_council.core.models.interaction import Critique, ProposerOutput
from coreason_council.core.models.persona import Persona

# Original imports were: Critique, Persona, ProposerOutput


@pytest.fixture
def mock_persona() -> Persona:
    return Persona(name="TestPersona", system_prompt="You are a test persona.")


@pytest.mark.asyncio
async def test_propose_success(mock_persona: Persona) -> None:
    """Test that propose returns a valid ProposerOutput using the LLM client."""
    expected_content = ProposalContent(content="Generated Proposal", confidence=0.85)
    client = MockLLMClient(return_json=expected_content)
    proposer = LLMProposer(client)

    output = await proposer.propose("Test Query", mock_persona)

    assert isinstance(output, ProposerOutput)
    assert output.content == "Generated Proposal"
    assert output.confidence == 0.85
    assert output.proposer_id == "llm-testpersona"
    assert output.metadata["persona"] == "TestPersona"


@pytest.mark.asyncio
async def test_propose_failure_no_structured_output(mock_persona: Persona) -> None:
    """Test that propose raises ValueError if LLM doesn't return structured data."""
    # Mock client returns plain text, not the expected object structure in raw_content
    client = MockLLMClient(return_content="Plain text")
    proposer = LLMProposer(client)

    with pytest.raises(ValueError, match="LLM failed to return structured ProposalContent"):
        await proposer.propose("Test Query", mock_persona)


@pytest.mark.asyncio
async def test_critique_proposal_failure_no_structured_output(mock_persona: Persona) -> None:
    """Test that critique_proposal raises ValueError if LLM doesn't return structured data."""
    client = MockLLMClient(return_content="Plain text")
    proposer = LLMProposer(client)

    target = ProposerOutput(
        proposer_id="target-id",
        content="Target Content",
        confidence=0.9,
    )

    with pytest.raises(ValueError, match="LLM failed to return structured CritiqueContent"):
        await proposer.critique_proposal(target, mock_persona)


@pytest.mark.asyncio
async def test_critique_proposal_success(mock_persona: Persona) -> None:
    """Test that critique_proposal returns a valid Critique."""
    expected_critique = CritiqueContent(
        content="Generated Critique",
        flaws_identified=["Flaw 1", "Flaw 2"],
        agreement_score=0.4,
    )
    client = MockLLMClient(return_json=expected_critique)
    proposer = LLMProposer(client)

    target = ProposerOutput(
        proposer_id="target-id",
        content="Target Content",
        confidence=0.9,
    )

    critique = await proposer.critique_proposal(target, mock_persona)

    assert isinstance(critique, Critique)
    assert critique.content == "Generated Critique"
    assert critique.flaws_identified == ["Flaw 1", "Flaw 2"]
    assert critique.agreement_score == 0.4
    assert critique.reviewer_id == "TestPersona"
    assert critique.target_proposer_id == "target-id"


@pytest.mark.asyncio
async def test_revise_proposal_success(mock_persona: Persona) -> None:
    """Test that revise_proposal returns a revised ProposerOutput."""
    expected_revision = ProposalContent(content="Revised Content", confidence=0.95)
    client = MockLLMClient(return_json=expected_revision)
    proposer = LLMProposer(client)

    original = ProposerOutput(
        proposer_id="llm-testpersona",
        content="Original Content",
        confidence=0.8,
    )
    critiques = [
        Critique(
            reviewer_id="Reviewer",
            target_proposer_id="llm-testpersona",
            content="Critique Content",
            flaws_identified=["Flaw"],
            agreement_score=0.2,
        )
    ]

    revised = await proposer.revise_proposal(original, critiques, mock_persona)

    assert revised.content == "Revised Content"
    assert revised.confidence == 0.95
    assert revised.proposer_id == original.proposer_id
    assert revised.metadata["revision_of"] == original.proposer_id
    assert revised.metadata["critique_count"] == 1


@pytest.mark.asyncio
async def test_revise_proposal_no_critiques(mock_persona: Persona) -> None:
    """Test that revise_proposal returns original if no critiques are provided."""
    client = MockLLMClient()  # Config doesn't matter, shouldn't be called
    proposer = LLMProposer(client)

    original = ProposerOutput(
        proposer_id="llm-testpersona",
        content="Original Content",
        confidence=0.8,
    )

    revised = await proposer.revise_proposal(original, [], mock_persona)

    assert revised is original


@pytest.mark.asyncio
async def test_revise_proposal_failure_no_structured_output(mock_persona: Persona) -> None:
    """Test that revise_proposal raises ValueError if LLM doesn't return structured data."""
    client = MockLLMClient(return_content="Plain text")
    proposer = LLMProposer(client)

    original = ProposerOutput(
        proposer_id="llm-testpersona",
        content="Original Content",
        confidence=0.8,
    )
    critiques = [
        Critique(
            reviewer_id="Reviewer",
            target_proposer_id="llm-testpersona",
            content="Critique Content",
            flaws_identified=["Flaw"],
            agreement_score=0.2,
        )
    ]

    with pytest.raises(ValueError, match="LLM failed to return structured ProposalContent"):
        await proposer.revise_proposal(original, critiques, mock_persona)
