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

from coreason_council.core.llm_client import LLMRequest, LLMResponse, MockLLMClient
from coreason_council.core.llm_proposer import CritiqueContent, LLMProposer, ProposalContent
from coreason_council.core.types import Critique, Persona, ProposerOutput


@pytest.fixture
def mock_persona() -> Persona:
    return Persona(name="TestPersona", system_prompt="You are a test persona.")


@pytest.mark.asyncio
async def test_propose_empty_query(mock_persona: Persona) -> None:
    """
    Edge Case: Empty query string.
    The LLMProposer should still attempt to send it (or validation could be added later).
    For now, we verify it constructs the request.
    """
    client = MockLLMClient(return_json=ProposalContent(content="I need a query.", confidence=0.1))
    proposer = LLMProposer(client)

    # We allow empty query at this level, expecting the LLM or higher layers to handle it.
    # But we want to ensure it doesn't crash.
    output = await proposer.propose("", mock_persona)

    assert output.content == "I need a query."
    assert output.confidence == 0.1


@pytest.mark.asyncio
async def test_metadata_propagation(mock_persona: Persona) -> None:
    """
    Edge Case: Verify metadata (usage, model) is correctly propagated from LLMResponse to ProposerOutput.
    """
    expected_usage = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}

    # Create a custom mock client that returns specific usage
    class UsageMockClient(MockLLMClient):
        async def get_completion(self, request: LLMRequest) -> LLMResponse:
            content_obj = ProposalContent(content="Metadata Test", confidence=0.9)
            return LLMResponse(
                content=content_obj.model_dump_json(),
                raw_content=content_obj,
                usage=expected_usage,
                finish_reason="stop",
                provider_metadata={"model": "gpt-4-turbo-mock", "id": "test-id"},
            )

    client = UsageMockClient()
    proposer = LLMProposer(client)

    output = await proposer.propose("Query", mock_persona)

    assert output.metadata["usage"] == expected_usage
    assert output.metadata["model"] == "gpt-4-turbo-mock"
    assert output.metadata["persona"] == "TestPersona"


@pytest.mark.asyncio
async def test_revise_proposal_many_critiques(mock_persona: Persona) -> None:
    """
    Complex Scenario: Revising a proposal with a large number of critiques.
    Verifies that all critiques are included in the prompt construction.
    """
    # Custom client to inspect the request content
    captured_request: LLMRequest | None = None

    class InspectionMockClient(MockLLMClient):
        async def get_completion(self, request: LLMRequest) -> LLMResponse:
            nonlocal captured_request
            captured_request = request
            return await super().get_completion(request)

    client = InspectionMockClient(return_json=ProposalContent(content="Revised", confidence=1.0))
    proposer = LLMProposer(client)

    original = ProposerOutput(
        proposer_id="llm-testpersona",
        content="Original",
        confidence=0.5,
    )

    # Generate 100 critiques
    critiques = []
    for i in range(100):
        critiques.append(
            Critique(
                reviewer_id=f"Reviewer_{i}",
                target_proposer_id="llm-testpersona",
                content=f"Critique {i}",
                flaws_identified=[f"Flaw {i}"],
                agreement_score=0.1,
            )
        )

    await proposer.revise_proposal(original, critiques, mock_persona)

    assert captured_request is not None
    user_content = captured_request.messages[0]["content"]

    # Verify strict inclusion
    assert "Original Proposal:" in user_content
    assert "Received Critiques:" in user_content
    assert f"Critique count: {len(critiques)}" not in user_content  # Not explicitly in prompt but implied by content

    # Check bounds
    assert "Critique from Reviewer_0:" in user_content
    assert "Critique from Reviewer_99:" in user_content
    assert "Critique 0" in user_content
    assert "Critique 99" in user_content


@pytest.mark.asyncio
async def test_propose_critique_revise_flow(mock_persona: Persona) -> None:
    """
    Complex Scenario: End-to-end flow of Propose -> Critique -> Revise.
    """
    # 1. Propose
    client_propose = MockLLMClient(return_json=ProposalContent(content="Initial", confidence=0.7))
    proposer_a = LLMProposer(client_propose)
    proposal = await proposer_a.propose("Is the sky blue?", mock_persona)
    assert proposal.content == "Initial"

    # 2. Critique (Proposer B critiques A)
    persona_b = Persona(name="Critic", system_prompt="You are a critic.")
    client_critique = MockLLMClient(
        return_json=CritiqueContent(content="It is actually violet.", flaws_identified=["Physics"], agreement_score=0.2)
    )
    proposer_b = LLMProposer(client_critique)

    critique = await proposer_b.critique_proposal(proposal, persona_b)
    assert critique.target_proposer_id == proposal.proposer_id
    assert critique.reviewer_id == "Critic"

    # 3. Revise (Proposer A revises based on B's critique)
    client_revise = MockLLMClient(
        return_json=ProposalContent(content="The sky appears blue due to scattering.", confidence=0.9)
    )
    # We reuse proposer_a but with a new client behavior (or we could use side_effect if we had a smarter mock)
    # For simplicity, we just create a new instance or inject the new client if possible.
    # Since LLMProposer takes client in init, let's just make a new one representing "Proposer A at time T2".
    proposer_a_revision = LLMProposer(client_revise)

    revised_proposal = await proposer_a_revision.revise_proposal(proposal, [critique], mock_persona)

    assert revised_proposal.content == "The sky appears blue due to scattering."
    assert revised_proposal.metadata["revision_of"] == proposal.proposer_id
    assert revised_proposal.confidence > proposal.confidence
