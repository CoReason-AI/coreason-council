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
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.proposer import MockProposer
from coreason_council.core.types import PersonaType


def test_panel_selector_init() -> None:
    selector = PanelSelector()
    assert selector
    assert len(selector.medical_panel) == 3
    assert len(selector.code_panel) == 3
    assert len(selector.general_panel) == 3


def test_select_medical_panel() -> None:
    selector = PanelSelector()
    query = "Is this drug safe for a cancer patient?"
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3

    # Check Personas
    names = [p.name for p in personas]
    assert "Oncologist" in names
    assert "Biostatistician" in names
    assert "Regulatory" in names

    # Check Capabilities
    capabilities = [c for p in personas for c in p.capabilities]
    assert PersonaType.ONCOLOGIST in capabilities

    # Check Proposers (default factory makes MockProposer)
    assert isinstance(proposers[0], MockProposer)
    assert "mock-oncologist" in proposers[0].proposer_id_prefix


def test_select_code_panel() -> None:
    selector = PanelSelector()
    query = "There is a bug in my python code."
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3

    names = [p.name for p in personas]
    assert "Architect" in names
    assert "Security" in names
    assert "QA" in names

    capabilities = [c for p in personas for c in p.capabilities]
    assert PersonaType.ARCHITECT in capabilities


def test_select_general_panel_fallback() -> None:
    selector = PanelSelector()
    query = "What is the meaning of life?"
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3

    names = [p.name for p in personas]
    assert "Generalist" in names
    assert "Skeptic" in names
    assert "Optimist" in names


def test_custom_proposer_factory() -> None:
    class CustomProposer(MockProposer):
        pass

    def custom_factory(persona) -> CustomProposer:  # type: ignore
        return CustomProposer(proposer_id_prefix=f"custom-{persona.name}")

    selector = PanelSelector(proposer_factory=custom_factory)
    query = "test"
    proposers, personas = selector.select_panel(query)

    assert isinstance(proposers[0], CustomProposer)
    # Check that the factory was used correctly
    assert "custom-" in proposers[0].proposer_id_prefix


def test_edge_case_empty_query() -> None:
    """Test that an empty query falls back to the General Panel."""
    selector = PanelSelector()
    query = ""
    _, personas = selector.select_panel(query)

    names = [p.name for p in personas]
    assert "Generalist" in names


def test_edge_case_whitespace_query() -> None:
    """Test that a whitespace-only query falls back to the General Panel."""
    selector = PanelSelector()
    query = "   \n  "
    _, personas = selector.select_panel(query)

    names = [p.name for p in personas]
    assert "Generalist" in names


def test_edge_case_ambiguous_query_priority() -> None:
    """
    Test that when keywords from multiple domains appear, priority is respected.
    Implementation priority: Medical > Code > General.
    Query: "Write python code to analyze cancer data."
    Expected: Medical Panel (because 'cancer' triggers medical).
    """
    selector = PanelSelector()
    query = "Write python code to analyze cancer data."
    _, personas = selector.select_panel(query)

    names = [p.name for p in personas]
    assert "Oncologist" in names
    # Verify we did NOT get Code panel
    assert "Architect" not in names


def test_edge_case_case_insensitivity_and_punctuation() -> None:
    """Test that matching handles mixed case and punctuation."""
    selector = PanelSelector()
    # "DRUG" is mixed case, "patient!" has punctuation
    query = "Is this DRUG safe for the patient!?"
    _, personas = selector.select_panel(query)

    names = [p.name for p in personas]
    assert "Oncologist" in names


def test_edge_case_factory_exception() -> None:
    """Test that exceptions in the factory are propagated."""

    def broken_factory(persona: object) -> MockProposer:
        raise ValueError("Factory failed")

    selector = PanelSelector(proposer_factory=broken_factory)
    query = "simple query"

    with pytest.raises(ValueError, match="Factory failed"):
        selector.select_panel(query)
