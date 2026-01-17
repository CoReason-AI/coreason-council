# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council


from coreason_council.core.models.persona import Persona
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.proposer import MockProposer

# Original imports were: Persona


def test_panel_selector_init() -> None:
    selector = PanelSelector()
    assert selector
    # We now look at presets dict instead of hardcoded attributes
    assert len(selector.presets["medical"]) == 3
    assert len(selector.presets["code"]) == 3
    assert len(selector.presets["general"]) == 3


def test_select_medical_panel() -> None:
    selector = PanelSelector()
    query = "Is this drug safe for a cancer patient?"
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3

    # Check if correct panel selected (Medical contains Oncologist)
    # The persona names are now capitalized in the YAML (Oncologist)
    # The PersonaType enum values were also capitalized in core/types.py
    assert personas[0].name == "Oncologist"
    assert isinstance(proposers[0], MockProposer)


def test_select_code_panel() -> None:
    selector = PanelSelector()
    query = "There is a bug in my python code."
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3
    assert personas[0].name == "Architect"


def test_select_general_panel_fallback() -> None:
    selector = PanelSelector()
    query = "What is the meaning of life?"
    proposers, personas = selector.select_panel(query)

    assert len(proposers) == 3
    assert len(personas) == 3
    assert personas[0].name == "Generalist"


def test_custom_proposer_factory() -> None:
    def custom_factory(p: Persona) -> MockProposer:
        return MockProposer(proposer_id_prefix=f"custom-{p.name}")

    selector = PanelSelector(proposer_factory=custom_factory)
    query = "Any query"
    proposers, personas = selector.select_panel(query)

    # MockProposer stores prefix in proposer_id_prefix attribute
    # And generates proposer_id during `propose` call (returned in ProposerOutput).
    # But wait, `MockProposer` objects themselves don't have `proposer_id` attribute unless generated?
    # No, looking at code:
    # return ProposerOutput(proposer_id=f"{self.proposer_id_prefix}-{persona.name}", ...)
    # The `MockProposer` instance only has `proposer_id_prefix`.

    assert isinstance(proposers[0], MockProposer)
    assert proposers[0].proposer_id_prefix.startswith("custom-")


def test_edge_case_empty_query() -> None:
    """Test behavior with empty query string."""
    selector = PanelSelector()
    proposers, personas = selector.select_panel("")

    # Expect fallback to General panel
    names = [p.name for p in personas]
    assert "Generalist" in names
    assert len(proposers) == 3


def test_edge_case_no_keywords_match() -> None:
    """Test query with no matching keywords defaults to General."""
    selector = PanelSelector()
    query = "xyz123 random text"
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


def test_edge_case_case_insensitivity_and_punctuation() -> None:
    """Test that matching handles mixed case and punctuation."""
    selector = PanelSelector()
    # "DRUG" is mixed case, "patient!" has punctuation
    query = "Is this DRUG safe for the patient!?"
    _, personas = selector.select_panel(query)

    names = [p.name for p in personas]
    assert "Oncologist" in names
