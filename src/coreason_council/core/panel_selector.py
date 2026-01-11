# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Callable

from coreason_council.core.proposer import BaseProposer, MockProposer
from coreason_council.core.types import Persona, PersonaType
from coreason_council.utils.logger import logger


class PanelSelector:
    """
    Component responsible for selecting the appropriate 'Board of Advisors' (Personas and Proposers)
    based on the incoming query type.
    """

    def __init__(self, proposer_factory: Callable[[Persona], BaseProposer] | None = None) -> None:
        """
        Initializes the PanelSelector.

        Args:
            proposer_factory: Optional callable to create a Proposer from a Persona.
                              Defaults to creating a MockProposer.
        """
        self.proposer_factory = proposer_factory or self._default_mock_factory

        # Define Persona Presets
        self.medical_panel = [
            Persona(
                name="Oncologist",
                system_prompt="You are an expert Oncologist.",
                capabilities=[PersonaType.ONCOLOGIST],
            ),
            Persona(
                name="Biostatistician",
                system_prompt="You are an expert Biostatistician.",
                capabilities=[PersonaType.BIOSTATISTICIAN],
            ),
            Persona(
                name="Regulatory",
                system_prompt="You are an expert Regulatory Affairs Specialist.",
                capabilities=[PersonaType.REGULATORY],
            ),
        ]

        self.code_panel = [
            Persona(
                name="Architect",
                system_prompt="You are a Senior Software Architect.",
                capabilities=[PersonaType.ARCHITECT],
            ),
            Persona(
                name="Security",
                system_prompt="You are a Security Engineer.",
                capabilities=[PersonaType.SECURITY],
            ),
            Persona(
                name="QA",
                system_prompt="You are a QA Engineer.",
                capabilities=[PersonaType.QA],
            ),
        ]

        self.general_panel = [
            Persona(
                name="Generalist",
                system_prompt="You are a helpful assistant.",
                capabilities=[PersonaType.GENERALIST],
            ),
            Persona(
                name="Skeptic",
                system_prompt="You are a critical thinker and skeptic.",
                capabilities=[PersonaType.SKEPTIC],
            ),
            Persona(
                name="Optimist",
                system_prompt="You are an optimistic visionary.",
                capabilities=[PersonaType.OPTIMIST],
            ),
        ]

    def _default_mock_factory(self, persona: Persona) -> BaseProposer:
        """Default factory that creates a MockProposer."""
        return MockProposer(proposer_id_prefix=f"mock-{persona.name.lower()}")

    def select_panel(self, query: str) -> tuple[list[BaseProposer], list[Persona]]:
        """
        Selects a panel of Proposers and Personas based on the query content.

        Args:
            query: The input query string.

        Returns:
            A tuple of (list[BaseProposer], list[Persona]).
        """
        query_lower = query.lower()
        selected_personas: list[Persona]

        # Heuristic Classification
        medical_keywords = {"drug", "medicine", "patient", "treatment", "dose", "clinical", "symptom", "cancer"}
        code_keywords = {"code", "python", "bug", "function", "api", "software", "debug", "compile", "class"}

        if any(keyword in query_lower for keyword in medical_keywords):
            logger.info("Query classified as MEDICAL. Selecting Medical Panel.")
            selected_personas = self.medical_panel
        elif any(keyword in query_lower for keyword in code_keywords):
            logger.info("Query classified as CODE. Selecting Code Panel.")
            selected_personas = self.code_panel
        else:
            logger.info("Query classified as GENERAL. Selecting General Panel.")
            selected_personas = self.general_panel

        # Instantiate Proposers
        proposers = [self.proposer_factory(p) for p in selected_personas]

        return proposers, selected_personas
