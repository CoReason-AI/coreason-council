# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from pathlib import Path
from typing import Callable

import yaml

from coreason_council.core.proposer import BaseProposer, MockProposer
from coreason_council.core.types import Persona, PersonaType
from coreason_council.settings import settings
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
        self._load_presets()

    def _load_presets(self) -> None:
        """Loads persona presets from YAML configuration."""
        presets_path = Path(settings.presets_file)
        if not presets_path.is_absolute():
            # Try to resolve relative to the package root if needed, or assume CWD
            # Usually settings.presets_file should be absolute or relative to CWD.
            # For this setup, we'll try to find it relative to CWD first.
            if not presets_path.exists():
                # fallback to looking in the module location
                import coreason_council

                module_path = Path(coreason_council.__file__).parent
                possible_path = module_path / "resources" / "presets.yaml"
                if possible_path.exists():
                    presets_path = possible_path

        if not presets_path.exists():
            logger.warning(f"Presets file not found at {presets_path}. Using empty presets.")
            self.presets: dict[str, list[Persona]] = {}
            return

        try:
            with open(presets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            self.presets = {}
            for category, items in data.items():
                self.presets[category] = []
                for item in items:
                    capabilities = [PersonaType(cap) for cap in item.get("capabilities", [])]
                    self.presets[category].append(
                        Persona(name=item["name"], system_prompt=item["system_prompt"], capabilities=capabilities)
                    )
        except Exception as e:
            logger.error(f"Failed to load presets from {presets_path}: {e}")
            self.presets = {}

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
        selected_personas: list[Persona] = []

        # Heuristic Classification
        medical_keywords = {"drug", "medicine", "patient", "treatment", "dose", "clinical", "symptom", "cancer"}
        code_keywords = {"code", "python", "bug", "function", "api", "software", "debug", "compile", "class"}

        if any(keyword in query_lower for keyword in medical_keywords):
            logger.info("Query classified as MEDICAL. Selecting Medical Panel.")
            selected_personas = self.presets.get("medical", [])
        elif any(keyword in query_lower for keyword in code_keywords):
            logger.info("Query classified as CODE. Selecting Code Panel.")
            selected_personas = self.presets.get("code", [])
        else:
            logger.info("Query classified as GENERAL. Selecting General Panel.")
            selected_personas = self.presets.get("general", [])

        # Fallback if presets are missing or empty
        if not selected_personas:
            logger.warning("No matching panel found or presets empty. Defaulting to a single Generalist.")
            selected_personas = [
                Persona(
                    name="Generalist",
                    system_prompt="You are a helpful assistant.",
                    capabilities=[PersonaType.GENERALIST],
                )
            ]

        # Instantiate Proposers
        proposers = [self.proposer_factory(p) for p in selected_personas]

        return proposers, selected_personas
