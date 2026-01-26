# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import asyncio
from pathlib import Path
from typing import Any

import yaml

from coreason_council.core.llm_aggregator import LLMAggregator
from coreason_council.core.llm_client import GatewayLLMClient
from coreason_council.core.llm_proposer import LLMProposer
from coreason_council.core.models.persona import Persona, PersonaType
from coreason_council.settings import settings
from coreason_council.utils.logger import logger


class CouncilService:
    """
    Service layer for the Consensus Microservice (Service L).
    Implements a Scatter-Gather pattern for parallel council execution.
    """

    def __init__(self) -> None:
        self.presets = self._load_presets()

    def _load_presets(self) -> dict[str, Persona]:
        """Loads persona presets from YAML configuration into a flat dict by name."""
        presets_path = Path(settings.presets_file)
        if not presets_path.is_absolute():
            import coreason_council

            module_path = Path(coreason_council.__file__).parent
            possible_path = module_path / "resources" / "presets.yaml"
            if possible_path.exists():
                presets_path = possible_path

        if not presets_path.exists():
            logger.warning(f"Presets file not found at {presets_path}. Using empty presets.")
            return {}

        personas = {}
        try:
            with open(presets_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if isinstance(data, dict):
                for _category, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if "name" in item:
                                name = item["name"]
                                capabilities = [PersonaType(cap) for cap in item.get("capabilities", [])]
                                personas[name] = Persona(
                                    name=name,
                                    system_prompt=item.get("system_prompt", ""),
                                    capabilities=capabilities,
                                )
        except Exception as e:
            logger.error(f"Failed to load presets from {presets_path}: {e}")
            return {}
        return personas

    def get_persona(self, name: str) -> Persona:
        """Retrieves a Persona by name, or creates a generic one if not found."""
        if name in self.presets:
            return self.presets[name]

        logger.info(f"Persona '{name}' not found in presets. Creating generic persona.")
        return Persona(
            name=name,
            system_prompt=f"You are {name}, a helpful advisor.",
            capabilities=[PersonaType.GENERALIST],
        )

    async def convene_session(self, topic: str, persona_names: list[str], model: str = "gpt-4o") -> dict[str, Any]:
        """
        Orchestrates a council session: Scatter (Propose) -> Gather -> Synthesize.
        """
        client = GatewayLLMClient()

        # Resolve Personas
        personas = [self.get_persona(name) for name in persona_names]

        # Phase 1: Scatter (Propose)
        # We assume LLMProposer and LLMAggregator will be updated to accept 'model'
        # or we rely on client default. Since we haven't updated them yet,
        # we'll instantiate them as is.
        # (Self-correction: I should update them to support model propagation).
        # For now, I'll pass kwargs to constructor if I update them,
        # or I'll patch them.

        # I will update LLMProposer and LLMAggregator in the next file writes.
        # So I will assume the constructor accepts `model`.
        proposers = [LLMProposer(client, model=model) for _ in personas]

        logger.info(f"Session Start: Topic='{topic}', Personas={persona_names}, Model='{model}'")

        # Parallel Execution
        proposal_tasks = []
        for proposer, persona in zip(proposers, personas, strict=True):
            proposal_tasks.append(proposer.propose(topic, persona))

        votes = await asyncio.gather(*proposal_tasks)

        # Phase 2: Synthesize (Aggregate)
        aggregator = LLMAggregator(client, model=model)
        verdict = await aggregator.aggregate(votes, critiques=[], is_deadlock=False)

        # Construct Response
        return {
            "verdict": verdict.content,
            "confidence_score": verdict.confidence_score,
            "dissent": "; ".join(verdict.dissenting_opinions),
            "votes": [
                {
                    "proposer": v.proposer_id,
                    "content": v.content,
                    "confidence": v.confidence,
                }
                for v in votes
            ],
        }
