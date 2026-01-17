# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from enum import Enum

from pydantic import BaseModel, Field


class PersonaType(str, Enum):
    ONCOLOGIST = "Oncologist"
    BIOSTATISTICIAN = "Biostatistician"
    REGULATORY = "Regulatory"
    ARCHITECT = "Architect"
    SECURITY = "Security"
    QA = "QA"
    SKEPTIC = "Skeptic"
    OPTIMIST = "Optimist"
    GENERALIST = "Generalist"


class Persona(BaseModel):
    name: str
    system_prompt: str
    capabilities: list[str] = Field(default_factory=list)
