# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Annotated, Any
from pydantic import BaseModel, Field

class ProposerOutput(BaseModel):
    proposer_id: str
    content: str
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    metadata: dict[str, Any] = Field(default_factory=dict)

class Critique(BaseModel):
    reviewer_id: str
    target_proposer_id: str
    content: str
    flaws_identified: list[str]
    agreement_score: Annotated[float, Field(ge=0.0, le=1.0)]
