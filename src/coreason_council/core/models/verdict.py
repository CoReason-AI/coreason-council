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
from typing import Annotated
from pydantic import BaseModel, Field

class VoteOption(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

class VerdictOption(BaseModel):
    label: str
    content: str
    supporters: list[str]

class Verdict(BaseModel):
    content: str
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    supporting_evidence: list[str] = Field(default_factory=list)
    dissenting_opinions: list[str] = Field(default_factory=list)
    alternatives: list[VerdictOption] = Field(default_factory=list)
