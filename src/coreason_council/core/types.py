# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any, Annotated
from pydantic import BaseModel, Field

class PersonaType(str, Enum):
    ONCOLOGIST = "oncologist"
    BIOSTATISTICIAN = "biostatistician"
    REGULATORY = "regulatory"
    ARCHITECT = "architect"
    SECURITY = "security"
    QA = "qa"
    SKEPTIC = "skeptic"
    OPTIMIST = "optimist"
    GENERALIST = "generalist"

class TopologyType(str, Enum):
    STAR = "star"
    CHAIN = "chain"
    MESH = "mesh"
    ROUND_TABLE = "round_table"

class VoteOption(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"

class Persona(BaseModel):
    name: str
    system_prompt: str
    capabilities: list[str] = Field(default_factory=list)

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

class Verdict(BaseModel):
    content: str
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    supporting_evidence: list[str] = Field(default_factory=list)
    dissenting_opinions: list[str] = Field(default_factory=list)

class TranscriptEntry(BaseModel):
    actor: str
    action: str
    content: str
    timestamp: datetime

class CouncilTrace(BaseModel):
    """
    Serializable log object for Council sessions (The "Glass Box").
    """
    session_id: str
    roster: list[str] # List of persona names/ids
    transcripts: list[TranscriptEntry] = Field(default_factory=list) # Chronological log of interactions
    topology: TopologyType
    entropy_score: Optional[float] = None
    vote_tally: Optional[dict[str, int]] = None
    final_verdict: Optional[Verdict] = None

    def log_interaction(self, actor: str, action: str, content: str) -> None:
        entry = TranscriptEntry(
            actor=actor,
            action=action,
            content=content,
            timestamp=datetime.now(timezone.utc)
        )
        self.transcripts.append(entry)
