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
from typing import Optional

from pydantic import BaseModel, Field

from coreason_council.core.models.interaction import ProposerOutput
from coreason_council.core.models.verdict import Verdict


class TopologyType(str, Enum):
    STAR = "star"
    CHAIN = "chain"
    MESH = "mesh"
    ROUND_TABLE = "round_table"


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
    roster: list[str]  # List of persona names/ids
    transcripts: list[TranscriptEntry] = Field(default_factory=list)  # Chronological log of interactions
    topology: TopologyType
    entropy_score: Optional[float] = None
    vote_tally: Optional[dict[str, int]] = None
    final_votes: Optional[list[ProposerOutput]] = None
    final_verdict: Optional[Verdict] = None

    def log_interaction(self, actor: str, action: str, content: str) -> None:
        entry = TranscriptEntry(actor=actor, action=action, content=content, timestamp=datetime.now(timezone.utc))
        self.transcripts.append(entry)
