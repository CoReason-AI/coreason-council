# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import List, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """
    Represents a plan proposed by the system or a user.
    """

    id: str = Field(..., description="Unique identifier for the plan")
    title: str = Field(..., description="Title of the plan")
    description: Optional[str] = Field(None, description="Detailed description")
    tools: List[str] = Field(default_factory=list, description="List of tools/actions required by this plan")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the plan (0.0-1.0)")
