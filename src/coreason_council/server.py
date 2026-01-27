# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coreason_council.core.council_service import CouncilService
from coreason_council.core.models.plan import Plan
from coreason_council.reviewer import ReviewResult, review_plan
from coreason_council.utils.logger import logger

try:
    from coreason_identity.models import UserContext
except ImportError:  # pragma: no cover
    # Fallback for when coreason-identity is not installed
    class UserContext(BaseModel):  # type: ignore
        sub: str
        email: str
        permissions: List[str] = []


app = FastAPI(
    title="CoReason Council Service",
    description="Microservice L: The Boardroom/Jury (Consensus Engine)",
    version="0.5.0",
)

# Initialize Service (Singleton pattern via module scope)
service = CouncilService()


class ConveneRequest(BaseModel):
    topic: str
    personas: List[str]
    model: str = "gpt-4o"


class VoteResponse(BaseModel):
    proposer: str
    content: str
    confidence: float


class ConveneResponse(BaseModel):
    verdict: str
    confidence_score: float
    dissent: Optional[str] = None
    votes: List[VoteResponse]


class ReviewRequest(BaseModel):
    plan: Plan
    user_context: UserContext


@app.post("/v1/plan/review", response_model=ReviewResult)
async def submit_for_review(request: ReviewRequest) -> Any:
    """
    Reviews a plan against security policies and user permissions.
    """
    logger.info(f"Received review request for plan: '{request.plan.id}'")
    return review_plan(request.plan, request.user_context)


@app.post("/v1/session/convene", response_model=ConveneResponse)
async def convene_session(request: ConveneRequest) -> Any:
    """
    Orchestrates a parallel debate/consensus session.
    """
    logger.info(f"Received convene request for topic: '{request.topic}'")
    try:
        result = await service.convene_session(
            topic=request.topic,
            persona_names=request.personas,
            model=request.model,
        )
        return result
    except Exception as e:
        logger.exception("Failed to convene session")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
