from typing import List

from pydantic import BaseModel

from coreason_council.core.models.plan import Plan
from coreason_council.reviewer import ApprovalStatus, review_plan


# Mock UserContext
class UserContext(BaseModel):
    user_id: str
    groups: List[str] = []


def test_review_plan_high_risk_admin() -> None:
    plan = Plan(id="1", title="Nuke DB", tools=["delete_database"], confidence=0.8)
    user = UserContext(user_id="admin_user", groups=["admin"])

    result = review_plan(plan, user)
    assert result.status == ApprovalStatus.APPROVED


def test_review_plan_high_risk_non_admin() -> None:
    plan = Plan(id="2", title="Nuke DB", tools=["delete_database"], confidence=0.8)
    user = UserContext(user_id="intern", groups=["intern"])

    result = review_plan(plan, user)
    assert result.status == ApprovalStatus.REJECTED
    assert result.rejection_reason is not None
    assert "lacks required role" in result.rejection_reason


def test_review_plan_safe_tool() -> None:
    plan = Plan(id="3", title="List files", tools=["ls"], confidence=0.8)
    user = UserContext(user_id="intern", groups=["intern"])

    result = review_plan(plan, user)
    assert result.status == ApprovalStatus.APPROVED


def test_review_plan_no_tools() -> None:
    plan = Plan(id="4", title="Think", confidence=0.9)
    user = UserContext(user_id="intern", groups=["intern"])

    result = review_plan(plan, user)
    assert result.status == ApprovalStatus.APPROVED
