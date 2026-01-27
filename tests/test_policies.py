from typing import List

from pydantic import BaseModel

from coreason_council.core.models.plan import Plan
from coreason_council.policies import require_medical_director_approval


# Mock UserContext
class UserContext(BaseModel):
    sub: str
    email: str
    permissions: List[str] = []


def test_require_medical_director_approval_authorized() -> None:
    plan = Plan(id="1", title="Surgery", confidence=0.5)
    user = UserContext(sub="dr_house", email="dr@h.com", permissions=["Medical Director"])
    assert require_medical_director_approval(plan, user) is True


def test_require_medical_director_approval_high_confidence() -> None:
    plan = Plan(id="2", title="Surgery", confidence=0.96)
    user = UserContext(sub="intern", email="intern@h.com", permissions=["intern"])
    assert require_medical_director_approval(plan, user) is True


def test_require_medical_director_approval_denied() -> None:
    plan = Plan(id="3", title="Surgery", confidence=0.5)
    user = UserContext(sub="intern", email="intern@h.com", permissions=["intern"])
    assert require_medical_director_approval(plan, user) is False
