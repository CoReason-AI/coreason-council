# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from typing import TYPE_CHECKING

from coreason_council.core.models.plan import Plan

if TYPE_CHECKING:
    from coreason_identity.models import UserContext


def require_medical_director_approval(plan: Plan, user_context: "UserContext") -> bool:
    """
    Policy: Returns True only if the user is a Medical Director
    OR if the plan confidence is extremely high (>0.95).
    """
    # Assuming UserContext has a 'permissions' attribute which is a list/set of strings
    if "Medical Director" in user_context.permissions:
        return True

    if plan.confidence > 0.95:
        return True

    return False
