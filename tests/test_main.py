# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

import pytest
from coreason_council.main import hello_world


def test_hello_world_runs() -> None:
    """Simple test to verify hello_world entry point runs."""
    try:
        result = hello_world()
        assert result == "Hello World!"
    except Exception as e:
        pytest.fail(f"hello_world() raised {e}")
