# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_council

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from coreason_council.server import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("coreason_council.server.service.convene_session", new_callable=AsyncMock)
def test_convene_session(mock_convene: AsyncMock) -> None:
    mock_convene.return_value = {
        "verdict": "APPROVED",
        "confidence_score": 0.9,
        "dissent": "None",
        "votes": [
            {"proposer": "A", "content": "Yes", "confidence": 0.9},
            {"proposer": "B", "content": "Yes", "confidence": 0.8},
        ],
    }

    payload = {"topic": "Test Topic", "personas": ["A", "B"], "model": "gpt-4o"}

    response = client.post("/v1/session/convene", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "APPROVED"
    assert data["confidence_score"] == 0.9
    assert len(data["votes"]) == 2

    mock_convene.assert_awaited_once_with(topic="Test Topic", persona_names=["A", "B"], model="gpt-4o")


@patch("coreason_council.server.service.convene_session", new_callable=AsyncMock)
def test_convene_session_failure(mock_convene: AsyncMock) -> None:
    mock_convene.side_effect = Exception("Service Error")

    payload = {"topic": "Test Topic", "personas": ["A"], "model": "gpt-4o"}

    response = client.post("/v1/session/convene", json=payload)

    assert response.status_code == 500
    assert "Service Error" in response.json()["detail"]
