from fastapi.testclient import TestClient

from coreason_council.server import app

client = TestClient(app)


def test_submit_review_approved() -> None:
    payload = {
        "plan": {"id": "p1", "title": "Safe Plan", "tools": ["read_file"], "confidence": 0.9},
        "user_context": {"user_id": "u1", "groups": ["user"]},
    }
    response = client.post("/v1/plan/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "approved"


def test_submit_review_rejected() -> None:
    payload = {
        "plan": {"id": "p2", "title": "Dangerous Plan", "tools": ["delete_database"], "confidence": 0.9},
        "user_context": {"user_id": "u2", "groups": ["user"]},
    }
    response = client.post("/v1/plan/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "rejected"
    assert "lacks required role" in data["rejection_reason"]
