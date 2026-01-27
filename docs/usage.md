# Usage Guide

Coreason-council can be used in three ways: via the CLI, as a library, or as a deployed microservice.

## 1. Microservice (Service L)

The primary deployment mode for production environments is as a Dockerized microservice. This allows `coreason-council` to act as the "Jury" service for the broader CoReason platform.

### Running the Service

You can run the service using Docker. Ensure you have the `GATEWAY_URL` configured to point to your AI Gateway.

```bash
docker run -d \
  -p 8000:8000 \
  -e GATEWAY_URL="http://coreason-ai-gateway:8000/v1" \
  coreason-council:0.4.0
```

### API Endpoints

The service exposes a REST API documentation at `http://localhost:8000/docs`.

#### Convene a Session

**Endpoint:** `POST /v1/session/convene`

**Request:**
```json
{
  "topic": "Should we migrate our database to PostgreSQL?",
  "personas": ["The Architect", "The DBA", "The CFO"],
  "model": "gpt-4o"
}
```

**Response:**
```json
{
  "verdict": "Migrating to PostgreSQL is recommended due to its robustness...",
  "confidence_score": 0.95,
  "dissent": null,
  "votes": [
    {
      "proposer": "llm-the architect",
      "content": "Yes, for long-term scalability...",
      "confidence": 0.9
    },
    {
      "proposer": "llm-the dba",
      "content": "Yes, specifically for ACID compliance...",
      "confidence": 0.98
    }
  ]
}
```

#### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

## 2. CLI Usage

The CLI is useful for ad-hoc queries, testing, and debugging.

```bash
# General query with default settings
poetry run council "Explain Quantum Entanglement"

# Specify a different model and increase debate rounds
poetry run council "Is Rust better than C++ for systems programming?" --max-rounds 5 --llm
```

## 3. Library Usage

You can import core components to build custom consensus flows.

```python
from coreason_council.core.council_service import CouncilService

service = CouncilService()
# Note: This is an async method
result = await service.convene_session(
    topic="What is the capital of France?",
    persona_names=["Generalist"],
    model="gpt-4o"
)
```
