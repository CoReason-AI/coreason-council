# coreason-council

**The Senate of the CoReason Platform**

[![CI](https://github.com/CoReason-AI/coreason_council/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_council/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Prosperity%203.0-blue.svg)](https://prosperitylicense.com/versions/3.0.0)

## Executive Summary

**coreason-council** acts as the Multi-Agent Consensus Engine ("The Jury") for the CoReason platform. While `coreason-cortex` executes the thinking, `coreason-council` validates it. Its primary mandate is to implement **Ensemble Reasoning** to reduce hallucination and bias through diversity of thought.

It operates on the principle that while one Large Language Model (LLM) may err, a diverse coalition of models (e.g., GPT-4, Claude, Llama) critiquing one another will converge on the truth. It is responsible for orchestrating **Mixture-of-Agents (MoA)** topologies, managing debate rounds, and synthesizing a final, high-confidence output from conflicting viewpoints.

## Functional Philosophy: The Divergence-Convergence Loop

1.  **Diversity by Design:** Never rely on a single weight distribution. True consensus requires distinct "Proposers" (different models or personas).
2.  **Blind Independence:** Initial proposals are generated in isolation to prevent "Anchoring Bias" (groupthink).
3.  **Adversarial Review:** Agreement is cheap; critique is valuable. The system actively seeks dissent before accepting a consensus.
4.  **Semantic Aggregation:** Do not just "vote" (A vs B). The final output is a *synthesis* that incorporates the best nuance from all perspectives.

## Core Architecture

The Council consists of four primary components:

### 1. The Chamber Speaker (The Orchestrator)
The central manager that convenes the council. It manages the lifecycle of a "Session", selects the appropriate "Board of Advisors" (Personas) based on the query, and orchestrates the debate phases.

### 2. The Proposers (The Voices)
Independent workers that generate initial answers. They are wrapped in specific system prompts (Personas) to enforce diverse perspectives (e.g., "The Skeptic", "The Architect", "The Oncologist"). They operate in isolation during the opening phase.

### 3. The Dissenter (The Critic)
A specialized role focused on **Falsification**. It scans proposals for logical fallacies, data inconsistencies, or hallucinations. It calculates the **Semantic Entropy** (disagreement score) between proposals. High entropy triggers debate rounds; low entropy allows for immediate aggregation.

### 4. The Aggregator (The Judge)
The synthesis engine that creates the final "Verdict". It takes the proposals and critiques to generate a new answer that resolves the conflict, including a "Consensus Confidence Score" (0.0 - 1.0).

## Installation

### Prerequisites
- Python 3.12+
- Poetry

### Steps

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_council.git
    cd coreason_council
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

## Microservice Deployment

Coreason-council is designed to operate as a standalone microservice (Service L) using Docker, leveraging **FastAPI** and **Uvicorn** for high-performance async execution.

### 1. Build the Docker Image
```sh
docker build -t coreason-council:0.6.0 .
```

### 2. Run the Service
The service requires the `GATEWAY_URL` environment variable to connect to the internal AI Gateway.

```sh
docker run -d \
  -p 8000:8000 \
  -e GATEWAY_URL="http://your-gateway-url:8000/v1" \
  coreason-council:0.6.0
```

### 3. API Endpoints
The service exposes a REST API for convening council sessions.

*   **POST /v1/session/convene**: Orchestrate a parallel debate session (Scatter-Gather).
*   **GET /health**: Health check for Kubernetes probes.

See [docs/usage.md](docs/usage.md) for detailed API usage.

## Configuration

To use the LLM capabilities (e.g., OpenAI models), you must set the following environment variable:

```sh
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### CLI Usage

You can run a council session directly from the command line:

```sh
# Run with Mock agents (default, no API key needed)
poetry run council "What is the best way to handle concurrent tasks in Python?"

# Run with Real LLM agents (requires OPENAI_API_KEY)
poetry run council --llm "Explain the concept of Semantic Entropy."

# View help and options
poetry run council --help
```

**Options:**
- `--llm`: Use real LLM agents (OpenAI) instead of Mock agents.
- `--max-rounds <int>`: Maximum number of debate rounds (default: 3).
- `--entropy-threshold <float>`: Entropy threshold for consensus (default: 0.1).
- `--max-budget <int>`: Maximum budget in operations (default: 100).
- `--show-trace`: Display the full debate transcript.

### Programmatic Usage

You can integrate `coreason-council` into your own Python applications:

```python
import asyncio
from coreason_council.core.speaker import ChamberSpeaker
from coreason_council.core.panel_selector import PanelSelector
from coreason_council.core.dissenter import JaccardDissenter
from coreason_council.core.aggregator import MockAggregator
from coreason_council.core.budget import SimpleBudgetManager

async def main():
    query = "Is 50mg of Aspirin safe for children?"

    # 1. Select Panel
    panel_selector = PanelSelector() # Defaults to MockProposers
    proposers, personas = panel_selector.select_panel(query)

    # 2. Initialize Components
    dissenter = JaccardDissenter()
    aggregator = MockAggregator()
    budget_manager = SimpleBudgetManager()

    # 3. Initialize Speaker
    speaker = ChamberSpeaker(
        proposers=proposers,
        personas=personas,
        dissenter=dissenter,
        aggregator=aggregator,
        budget_manager=budget_manager,
    )

    # 4. Resolve Query
    verdict, trace = await speaker.resolve_query(query)

    print(f"Verdict: {verdict.content}")
    print(f"Confidence: {verdict.confidence_score}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Development

-   **Run Linter (Ruff) & Type Checker (MyPy)**:
    ```sh
    poetry run pre-commit run --all-files
    ```

-   **Run Tests**:
    ```sh
    poetry run pytest
    ```

## License

This project is licensed under the **Prosperity Public License 3.0**.
See the [LICENSE](LICENSE) file for details.
