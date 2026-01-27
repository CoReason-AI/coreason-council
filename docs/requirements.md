# Requirements

Coreason-council relies on a modern Python stack to ensure type safety, robust configuration, and high-performance async execution.

## Core Runtime
*   **Python 3.12+**: The project leverages the latest Python features for performance and typing.

## Microservice & API
*   **FastAPI**: For building the high-performance REST API.
*   **Uvicorn**: An ASGI web server implementation for Python.
*   **Pydantic**: For data validation and settings management.
*   **Pydantic Settings**: For 12-factor app configuration management via environment variables.

## Networking & LLM
*   **HTTPX**: A next-generation HTTP client for Python, used for making async calls to the AI Gateway.
*   **OpenAI**: Official Python library for the OpenAI API (v1.x), used when running in direct LLM mode.

## Utilities
*   **Loguru**: For simplified and powerful logging.
*   **Typer**: For building the CLI interface.
*   **PyYAML**: For parsing YAML configuration files (presets).

## Development & Testing
*   **Poetry**: For dependency management and packaging.
*   **Pytest**: For robust testing.
*   **Ruff**: An extremely fast Python linter and code formatter.
*   **MyPy**: For static type checking.
*   **Pre-commit**: For managing git hooks to ensure code quality.
