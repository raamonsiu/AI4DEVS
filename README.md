# CAG Estimator

![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
![CAG](https://img.shields.io/badge/architecture-CAG-purple)
![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)
![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9)
![License](https://img.shields.io/badge/license-MIT-green)

API for software project estimation using LLMs, based on a CAG (Context-Augmented Generation) architecture: historical estimation examples are injected into the prompt to guide the model toward more accurate and consistent budgets.

## Project Structure
```
cag-estimator/
├── app/
│   ├── main.py             -- FastAPI app entrypoint
│   ├── config.py           -- Settings loaded from .env
│   ├── constants.py        -- Model pricing table
│   ├── routers/            -- Endpoint handlers
│   │   └── estimations.py
│   ├── services/           -- LLM calls and business logic
│   │   └── llm_service.py
│   ├── context/            -- CAG reference data
│   │   └── examples.py
│   └── schemas/            -- Request/response schemas
│       └── estimations.py
├── .env.example
├── pyproject.toml
└── README.md
```

## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup
```bash
# Install dependencies
uv sync

# Copy the environment file and fill in your API key
cp .env.example .env
```

## Running the API
```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## Environment Variables
See [`.env.example`](.env.example) for the full list of variables and their possible values.

## Endpoints

| Method | Path                | Description                          |
|--------|---------------------|---------------------------------------|
| GET    | `/health`           | Health check                          |
| POST   | `/api/v1/estimate`  | Generate a project estimation         |

### `POST /api/v1/estimate`

**Request body:**
```json
{
  "transcription": "Client meeting summary describing the project..."
}
```

**Response:**
```json
{
  "estimation": "...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "tokens_used": 1234,
  "estimated_cost": 0.0021
}
```
