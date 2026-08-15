# CAG Estimator

![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688)
![CAG](https://img.shields.io/badge/architecture-CAG-purple)
![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)
![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

API for software project estimation using LLMs, based on a CAG (Context-Augmented Generation) architecture: historical estimation examples are injected into the prompt to guide the model toward more accurate and consistent budgets.

Every LLM call goes through a [LiteLLM](https://docs.litellm.ai/)-backed wrapper that adds:
- **Provider fallback** : tries `PRIMARY_MODEL` first (OpenAI by default), retries it a few times, then falls over to `FALLBACK_MODEL` (Anthropic by default) so a single expired key, rate limit, or outage doesn't take the whole feature down.
- **Redis exact-match cache** : identical requests (same system prompt, question, model and `max_tokens`) are served from Redis instead of calling the LLM again. The question text is normalized (trimmed, lowercased, whitespace-collapsed) before hashing so trivial formatting differences still hit the cache; this does not catch typos or paraphrases (that would need a semantic/embedding-based cache).
- **Structured logging** : every phase of a call (cache check, dispatch, retry, fallback, success/failure) is logged via `structlog`, so a single request can be traced end-to-end.

Streamlit is a thin HTTP client of the API (not a second place that talks to the LLM directly): it POSTs to `/api/v1/estimate/stream` and renders the response as it streams in.

## Project Structure
```
cag-estimator/
├── app/
│   ├── main.py             -- FastAPI app entrypoint
│   ├── config.py           -- Settings loaded from .env
│   ├── constants.py        -- Model pricing table
│   ├── dependencies.py     -- Shared singletons (cache, LLM wrapper)
│   ├── routers/            -- Endpoint handlers
│   │   └── estimations.py
│   ├── services/           -- LLM calls and business logic
│   │   ├── llm_service.py      -- Prompt building + orchestration
│   │   ├── llm_wrapper.py      -- LiteLLM fallback, cache, cost tracking
│   │   ├── cache.py            -- Redis exact-match cache
│   │   └── evaluation.py
│   ├── context/            -- CAG reference data
│   │   └── examples.py
│   └── schemas/            -- Request/response schemas
│       └── estimations.py
├── streamlit_app.py         -- Streamlit chat UI (shares the API's system prompt)
├── .env.example
├── pyproject.toml
└── README.md
```

## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- Redis (used by the exact-match cache; `docker compose up` starts one automatically)
- At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` : the app validates this at startup and refuses to boot without one

## Setup
```bash
# Install dependencies
uv sync

# Copy the environment file and fill in your API key(s)
cp .env.example .env
```

## Running the API
```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `/docs`. Requires a Redis instance reachable at `REDIS_URL` (defaults to `redis://localhost:6379`) : run one locally with `docker run -p 6379:6379 redis:7-alpine` if you're not using `docker compose`.

### Running with Docker
```bash
docker compose up --build
```
This also starts a `redis` service and points the API at it automatically.

## Running the Streamlit Chat
```bash
uv run streamlit run streamlit_app.py
```

Streamlit talks to the API over HTTP at `ESTIMATOR_API_BASE_URL` (from the same `.env`) : it holds no LLM API key and never calls a provider directly. Its sidebar shows:
- The active system prompt and injected CAG examples (read-only, built locally for display only)
- The configured primary/fallback model and cache TTL
- Response time for the last call

## Environment Variables
See [`.env.example`](.env.example) for the full list of variables and their possible values.

## Endpoints

| Method | Path                       | Description                                    |
|--------|----------------------------|-------------------------------------------------|
| GET    | `/health`                  | Health check                                     |
| POST   | `/api/v1/estimate`         | Generate a project estimation (blocking)         |
| POST   | `/api/v1/estimate/stream`  | Generate a project estimation (Server-Sent Events) |

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
  "model": "gpt-4o-mini-2024-07-18",
  "provider": "openai",
  "tokens_used": 1234,
  "estimated_cost": 0.0021,
  "cache_hit": false,
  "evaluation": {
    "score": 1.0,
    "issues": []
  }
}
```

`cache_hit: true` means the response was served straight from Redis without calling the LLM. `provider`/`model` reflect whichever deployment actually answered : the primary model, or the fallback if the primary failed.

### `POST /api/v1/estimate/stream`

Same request body as above. Responds with `text/event-stream`: an `event: token` per chunk of generated text, followed by a final `event: done`, or `event: error` if the call fails after exhausting retries and fallback.

## Testing an Estimation
```bash
# Blocking
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{"transcription": "Client meeting summary describing the project..."}'

# Streaming
curl -N -X POST http://127.0.0.1:8000/api/v1/estimate/stream \
  -H "Content-Type: application/json" \
  -d '{"transcription": "Client meeting summary describing the project..."}'
```
