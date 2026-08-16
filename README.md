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
- **Two-layer Redis cache** : an exact-match layer (same system prompt, question, model and `max_tokens`, question normalized before hashing) is checked first; on a miss, a semantic layer embeds the question (OpenAI `text-embedding-3-small`) and looks for a prior response above a cosine-similarity threshold within the same system-prompt/model bucket, catching paraphrases and typos the exact layer can't. Input guardrails always run before either cache layer, and a response is only written to cache after it passes output validation.
- **Structured logging** : every phase of a call (cache check, dispatch, retry, fallback, success/failure) is logged via `structlog`, so a single request can be traced end-to-end.
- **Input guardrails** : every request is checked for prompt-injection patterns, off-topic content (a cheap LLM-based topic classifier), and OpenAI moderation flags before it reaches the estimation prompt. A rejection returns `400` with a plain, user-facing message.

Streamlit is a thin HTTP client of the API (not a second place that talks to the LLM directly), with two tabs: **Chat**, which POSTs to `/api/v1/estimate/stream` and renders the response as it streams in, and **Structured Estimate**, an `st.form` that POSTs to `/api/v2/estimate` and renders the validated, phase-by-phase result.

## Project Structure
```
cag-estimator/
├── app/
│   ├── main.py             -- FastAPI app entrypoint
│   ├── config.py           -- Settings loaded from .env
│   ├── constants.py        -- Model pricing table
│   ├── dependencies.py     -- Shared singletons (cache, LLM wrapper)
│   ├── routers/            -- Endpoint handlers
│   │   ├── estimations.py      -- v1: free-text estimation (blocking + SSE)
│   │   └── estimation_v2.py    -- v2: structured estimation (schema in/out)
│   ├── services/           -- LLM calls and business logic
│   │   ├── llm_service.py      -- Prompt building + orchestration (v1)
│   │   ├── llm_wrapper.py      -- LiteLLM fallback, cache, cost tracking
│   │   ├── cache.py            -- Redis cache: exact-match + semantic layers
│   │   ├── embeddings.py       -- Embedding client for the semantic cache
│   │   ├── guardrails.py       -- Input guardrails (injection, topic, moderation)
│   │   └── evaluation.py
│   ├── prompts/             -- Jinja2-templated, versioned prompts (v2)
│   │   ├── loader.py            -- render_estimation_prompt(request, version)
│   │   └── estimation/
│   │       ├── v1/                  -- system.j2, user.j2, examples.j2
│   │       └── v2/                  -- risk-first variant, own examples
│   ├── context/            -- CAG reference data (v1)
│   │   └── examples.py
│   └── schemas/            -- Request/response schemas
│       ├── estimations.py      -- v1: transcription-based
│       └── estimation.py       -- v2: description + enums, structured result
├── streamlit_app.py         -- Streamlit UI: Chat (v1) + Structured Estimate (v2) tabs
├── .env.example
├── pyproject.toml
└── README.md
```

## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- Redis (used by both cache layers; `docker compose up` starts one automatically)
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

## Running the Streamlit UI
```bash
uv run streamlit run streamlit_app.py
```

Streamlit talks to the API over HTTP at `ESTIMATOR_API_BASE_URL` (from the same `.env`) : it holds no LLM API key and never calls a provider directly. It has two tabs:
- **Chat** : free-text SSE chat against `/api/v1/estimate/stream`, with client-side length checks and friendly errors for guardrail rejections.
- **Structured Estimate** : an `st.form` (description + project type/detail level/output format) against `/api/v2/estimate`, rendering the validated result as summary, totals and a phase table.

Its sidebar shows:
- The active system prompt and injected CAG examples (read-only, built locally for display only)
- The configured primary/fallback model and cache TTL
- Response time for the last call

## Environment Variables
See [`.env.example`](.env.example) for the full list of variables and their possible values.

## Endpoints

| Method | Path                                    | Description                                        |
|--------|------------------------------------------|-----------------------------------------------------|
| GET    | `/health`                                | Health check                                         |
| POST   | `/api/v1/estimate`                       | Free-text project estimation (blocking)              |
| POST   | `/api/v1/estimate/stream`                | Free-text project estimation (Server-Sent Events)    |
| POST   | `/api/v2/estimate?prompt_version=v1\|v2` | Structured project estimation (schema in/out)        |

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

### `POST /api/v2/estimate`

Structured estimation: the request is rendered through a Jinja2 prompt template (see [Prompt Versions](#prompt-versions) below) and the LLM's answer is validated against a Pydantic schema (via [Instructor](https://python.useinstructor.com/)) instead of returned as free text.

**Request body:**
```json
{
  "description": "Internal admin tool to manage support tickets, with role-based access and Slack notifications.",
  "project_type": "internal_tool",
  "detail_level": "detailed",
  "output_format": "phases_table",
  "reference_projects": null
}
```
`project_type`: `mobile_app` | `web_saas` | `internal_tool` | `data_pipeline`. `detail_level`: `summary` | `medium` | `detailed`. `output_format`: `phases_table` | `line_items` | `narrative`. `reference_projects` is optional : a list of `{name, description, actual_duration_weeks, actual_cost_eur}` past projects used as an extra calibration anchor.

**Response:**
```json
{
  "result": {
    "summary": "...",
    "total_duration_weeks": 8,
    "total_cost_eur": 12500,
    "confidence_pct": 75,
    "phases": [
      {"name": "Discovery", "duration_weeks": 1, "cost_eur": 625, "confidence_pct": 80, "assumptions": ["..."]}
    ]
  },
  "prompt_version": "v1"
}
```
Totals are summed from `phases` in Python rather than asked of the LLM, since small models are unreliable at keeping a separately-declared total consistent with the phases across retries. A confidence below 30% must come with `summary` starting in `"Out of scope:"` and `phases: []`, enforced by a Pydantic validator that Instructor retries against.

#### Prompt Versions

Prompts live under `app/prompts/estimation/<version>/` (`system.j2`, `user.j2`, `examples.j2`), rendered by `app/prompts/loader.py`. `v1` and `v2` differ deliberately (tone and few-shot examples), and the endpoint picks one via the `?prompt_version=` query param (defaults to `v1`); an unknown version returns `400`. Every render logs a `prompt_rendered` event with the version and a content hash, for tracing which exact prompt a given call used.

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

# Structured (v2)
curl -X POST http://127.0.0.1:8000/api/v2/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Internal admin tool to manage support tickets, with role-based access and Slack notifications.",
    "project_type": "internal_tool",
    "detail_level": "detailed",
    "output_format": "phases_table"
  }'
```
