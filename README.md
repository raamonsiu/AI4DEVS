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

A typed request (`description` + project type / detail level / output format) maps to a typed, validated `EstimationResult` — phases, totals and a confidence score — via [Instructor](https://python.useinstructor.com/) + Pydantic. The whole pipeline lives in a single service class, `app/services/estimation.py`:

1. **Input guardrails** — moderation, prompt-injection regex, PII heuristics.
2. **Exact-match cache** lookup (Redis).
3. **Semantic cache** lookup (embedding similarity, Redis Stack + redisvl).
4. **Render** the versioned Jinja2 prompt.
5. **LLM call** via Instructor, re-prompting on validator failures.
6. **Output guardrail** — normalise low-confidence answers into an explicit "Out of scope:" response.
7. **Write both caches** — only after the output passed validation.

Order matters: guardrails run *before* any cache so a malicious or PII description can never be served from cache, and both cache writes happen *after* output validation so a bad estimation is never persisted and replayed for the whole TTL.

Every LLM call goes through a [LiteLLM](https://docs.litellm.ai/)-backed wrapper that adds:
- **Provider fallback** — tries `PRIMARY_MODEL` first (OpenAI by default), retries it, then falls over to `FALLBACK_MODEL` (Anthropic by default) so an expired key, rate limit or outage doesn't take the feature down. Structured calls go through the same Router, so they keep the fallback guarantee.
- **Cost tracking** — every response reports the model, provider, latency and USD cost of that call. A cache hit reports `cost_usd: 0.0`, because nothing was spent.
- **Structured logging** — every phase of a call (guardrails, cache check, prompt render, dispatch, retry, fallback, success/failure) is logged via `structlog`, so a request can be traced end-to-end.

Streamlit is a thin HTTP client of the API (not a second place that talks to the LLM), with two tabs: a **structured estimate** form that POSTs an `EstimationRequest` and renders the validated result, and a **chat** that streams a free-text estimation from the SSE endpoint.

## Project Structure
```
cag-estimator/
├── app/
│   ├── main.py             -- FastAPI app entrypoint
│   ├── config.py           -- Settings loaded from .env
│   ├── constants.py        -- Model pricing table
│   ├── dependencies.py     -- Shared singletons (caches, wrapper, service)
│   ├── routers/
│   │   ├── estimations.py      -- POST /api/v1/estimate (HTTP error mapping only)
│   │   └── estimations_text.py -- Free-text endpoints: /estimate/text, /estimate/stream
│   ├── services/
│   │   ├── estimation.py       -- EstimationService: the structured pipeline
│   │   ├── llm_service.py      -- Free-text prompt building + orchestration
│   │   ├── evaluation.py       -- Static structural scoring of free-text output
│   │   ├── llm_wrapper.py      -- LiteLLM fallback, Instructor, streaming, cost
│   │   └── cache.py            -- Exact-match Redis cache
│   ├── cache/
│   │   └── semantic.py         -- Vector-similarity cache (redisvl + Redis Stack)
│   ├── guardrails/
│   │   ├── input.py            -- Moderation, prompt injection, PII (exception policy)
│   │   └── output.py           -- enforce_scope_response (filter policy)
│   ├── prompts/
│   │   ├── loader.py            -- render_estimation_prompt(request, version)
│   │   └── estimation/v1/       -- system.j2, user.j2, examples.j2
│   ├── context/
│   │   └── examples.py         -- CAG reference examples for the free-text flow
│   └── schemas/
│       ├── estimation.py       -- Structured: Request, Draft, Result, Response
│       └── estimations.py      -- Free-text: transcription in, Markdown + evaluation out
├── streamlit_app.py         -- Streamlit UI: structured form + streaming chat
├── tests/
├── .env.example
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Requirements
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- **Redis Stack** — the semantic cache needs the RediSearch module for vector queries. `docker compose up` starts `redis/redis-stack` automatically. On vanilla `redis:7-alpine` the app still runs, but the semantic cache disables itself at startup (logged as `semantic_cache_disabled`) and only the exact-match layer works.
- At least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — the app validates this at startup and refuses to boot without one. `OPENAI_API_KEY` is additionally required for the moderation guardrail and the semantic cache's embeddings.

## Setup
```bash
uv sync
cp .env.example .env   # then fill in your API key(s)
```

## Running the API
```bash
uv run uvicorn app.main:app --reload
```
Available at `http://127.0.0.1:8000`, interactive docs at `/docs`. Needs Redis reachable at `REDIS_URL` — start one with `docker run -p 6379:6379 redis/redis-stack:7.4.0-v0` if you're not using `docker compose`.

### Running with Docker
```bash
docker compose up --build
```
Starts the API plus a Redis Stack service (RedisInsight UI on port 8001) and wires them together.

## Running the Streamlit UI
```bash
uv run streamlit run streamlit_app.py
```
It talks to the API over HTTP at `ESTIMATOR_API_BASE_URL` and holds no LLM API key. Two tabs: **Structured estimate** (the form) and **Chat** (free-text, streamed). The sidebar shows the configured models and cache TTL, the chat's system prompt and injected CAG examples (read-only, for visibility), and for the last call: response time, whether it was served from cache, the USD cost, and which model answered.

## Environment Variables
See [`.env.example`](.env.example) for the full list. Beyond the API keys and models:

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROMPT_VERSION` | `v1` | Which template under `app/prompts/estimation/` is served |
| `CACHE_TTL` | `86400` | Exact-match cache TTL, in seconds |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for the semantic cache |
| `SEMANTIC_CACHE_THRESHOLD` | `0.85` | Minimum cosine similarity to serve a semantic hit |
| `SEMANTIC_CACHE_LOG_ONLY` | `false` | Log would-be hits without serving them (threshold calibration) |

## Endpoints

| Method | Path                        | Description                                             |
|--------|-----------------------------|---------------------------------------------------------|
| GET    | `/health`                   | Health check                                            |
| POST   | `/api/v1/estimate`          | Structured estimation (typed in, validated schema out)  |
| POST   | `/api/v1/estimate/text`     | Free-text estimation (transcription in, Markdown out)   |
| POST   | `/api/v1/estimate/stream`   | Free-text estimation streamed over Server-Sent Events   |

The structured endpoint is the main one: typed request, validated `EstimationResult`, both cache layers. The two free-text endpoints take a raw meeting transcription and return Markdown — they share the same input guardrails, wrapper and exact-match cache, but not the semantic cache (whose bucket is keyed on the typed form options a free-text request doesn't have). `/estimate/text` also runs a static structural evaluation of the generated Markdown (`app/services/evaluation.py`); `/estimate/stream` is what the Streamlit chat consumes.

### `POST /api/v1/estimate`

**Request body:**
```json
{
  "description": "A small B2B SaaS to manage employee equipment loans across teams, with role-based access for HR and IT.",
  "project_type": "web_saas",
  "detail_level": "medium",
  "output_format": "phases_table"
}
```
`project_type`: `mobile_app` | `web_saas` | `internal_tool` | `data_pipeline`. `detail_level`: `summary` | `medium` | `detailed`. `output_format`: `phases_table` | `line_items` | `narrative`.

**Response:**
```json
{
  "result": {
    "summary": "...",
    "confidence_pct": 80,
    "phases": [
      {"name": "Discovery", "duration_weeks": 1, "cost_eur": 2500, "summary": "Workshops and scoping."}
    ],
    "total_duration_weeks": 9,
    "total_cost_eur": 24500
  },
  "prompt_version": "v1",
  "cached": false,
  "meta": {"model": "gpt-4o-mini", "provider": "openai", "cost_usd": 0.000446, "latency_ms": 2607}
}
```

**Errors:**

| Status | When |
|--------|------|
| `400`  | An input guardrail rejected the request. Body is `{"detail": {"reason": "prompt_injection" \| "pii" \| "moderation", "message": "..."}}` |
| `422`  | The request body failed schema validation (missing field, bad enum, description too short) |
| `502`  | The upstream LLM call failed, including Instructor exhausting its retries |

An off-topic or unsizeable description is **not** an error: it returns `200` with a `summary` starting in `"Out of scope:"`, `confidence_pct` below 30, and a single `Not estimated` phase. The UI renders that as a warning rather than a fake estimate.

### `POST /api/v1/estimate/text` and `/api/v1/estimate/stream`

**Request body** (both): `{"transcription": "Client meeting summary describing the project..."}` — minimum 50 characters.

`/estimate/text` responds with the Markdown estimation plus usage, cost, `cache_hit`, and an `evaluation` object scoring the output's structure (does it have the breakdown table, do the declared totals match the summed rows, was it cut off). `/estimate/stream` responds with `text/event-stream`: an `event: token` per chunk, then `event: meta` (cache hit, cost, model) and `event: done`, or `event: error` if the call fails after retries and fallback.

### Totals are computed in Python, not by the LLM

The model fills an `EstimationDraft` (summary, confidence, phases) and the service sums the phases to produce `total_duration_weeks` / `total_cost_eur`. This is a measured decision, not a stylistic one: asking `gpt-4o-mini` for a grand total alongside the phases failed every one of Instructor's 7 attempts in a live run (~24k tokens burned, drifting further each retry) before the request 502'd. Field ordering and an explicit "compute the sum" instruction were not enough. `EstimationResult.phases_sum_matches_total` is kept as a genuine invariant assertion over the computed value.

### Prompt versions

Prompts live under `app/prompts/estimation/<version>/` (`system.j2`, `user.j2`, `examples.j2`) and are rendered by `app/prompts/loader.py`. The active version is the `PROMPT_VERSION` setting — a deploy-time decision, not a client-supplied parameter, so rolling out a new prompt is a config change and every response echoes back the version that produced it. Each render logs a `prompt_rendered` event with the version and a content hash (not the text, so descriptions stay out of the logs).

## Testing
```bash
uv run pytest
```
The suite is offline: template rendering, schema validators, guardrail regexes, and the endpoint with the service faked out. No API keys or Redis needed.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "description": "A small B2B SaaS to manage employee equipment loans across teams.",
    "project_type": "web_saas",
    "detail_level": "medium",
    "output_format": "phases_table"
  }'
```
