"""Streamlit UI for the estimator.

Streamlit acts as a pure HTTP client of the FastAPI service. It holds no LLM API
key and never calls a provider directly — the API owns the guardrails, the
caches, the prompt versioning and the provider fallback. Two tabs, two flows:

- **Structured estimate**: a typed ``EstimationRequest`` to
  ``POST /api/v1/estimate``, rendering the validated ``EstimationResult``.
- **Chat**: a free-text transcription streamed from
  ``POST /api/v1/estimate/stream`` and rendered token by token.

The prompt version is deliberately NOT surfaced: which template the service
runs is a deploy-time decision (``PROMPT_VERSION`` in settings), not something
the person asking for an estimate should have to reason about. It is still
logged and returned in the API response for traceability.
"""

from __future__ import annotations

import json
import time

import httpx
import streamlit as st

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples
from app.schemas.estimation import DetailLevel, OutputFormat, ProjectType
from app.services.llm_service import build_system_prompt

settings = get_settings()
API_BASE = settings.ESTIMATOR_API_BASE_URL.rstrip("/")
ESTIMATE_ENDPOINT = f"{API_BASE}/api/v1/estimate"
STREAM_ENDPOINT = f"{API_BASE}/api/v1/estimate/stream"

MIN_DESCRIPTION_LENGTH = 20
MIN_TRANSCRIPTION_LENGTH = 50

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Describe a software project and get a phase-by-phase estimate with costs, "
    "duration and a confidence score."
)


def humanise(value: str) -> str:
    return value.replace("_", " ").capitalize()


def describe_rejection(payload: object, status_code: int) -> str:
    """Turn an API error body into a message fit for the end user.

    - Guardrail rejections (400) arrive as ``{"reason": ..., "message": ...}``.
    - FastAPI validation errors (422) arrive as a list of technical dicts.
    - Anything else falls back to the raw text.
    """
    if isinstance(payload, dict) and "message" in payload:
        labels = {
            "prompt_injection": "That looks like an attempt to instruct the AI rather "
            "than describe a project.",
            "pii": "Please remove personal data from the description.",
            "moderation": "That content can't be processed.",
        }
        prefix = labels.get(str(payload.get("reason")), "")
        return f"{prefix} {payload['message']}".strip()
    if isinstance(payload, list):
        parts = []
        for err in payload:
            loc = err.get("loc", [])
            field = humanise(str(loc[-1])) if loc else "Input"
            parts.append(f"{field}: {err.get('msg', 'is invalid')}")
        return "Please check the form — " + "; ".join(parts) + "."
    return f"The service returned an error ({status_code})."


def request_estimation(payload: dict) -> dict:
    """POST to the estimate endpoint, raising RuntimeError with a friendly
    message on any rejection (400 guardrail, 422 validation, 502 upstream)."""
    response = httpx.post(
        ESTIMATE_ENDPOINT, json=payload, timeout=httpx.Timeout(180.0, connect=10.0)
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(describe_rejection(detail, response.status_code))
    return response.json()


def stream_estimation(transcription: str, meta_holder: dict):
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec a message can carry multiple ``data:`` lines that must be
    joined with a newline, and a blank line terminates the message. The final
    ``meta`` event (cache hit, cost, model) is parsed into ``meta_holder``
    rather than yielded as chat text.
    """
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json={"transcription": transcription},
        timeout=httpx.Timeout(180.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code >= 400:
            # Read the body here, while the stream is still open: once this
            # `with` block exits httpx closes the stream and reading it
            # afterwards raises StreamClosed. Bake the message into a plain
            # exception rather than re-raising raise_for_status()'s.
            response.read()
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(describe_rejection(detail, response.status_code))

        current_event = "token"
        data_lines: list[str] = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
                    elif current_event == "meta":
                        meta_holder.update(json.loads(payload_text))
                    elif current_event == "error":
                        yield f"\n\n[error] {payload_text}"
                    elif current_event == "done":
                        return
                current_event = "token"
                continue
            if raw_line.startswith("event:"):
                current_event = raw_line[6:].strip()
            elif raw_line.startswith("data:"):
                data_lines.append(
                    raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:]
                )


form_tab, chat_tab = st.tabs(["Structured estimate", "Chat"])

with form_tab:
    with st.form("estimation_form"):
        description = st.text_area(
            "Project description",
            height=200,
            placeholder="Describe the project: goals, key features, constraints…",
            help=f"At least {MIN_DESCRIPTION_LENGTH} characters.",
        )
        project_type = st.selectbox(
            "Project type", options=list(ProjectType), format_func=lambda v: humanise(v.value)
        )
        detail_level = st.radio(
            "Detail level",
            options=list(DetailLevel),
            index=1,
            horizontal=True,
            format_func=lambda v: humanise(v.value),
        )
        output_format = st.selectbox(
            "Output format", options=list(OutputFormat), format_func=lambda v: humanise(v.value)
        )
        submitted = st.form_submit_button("Generate estimation", type="primary")

    if submitted:
        if len(description.strip()) < MIN_DESCRIPTION_LENGTH:
            st.error(
                f"The description is too short ({len(description.strip())} characters). "
                f"Please describe the project in at least {MIN_DESCRIPTION_LENGTH} characters."
            )
        else:
            started = time.perf_counter()
            try:
                with st.spinner("Estimating…"):
                    body = request_estimation(
                        {
                            "description": description.strip(),
                            "project_type": project_type.value,
                            "detail_level": detail_level.value,
                            "output_format": output_format.value,
                        }
                    )
            except RuntimeError as exc:
                st.error(str(exc))
            except httpx.HTTPError as exc:
                st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
            else:
                elapsed = round(time.perf_counter() - started, 2)
                result = body["result"]

                # The service normalises anything it could not size confidently
                # into a summary starting with "Out of scope:" — show that as a
                # warning rather than dressing a non-estimate up as a real one.
                if result["summary"].startswith("Out of scope:"):
                    st.warning(result["summary"])
                else:
                    st.success(result["summary"])

                    left, middle, right = st.columns(3)
                    left.metric("Total duration", f"{result['total_duration_weeks']} weeks")
                    middle.metric("Total cost", f"€{result['total_cost_eur']:,}")
                    right.metric("Confidence", f"{result['confidence_pct']}%")

                    st.subheader("Phases")
                    st.table(
                        [
                            {
                                "Phase": phase["name"],
                                "Duration (weeks)": phase["duration_weeks"],
                                "Cost (EUR)": f"€{phase['cost_eur']:,}",
                                "Detail": phase["summary"],
                            }
                            for phase in result["phases"]
                        ]
                    )

                st.session_state.last_call = {
                    "elapsed": elapsed,
                    "cached": body.get("cached", False),
                    **body.get("meta", {}),
                }

with chat_tab:
    st.caption(
        "Paste a client-meeting transcription and get a free-text estimation "
        "streamed token by token. Same guardrails and cache as the form."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        f"Describe your project (min {MIN_TRANSCRIPTION_LENGTH} characters)…"
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Fast client-side check: no point round-tripping to the API (and
            # eventually the LLM) for something we already know is too short.
            if len(prompt) < MIN_TRANSCRIPTION_LENGTH:
                answer = (
                    f"Your message is too short ({len(prompt)} characters). Please "
                    f"describe your project in at least {MIN_TRANSCRIPTION_LENGTH} "
                    "characters so there's enough to estimate."
                )
                placeholder.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            answer = ""
            meta_holder: dict = {}
            started = time.perf_counter()
            try:
                for chunk in stream_estimation(prompt, meta_holder):
                    answer += chunk
                    placeholder.markdown(answer + "▍")
                placeholder.markdown(answer)
            except RuntimeError as exc:
                answer = str(exc)
                placeholder.error(answer)
            except httpx.HTTPError as exc:
                answer = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
                placeholder.error(answer)
            elapsed = round(time.perf_counter() - started, 2)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_call = {
            "elapsed": elapsed,
            "cached": meta_holder.get("cache_hit", False),
            "cost_usd": meta_holder.get("cost_usd", 0.0),
            "model": meta_holder.get("model"),
            "provider": meta_holder.get("provider"),
        }
        st.rerun()

with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    st.code(STREAM_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{settings.PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{settings.FALLBACK_MODEL}`")
    st.markdown(f"**Cache TTL:** `{settings.CACHE_TTL}s`")

    st.header("CAG Context")
    with st.expander("Chat system prompt"):
        st.text_area(
            "Chat system prompt",
            value=build_system_prompt(),
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
    with st.expander("Injected examples"):
        st.text_area(
            "Injected examples",
            value=format_examples(ESTIMATION_EXAMPLES),
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

    st.header("Last Call")
    last_call = st.session_state.get("last_call")
    if last_call:
        st.metric("Response time (s)", f"{last_call['elapsed']:.2f}")
        st.metric("Served from cache", "Yes" if last_call.get("cached") else "No")
        # Costs are fractions of a cent : st.metric would round 0.000385 to
        # "$0.00", so format explicitly. A cache hit genuinely costs nothing.
        st.metric("Cost (USD)", f"${last_call.get('cost_usd', 0.0):.6f}")
        if last_call.get("model"):
            st.caption(
                f"Answered by `{last_call['model']}` "
                f"({last_call.get('provider', 'unknown')})"
            )
    else:
        st.caption("No estimations yet.")
