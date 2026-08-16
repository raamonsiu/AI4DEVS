# Streamlit chat UI for the estimator.
#
# Streamlit is a pure HTTP client of the FastAPI service now: it POSTs to
# /api/v1/estimate/stream and renders the Server-Sent Events chunks live.
# It no longer holds an LLM API key or calls OpenAI/Anthropic directly : the
# API owns the cache, the provider fallback and the system prompt dispatch.
# The system prompt and CAG examples shown in the sidebar are built locally
# purely for display (no LLM call involved), from the same functions the API
# uses, so what you see is guaranteed to match what the API actually sends.

import json
import time
import httpx
import streamlit as st
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples
from app.services.llm_service import build_system_prompt

settings = get_settings()
STREAM_ENDPOINT = f"{settings.ESTIMATOR_API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"

MIN_MESSAGE_LENGTH = 50

system_prompt = build_system_prompt()
examples_text = format_examples(ESTIMATION_EXAMPLES)

def format_rejection_message(detail) -> str:
    """Turn an API rejection into a message fit for the end user.

    - Our own guardrail errors (400/502) already send a plain, user-facing
      string as `detail` : shown as-is.
    - FastAPI's built-in validation errors (422) send a list of technical
      dicts (field/type/msg) : translated into a short, readable sentence.
    """
    if isinstance(detail, list):
        field_names = {"transcription": "your message"}
        parts = []
        for err in detail:
            loc = err.get("loc", [])
            field = field_names.get(loc[-1], loc[-1]) if loc else "your input"
            parts.append(f"{field} : {err.get('msg', 'is invalid')}")
        return "Please check your message: " + "; ".join(parts) + "."
    return str(detail)

def stream_estimation(transcription: str, meta_holder: dict):
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec, a message can carry multiple `data:` lines that must be
    joined with `\\n` to reconstruct the payload, and a blank line terminates
    the message. Line endings vary by server (`\\n` vs `\\r\\n`), so both are
    accepted. The final `meta` event (cache hit, cost, model) is parsed into
    ``meta_holder`` instead of being yielded as chat text.
    """
    payload = {"transcription": transcription}
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        if response.status_code >= 400:
            # Must read the body here, while the stream is still open : once
            # this `with` block exits (e.g. the exception propagating out),
            # httpx closes the underlying stream and reading it afterwards
            # raises StreamClosed. Bake the message into a plain exception
            # instead of re-raising response.raise_for_status()'s.
            response.read()
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(format_rejection_message(detail))
        current_event = "token"
        data_lines = []
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
                data_lines.append(raw_line[6:] if raw_line.startswith("data: ") else raw_line[5:])

with st.sidebar:
    st.header("CAG Context")

    with st.expander("System Prompt"):
        st.text_area("System Prompt", value=system_prompt, height=300, disabled=True, label_visibility="collapsed")

    with st.expander("Injected Examples"):
        st.text_area("Injected Examples", value=examples_text, height=300, disabled=True, label_visibility="collapsed")

    st.header("Service")
    st.code(STREAM_ENDPOINT, language="text")
    st.markdown(f"**Primary model:** `{settings.PRIMARY_MODEL}`")
    st.markdown(f"**Fallback model:** `{settings.FALLBACK_MODEL}`")
    st.markdown(f"**Cache TTL:** `{settings.CACHE_TTL}s`")

    st.header("Last Call Metrics")
    last_call = st.session_state.get("last_call")
    if last_call:
        st.metric("Response time (s)", f"{last_call['elapsed']:.2f}")
        # Costs are tiny (fractions of a cent) : st.metric would otherwise
        # round a raw float like 0.000385 down to "$0.00". Format explicitly.
        st.metric("Estimated cost (USD)", f"${last_call.get('cost_usd', 0.0):.6f}")
        st.metric("Cache hit", "Yes" if last_call.get("cache_hit") else "No")
        if last_call.get("model"):
            st.caption(f"Answered by `{last_call['model']}` ({last_call.get('provider', 'unknown')})")
    else:
        st.caption("No calls made yet.")

# Set Up history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render actual history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input(f"Describe your software project (min {MIN_MESSAGE_LENGTH} characters)..."):
    # Show users message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # Fast client-side check: no point round-tripping to the API (and
        # eventually the LLM) for something we already know is too short.
        if len(prompt) < MIN_MESSAGE_LENGTH:
            full_response = (
                f"Your message is too short ({len(prompt)} characters). Please "
                f"describe your project in at least {MIN_MESSAGE_LENGTH} "
                "characters so there's enough to estimate."
            )
            placeholder.error(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.stop()

        full_response = ""
        meta_holder = {}
        start_time = time.time()
        try:
            for chunk in stream_estimation(prompt, meta_holder):
                full_response += chunk
                placeholder.markdown(full_response + "▍")
            placeholder.markdown(full_response)
        except RuntimeError as exc:
            # The API rejected the request (e.g. 422 validation, 400 guardrail,
            # 502 model failure) : surface its actual detail, not a generic
            # "connection failed" message.
            full_response = str(exc)
            placeholder.error(full_response)
        except httpx.HTTPError as exc:
            full_response = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
            placeholder.error(full_response)
        elapsed = round(time.time() - start_time, 2)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.last_call = {
        "elapsed": elapsed,
        "cost_usd": meta_holder.get("cost_usd", 0.0),
        "cache_hit": meta_holder.get("cache_hit", False),
        "model": meta_holder.get("model"),
        "provider": meta_holder.get("provider"),
    }
    st.rerun()
