# Streamlit chat UI for the estimator.
#
# Streamlit is a pure HTTP client of the FastAPI service now: it POSTs to
# /api/v1/estimate/stream and renders the Server-Sent Events chunks live.
# It no longer holds an LLM API key or calls OpenAI/Anthropic directly : the
# API owns the cache, the provider fallback and the system prompt dispatch.
# The system prompt and CAG examples shown in the sidebar are built locally
# purely for display (no LLM call involved), from the same functions the API
# uses, so what you see is guaranteed to match what the API actually sends.

import time
import httpx
import streamlit as st
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples
from app.services.llm_service import build_system_prompt

settings = get_settings()
STREAM_ENDPOINT = f"{settings.ESTIMATOR_API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"

system_prompt = build_system_prompt()
examples_text = format_examples(ESTIMATION_EXAMPLES)

def stream_estimation(transcription: str):
    """POST to the SSE endpoint and yield text chunks as they arrive.

    Per the SSE spec, a message can carry multiple `data:` lines that must be
    joined with `\\n` to reconstruct the payload, and a blank line terminates
    the message. Line endings vary by server (`\\n` vs `\\r\\n`), so both are
    accepted.
    """
    payload = {"transcription": transcription}
    with httpx.stream(
        "POST",
        STREAM_ENDPOINT,
        json=payload,
        timeout=httpx.Timeout(120.0, connect=10.0),
        headers={"Accept": "text/event-stream"},
    ) as response:
        response.raise_for_status()
        current_event = "token"
        data_lines = []
        for raw_line in response.iter_lines():
            if raw_line == "":
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    if current_event == "token":
                        yield payload_text
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
        st.metric("Response time (s)", last_call["elapsed"])
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
if prompt := st.chat_input("Type your message: "):
    # Show users message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show answer (streaming from the FastAPI SSE endpoint)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        try:
            for chunk in stream_estimation(prompt):
                full_response += chunk
                placeholder.markdown(full_response + "▍")
            placeholder.markdown(full_response)
        except httpx.HTTPError as exc:
            full_response = f"Could not reach the estimator at `{STREAM_ENDPOINT}`: {exc}"
            placeholder.error(full_response)
        elapsed = round(time.time() - start_time, 2)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.last_call = {"elapsed": elapsed}
    st.rerun()
