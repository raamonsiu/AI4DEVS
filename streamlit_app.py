import time
import streamlit as st
from openai import OpenAI
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples
from app.services.llm_service import build_system_prompt

MODEL = "gpt-4o-mini"

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
system_prompt = build_system_prompt()
examples_text = format_examples(ESTIMATION_EXAMPLES)

def stream_response_text(stream, usage_holder):
    """Yield text deltas and stash the final usage chunk into usage_holder."""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
        if chunk.usage:
            usage_holder["usage"] = chunk.usage

with st.sidebar:
    st.header("CAG Context")

    with st.expander("System Prompt"):
        st.text_area("System Prompt", value=system_prompt, height=300, disabled=True, label_visibility="collapsed")

    with st.expander("Injected Examples"):
        st.text_area("Injected Examples", value=examples_text, height=300, disabled=True, label_visibility="collapsed")

    st.header("Last Call Metrics")
    last_call = st.session_state.get("last_call")
    if last_call:
        st.metric("Model", last_call["model"])
        st.metric("Input tokens", last_call["input_tokens"])
        st.metric("Output tokens", last_call["output_tokens"])
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

    # Generate and show answer (streaming)
    with st.chat_message("assistant"):
        usage_holder = {}
        start_time = time.time()
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        response = st.write_stream(stream_response_text(stream, usage_holder))
        elapsed = round(time.time() - start_time, 2)

    st.session_state.messages.append({"role": "assistant", "content": response})

    if "usage" in usage_holder:
        st.session_state.last_call = {
            "model": MODEL,
            "input_tokens": usage_holder["usage"].prompt_tokens,
            "output_tokens": usage_holder["usage"].completion_tokens,
            "elapsed": elapsed,
        }
        st.rerun()