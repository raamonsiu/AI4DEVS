import streamlit as st
from openai import OpenAI
from app.config import get_settings
from app.services.llm_service import build_system_prompt

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
system_prompt = build_system_prompt()

def stream_response_text(stream):
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

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
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(stream_response_text(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})