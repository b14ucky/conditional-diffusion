import streamlit as st
import matplotlib.pyplot as plt

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input(placeholder="How can I help you today?")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"You said: {prompt}",
        }
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
