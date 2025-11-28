import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input(placeholder="Ask anything")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    fig = plt.figure()
    plt.bar(np.arange(10), np.random.rand(10))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"You said: {prompt}",
            "figure": fig,
        }
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        figure = message.get("figure", None)
        if figure:
            st.pyplot(figure)
