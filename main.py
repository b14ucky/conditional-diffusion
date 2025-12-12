import os
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = genai.Client(api_key=API_KEY)


def display_chat_history(history: list[dict]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            figure = message.get("figure", None)
            if figure:
                st.pyplot(figure)


def get_contents_from_history(history: list[dict]) -> list[types.Content]:
    contents = []
    for message in history:
        role = message["role"]
        if role == "assistant":
            role = "model"
        content = types.Content(
            role=role, parts=[types.Part.from_text(text=message["content"])]
        )
        contents.append(content)

    return contents


if "history" not in st.session_state:
    st.session_state.history = []

display_chat_history(st.session_state.history)

prompt = st.chat_input(placeholder="Ask anything")

if prompt:
    st.session_state.history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    chat = client.chats.create(
        model="gemini-2.5-flash-lite",
        history=get_contents_from_history(st.session_state.history),  # type: ignore
    )

    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.text for chunk in chat.send_message_stream(prompt)
        )

    # fig = plt.figure()
    # plt.bar(np.arange(10), np.random.rand(10))

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": response,
            # "figure": fig,
        }
    )
