import os
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from model import LabelEncoder, EMA, UNet

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = genai.Client(api_key=API_KEY)


def display_chat_history(history: list[dict]) -> None:
    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            figure = message.get("figure", None)
            if figure:
                st.pyplot(figure, use_container_width=False)


def get_contents_from_history(history: list[dict]) -> list[types.Content]:
    contents = []
    for message in history:
        role = message["role"]
        if role == "assistant":
            role = "model"
        content = types.Content(
            role=role,
            parts=[types.Part.from_text(text=message["content"])],
        )
        contents.append(content)

    return contents


if "model" not in st.session_state:
    st.session_state.model = EMA(model=UNet())
    st.session_state.model.load("./models/checkpoint.pt")

if "encoder" not in st.session_state:
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    st.session_state.encoder = LabelEncoder(labels)


def generate_cifar_image(label: str) -> Figure:
    print(f"{label = }")
    label_tensor = st.session_state.encoder(label)
    image = st.session_state.model.generate(label=label_tensor, times=None)
    fig = plt.figure(figsize=(2, 2))
    image = image[0].squeeze(0)
    image = image.permute(1, 2, 0).detach().numpy()
    plt.imshow(image)
    plt.axis("off")
    return fig


tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="generate_cifar_image",
            description="Generate CIFAR-10 image",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "One of the 10 valid CIFAR-10 classes",
                    },
                },
                "required": ["label"],
            },
        )
    ]
)


config = types.GenerateContentConfig(
    system_instruction="""You are a virtual assistant that helps the user with image generation.
    If user asks you to generate an image you invoke the diffusion model trained on CIFAR-10 dataset,
    only if the user wants to generate an image from the classes available in the dataset.
    Only then you call the function with appropriate class name.
    If the user asks for something like a car for example, you should pass the correct class name
    to the function, so for the car it would be 'automobile', etc.
    Otherwise you should tell the user, that it is not possible to generate the desired image.
    Make sure to always respond with some kind of a nice reply, like "Here's your image of a cat" """,
    tools=[tool],
)


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
        config=config,
    )

    fig_container = [None]

    def stream_generator():

        for chunk in chat.send_message_stream(prompt):
            if hasattr(chunk, "function_calls"):
                if chunk.function_calls:
                    with st.spinner("Generating..."):
                        fig_container[0] = generate_cifar_image(**chunk.function_calls[0].args)  # type: ignore

            yield chunk.text if chunk.text else ""

    with st.chat_message("assistant"):
        response = st.write_stream(stream_generator())
        if fig_container[0]:
            st.pyplot(fig_container[0], use_container_width=False)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": response,
            "figure": fig_container[0],
        }
    )
