# Conditional Diffusion (CIFAR-10)

This project implements a **class-conditional diffusion model (DDPM)** trained on the **CIFAR-10** dataset, together with an interactive **Streamlit chat interface** powered by **Google Gemini**.  
Users can request images of CIFAR-10 classes in natural language, and the system generates them using a trained diffusion model.

---

## Features

-   Conditional DDPM trained on CIFAR-10 (32×32 RGB images)
-   U-Net with attention and sinusoidal time embeddings
-   EMA (Exponential Moving Average) for stable sampling
-   Streamlit chat UI
-   Gemini function calling to trigger image generation
-   Supports all 10 CIFAR-10 classes

Supported classes:

```bash
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

---

## Pretrained Model

A trained model checkpoint is available and can be downloaded from Kaggle:

[Conditional Diffusion Pretrained Model](https://www.kaggle.com/models/b14ucky/conditional-diffusion/)

After downloading, create a folder named `models` in the project root (if it doesn't exist) and place the checkpoint there:

```bash
mkdir models
mv checkpoint.pt models/
```

This ensures that the Streamlit app (`main.py`) can correctly load the model for inference.-

## Project Structure

```bash
conditional-diffusion/
├── main.py # Streamlit app + Gemini integration
└── model.py # Diffusion model, UNet, EMA, training utilities
```

---

## How It Works (High-Level)

1. **Training (offline)**

    - A conditional DDPM is trained on CIFAR-10.
    - A U-Net predicts the noise $\epsilon$ given `(x_t, t, class_label)`.
    - EMA weights are maintained for higher-quality sampling.
    - A checkpoint (`checkpoint.pt`) is saved.

2. **Inference (online)**
    - The Streamlit app loads the EMA model from a checkpoint.
    - User chats with a Gemini-powered assistant.
    - If the user requests a valid CIFAR-10 class, Gemini calls:
        ```python
        generate_cifar_image(label="cat")
        ```
    - The diffusion model generates and displays the image.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/b14ucky/conditional-diffusion.git
cd conditional-diffusion
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows
```

### 3. Install dependencies

```bash
pip install torch torchvision streamlit matplotlib tqdm python-dotenv google-genai
```

Make sure your PyTorch installation matches your CUDA setup if using GPU.

---

## Environment Variables

Create a `.env` file in the project root:

```bash
API_KEY=your_google_gemini_api_key
```

This key is required for Gemini chat and function calling.
The API key can be obtained [here](https://aistudio.google.com/api-keys).

---

## Running the App

```bash
streamlit run main.py
```

Then open your browser at:

```bash
http://localhost:8501
```

---

## Using the Chat Interface

Examples of valid prompts:

-   "Generate a cat"
-   "Show me a ship"
-   "I want an image of an airplane"

Examples of invalid prompts:

-   "Generate a dragon"
-   "Make a 4K portrait of a person"

If the requested object is **not part of CIFAR-10**, the assistant will politely refuse.

---

## Model Details

### Architecture

-   **U-Net** with:
    -   Residual blocks
    -   Group normalization
    -   Multi-head self-attention at selected resolutions
-   **Sinusoidal time embeddings**
-   **Class embeddings** injected into residual blocks

### Diffusion

-   **DDPM** with linear $\beta$ schedule
-   1000 diffusion steps
-   Reverse process implemented manually

### EMA

-   EMA decay: 0.9999
-   EMA weights used for sampling

---

## Training the Model (Optional)

You can train the model yourself using `ModelTrainer` in `model.py`.

Example:

```python
from model import ModelTrainer

trainer = ModelTrainer(
    batch_size=64,
    time_steps=1000,
    lr=2e-5,
)

trainer.train(
    n_epochs=75,
    checkpoint_output_path="checkpoint.pt",
)
```

Training CIFAR-10 diffusion models is compute-intensive. A GPU is strongly recommended.

---

## Key Classes & Components

-   `UNet` – Conditional U-Net backbone
-   `ResBlock` – Residual blocks with label conditioning
-   `Attention` – Multi-head self-attention
-   `SinusoidalEmbeddings` – Time-step embeddings
-   `DDPMScheduler` – Noise schedule
-   `EMA` – Exponential Moving Average wrapper
-   `LabelEncoder` – Maps class names → label tensors
-   `ModelTrainer` – Training loop and checkpointing

---

## Limitations

-   Image resolution fixed to 32×32
-   Only CIFAR-10 classes supported
-   Sampling is relatively slow (pure PyTorch DDPM)
-   Not intended for photorealistic generation

---

## License

This project is licensed under the **MIT License**.
