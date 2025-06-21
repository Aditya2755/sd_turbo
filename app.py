import os
import torch
from diffusers import AutoPipelineForText2Image
import gradio as gr

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_ID = os.getenv("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional HuggingFace token

# -----------------------------------------------------------------------------
# Load model (one-time at startup)
# -----------------------------------------------------------------------------

def load_pipeline():
    print(f"Loading {MODEL_ID} on {DEVICE} ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        variant="fp16",
        use_auth_token=HF_TOKEN,
    )
    pipe.to(DEVICE)

    # Memory efficient attention (needs xformers)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    # Optional: compile UNet on PyTorch 2.x for speedup
    try:
        if torch.__version__.startswith("2") and DEVICE == "cuda":
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as err:
        print(f"UNet compile skipped: {err}")

    return pipe

pipe = load_pipeline()

# -----------------------------------------------------------------------------
# Inference function
# -----------------------------------------------------------------------------

def generate(
    prompt: str,
    steps: int = 1,
    seed: int = -1,
    width: int = 512,
    height: int = 512,
):
    """Generate an image with SDXL-Turbo."""
    if seed is None:
        seed = -1
    generator = None
    if seed != -1:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=0.0,  # always 0.0 for Turbo
        height=height,
        width=width,
        generator=generator,
    )
    return result.images[0]

# -----------------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------------

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="a cinematic shot of a baby raccoon wearing an intricate italian priest robe"),
        gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Inference Steps"),
        gr.Number(value=-1, label="Seed (-1 for random)"),
        gr.Dropdown(values=[512, 768, 1024], value=512, label="Width"),
        gr.Dropdown(values=[512, 768, 1024], value=512, label="Height"),
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="SDXL-Turbo (Real-Time Text-to-Image)",
    description="Fast 1-4 step image synthesis using Stability AI's SDXL-Turbo.",
)

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # listen on all interfaces
        server_port=int(os.getenv("PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        inbrowser=False,
    ) 