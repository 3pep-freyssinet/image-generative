import gradio as gr
from transformers import StableDiffusionPipeline

# Load the Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move model to GPU if available

# Define the inference function
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Set up the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.inputs.Textbox(label="Enter a text description"),
    outputs=gr.outputs.Image(label="Generated Image"),
    title="Text to Image Generation",
    description="Generate images based on your text descriptions."
)

# Launch the interface
iface.launch()
