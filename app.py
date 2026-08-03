import gradio as gr
from main import app as fastapi_app

# Create a simple UI or health check interface
demo = gr.Interface(
    fn=lambda name: f"Interview Analyzer API is running!",
    inputs="text",
    outputs="text",
    title="Interview Analyzer API",
    description="Your FastAPI backend is active and ready to accept requests."
)

# Mount FastAPI app into Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/")