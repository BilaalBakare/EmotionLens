import gradio as gr

def passthrough(frame):
    return frame

with gr.Blocks() as demo:
    cam = gr.Image(sources=["webcam"], streaming=True)
    cam.stream(fn=passthrough, inputs=cam, outputs=cam)

demo.launch()