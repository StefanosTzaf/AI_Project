import gradio as gr
import os
import tempfile
from infer import infer  # your existing function
import cv2
from types import SimpleNamespace

def run_inference(image):
    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_path = tmp.name
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    
    args = SimpleNamespace(config_path='config.yaml', evaluate=False, infer_samples=True, path=image_path)
    output_path=infer(args)
    print(output_path)

    # Load result image (assume saved to same filename in 'outputs/')
    
    if os.path.exists(output_path):
        result = cv2.imread(output_path)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    else:
        return "Error: Output image not found."

# Gradio UI
iface = gr.Interface(
    fn=run_inference,
    inputs=gr.Image(type="numpy",sources="upload"),
    outputs=gr.Image(type="numpy"),
    title="X_Ray inference app",
    description="Upload an image and get the inference result."
)

if __name__ == "__main__":
    iface.launch()
