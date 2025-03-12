import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
from model import load_model
from torchvision import transforms
from huggingface_hub import hf_hub_download
import os

port = int(os.environ.get("PORT", 8080))  # Default to 8080 for local testing

# ✅ Load the trained model
# Download the model from Hugging Face
model_path = hf_hub_download(
    repo_id="balaaa6414/BrainTumor-Detection-Using-CNN",  # Your HF repo
    filename="model.pth",
    local_dir="models"  # Save it locally in the models/ folder
)
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError("Failed to load model.")

# ✅ Define class labels and tumor characteristics
labels = {
    "Glioma": {
        "info": "⚠️ Gliomas originate in glial cells and can be aggressive.",
        "basis": "Detected in deep brain regions with irregular shapes."
    },
    "Meningioma": {
        "info": "⚠️ Meningiomas arise from the meninges, the membranes covering the brain.",
        "basis": "Typically well-defined and located in the outer brain layers."
    },
    "Pituitary Tumor": {
        "info": "⚠️ Pituitary tumors affect hormone production and cause various symptoms.",
        "basis": "Found in the pituitary gland at the base of the brain."
    },
    "No Tumor": {
        "info": "✅ No tumor detected. However, consult a doctor for further evaluation.",
        "basis": "No abnormal mass or irregularity detected in the scan."
    }
}

# ✅ Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)

# ✅ Heatmap Generation (Grad-CAM)
def generate_heatmap(image, model):
    image.requires_grad_()
    gradients, activations = [], []

    def hook_function(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.layer4[1].conv2
    backward_hook = target_layer.register_full_backward_hook(hook_function)
    forward_hook = target_layer.register_forward_hook(forward_hook)

    output = model(image)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    output[:, pred_class].backward(retain_graph=True)

    backward_hook.remove()
    forward_hook.remove()

    heatmap = torch.mean(gradients[0], dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap, pred_class

# ✅ Prediction Function
def predict(image):
    image_pil = Image.fromarray(image)
    tensor_image = preprocess_image(image_pil)
    heatmap, pred_class = generate_heatmap(tensor_image, model)
    
    predicted_label = list(labels.keys())[pred_class]
    tumor_info = labels[predicted_label]["info"]
    model_basis = labels[predicted_label]["basis"]
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig_image = np.array(image_pil.resize((224, 224)))[:, :, ::-1]
    overlayed_img = cv2.addWeighted(orig_image, 0.6, heatmap_colored, 0.4, 0)

    return predicted_label, tumor_info, model_basis, overlayed_img

# ✅ Create Gradio UI
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Tumor Information"),
        gr.Textbox(label="Model Basis"),
        gr.Image(label="Heatmap")
    ],
    title="Brain Tumor Detection using CNN",
    description="Upload a brain MRI scan to detect tumors, view the heatmap."
)

iface.launch(server_name="0.0.0.0", server_port=port)
