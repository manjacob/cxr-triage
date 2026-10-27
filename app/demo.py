import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys
import os
from monai.transforms import (
    EnsureChannelFirst, ScaleIntensity,
    Resize, RepeatChannel, Compose
)
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.dataset import LABELS

model = build_model(len(LABELS))
model.load_state_dict(torch.load('/workspace/outputs/densenet121_epoch9.pt'))
model.eval()

# Condition explanations dictionary
condition_explanations = {
    'Atelectasis': 'Partial collapse or incomplete expansion of a portion of the lung.',
    'Cardiomegaly': 'An enlarged heart silhouette on the X-ray.',
    'Effusion': 'Extra fluid in the space between the lung and chest wall.',
    'Infiltration': 'A nonspecific hazy or cloudy area in the lung, possibly from infection, inflammation, fluid, or other causes.',
    'Mass': 'A larger abnormal growth or opacity than a nodule.',
    'Nodule': 'A small, usually rounded spot or opacity in the lung; it can have many causes.',
    'Pneumonia': 'Infection or inflammation of lung tissue.',
    'Pneumothorax': 'Air leaking into the space around the lung, potentially causing collapse.',
    'Consolidation': 'Lung airspaces filled with material such as fluid, pus, blood, or cells.',
    'Edema': 'Excess fluid in the lung tissue or airspaces.',
    'Emphysema': 'Damage and enlargement of lung airspaces, reducing efficient airflow.',
    'Fibrosis': 'Scar tissue in the lungs.',
    'Pleural_Thickening': 'Thickening or scarring of the thin lining around the lung.',
    'Hernia': 'Tissue from the abdomen pushing through the diaphragm into the chest area.',
}

# Create a transform pipeline for inference (without LoadImage)
inference_transforms = Compose([
    EnsureChannelFirst(channel_dim='no_channel'),
    ScaleIntensity(),
    Resize((224, 224)),
    RepeatChannel(repeats=3),
])

def normalize_channels(x):
    """Ensure image has exactly 1 channel by taking mean if multiple channels"""
    if len(x.shape) == 3 and x.shape[0] > 1:
        x = np.mean(x, axis=0, keepdims=True)
    return x

def generate_cam(model, input_tensor, target_class_idx):
    """Simple CAM implementation without pytorch-grad-cam"""
    # Forward pass to get feature maps
    features = model.features(input_tensor)
    
    # Global average pooling
    pooled = F.adaptive_avg_pool2d(features, (1, 1))
    pooled = pooled.view(pooled.size(0), -1)
    
    # Get classifier weights (classifier is a single Linear layer)
    weight = model.classifier.weight[target_class_idx]  # Shape: [num_features]
    
    # Generate CAM by weighted sum of feature maps
    cam = torch.zeros(features.shape[2:], dtype=torch.float32)
    for i, feat in enumerate(features[0]):
        cam += weight[i] * feat
    
    # ReLU and normalize
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam.detach().cpu().numpy()

def predict(image):
    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image_np = np.array(image)
    image_np = normalize_channels(image_np)
    
    # Apply transforms
    image_tensor = inference_transforms(image_np)
    
    # Convert MONAI MetaTensor to regular torch tensor if needed
    if hasattr(image_tensor, 'array'):
        image_tensor = torch.from_numpy(image_tensor.array)
    
    input_tensor = image_tensor.unsqueeze(0)
    
    # Run model to get predictions
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0]
    
    # Create label dictionary
    label_dict = {label: float(prob) for label, prob in zip(LABELS, probs)}
    
    # Get top 5 conditions for explanations
    top_5_labels = sorted(label_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    explanations_md = "### Top 5 Predicted Conditions:\n\n"
    for label, prob in top_5_labels:
        explanation = condition_explanations.get(label, "No explanation available.")
        explanations_md += f"**{label}** ({prob:.2%}): {explanation}\n\n"
    
    # Try to generate CAM heatmap
    try:
        # Get the highest probability class
        target_class_idx = torch.argmax(probs).item()
        
        # Generate CAM
        cam = generate_cam(model, input_tensor, target_class_idx)
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(cam, (image.width, image.height))
        
        # Convert original image to RGB for overlay
        image_rgb = np.array(image.convert('RGB'))
        
        # Apply colormap to heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image_rgb, 0.5, heatmap, 0.5, 0)
        
        return label_dict, Image.fromarray(overlay), explanations_md
    except Exception as e:
        print(f"CAM error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original image if CAM fails
        return label_dict, image, explanations_md

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Label(num_top_classes=5), gr.Image(), gr.Markdown()],
    title='Chest X-Ray Triage Demo (research/portfolio use only)',
)
demo.launch(server_name='0.0.0.0', server_port=7860)
