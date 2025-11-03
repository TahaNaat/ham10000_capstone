import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from src.data.dataset import LABEL_ORDER

# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Skin Cancer Classifier", layout="wide")
st.title("🩺 Advanced Skin Lesion Classification with Grad-CAM")
st.write("Upload dermatoscopic images to classify and visualize model attention regions (HAM10000 dataset).")

# -------------------------
# Sidebar Configuration
# -------------------------
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Show warning if prediction confidence is below this threshold"
)

show_all_predictions = st.sidebar.checkbox(
    "Show All Class Probabilities",
    value=False,
    help="Display prediction scores for all classes"
)

comparison_mode = st.sidebar.checkbox(
    "Enable Multi-Image Comparison",
    value=False,
    help="Upload and compare predictions for multiple images"
)

st.sidebar.markdown("---")
st.sidebar.info("**Model:** ResNet50 (Fine-tuned)\n\n**Classes:** 7 skin lesion types")

# -------------------------
# Model loading
# -------------------------
@st.cache_resource
def load_model(model_path="runs/resnet50_colab_long/best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_ORDER))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model, device

# Load model with progress
with st.spinner("🔄 Loading model..."):
    model, device = load_model()
    st.sidebar.success(f"✅ Model loaded on: **{device.upper()}**")

# -------------------------
# Image preprocessing
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------
# Helper Functions
# -------------------------
def validate_image(image_file):
    """Validate uploaded image"""
    try:
        img = Image.open(image_file).convert("RGB")
        # Check image dimensions
        if img.size[0] < 50 or img.size[1] < 50:
            return None, "Image too small (minimum 50x50 pixels)"
        # Check file size (max 10MB)
        image_file.seek(0, 2)  # Seek to end
        size = image_file.tell()
        image_file.seek(0)  # Reset
        if size > 10 * 1024 * 1024:
            return None, "Image too large (maximum 10MB)"
        return img, None
    except Exception as e:
        return None, f"Invalid image file: {str(e)}"

def process_image(image, image_name="Image"):
    """Process a single image and return results"""
    try:
        # Preprocess
        with st.spinner(f"🔍 Analyzing {image_name}..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                pred_idx = output.argmax(1).item()
                pred_label = LABEL_ORDER[pred_idx]
                confidence = probabilities[pred_idx].item()
            
            # Grad-CAM
            cam_extractor = SmoothGradCAMpp(model, target_layer='layer4')
            _ = model(input_tensor)
            activation_map = cam_extractor(pred_idx, _)[0].squeeze().cpu().numpy()
            
            # Overlay
            img_resized = image.resize((224, 224))
            gradcam_overlay = overlay_mask(
                img_resized, 
                Image.fromarray(np.uint8(activation_map * 255), mode='L'), 
                alpha=0.6
            )
            
            return {
                'pred_label': pred_label,
                'confidence': confidence,
                'probabilities': probabilities,
                'pred_idx': pred_idx,
                'gradcam_overlay': gradcam_overlay,
                'original_resized': img_resized
            }
    except Exception as e:
        st.error(f"❌ Error processing {image_name}: {str(e)}")
        return None

def display_results(results, image_name="Image", col_container=None):
    """Display prediction results"""
    if results is None:
        return
    
    container = col_container if col_container else st
    
    # Prediction with confidence check
    confidence_pct = results['confidence'] * 100
    if confidence_pct >= confidence_threshold:
        container.success(f"### 🧠 Predicted: **{results['pred_label'].upper()}**")
    else:
        container.warning(f"### ⚠️ Predicted: **{results['pred_label'].upper()}** (Low Confidence)")
    
    container.markdown(f"**Confidence:** {confidence_pct:.2f}%")
    
    # Progress bar for confidence
    container.progress(results['confidence'])
    
    # Side-by-side comparison
    container.markdown("---")
    container.subheader("📊 Visual Analysis")
    
    col1, col2 = container.columns(2)
    with col1:
        st.markdown("**Original Image**")
        st.image(results['original_resized'], use_container_width=True)
    
    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        st.image(results['gradcam_overlay'], use_container_width=True)
    
    # All class probabilities
    if show_all_predictions:
        container.markdown("---")
        container.subheader("📈 All Class Probabilities")
        prob_data = {
            label: f"{results['probabilities'][i].item() * 100:.2f}%" 
            for i, label in enumerate(LABEL_ORDER)
        }
        # Sort by probability
        sorted_probs = sorted(
            [(label, results['probabilities'][i].item()) for i, label in enumerate(LABEL_ORDER)],
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in sorted_probs:
            container.write(f"**{label.upper()}:** {prob * 100:.2f}%")
            container.progress(prob)

# -------------------------
# Upload Section
# -------------------------
st.markdown("---")

if comparison_mode:
    st.header("🔄 Multi-Image Comparison Mode")
    uploaded_files = st.file_uploader(
        "📸 Upload multiple dermatoscopic images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload 2-4 images to compare predictions"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        if len(uploaded_files) > 4:
            st.warning("⚠️ Please upload maximum 4 images for comparison")
            uploaded_files = uploaded_files[:4]
        
        # Process all images
        cols = st.columns(min(len(uploaded_files), 2))
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col_idx = idx % 2
            with cols[col_idx]:
                st.markdown(f"### Image {idx + 1}: {uploaded_file.name}")
                
                # Validate
                image, error = validate_image(uploaded_file)
                if error:
                    st.error(f"❌ {error}")
                    continue
                
                # Display original
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                
                # Process and display results
                results = process_image(image, f"Image {idx + 1}")
                if results:
                    display_results(results, f"Image {idx + 1}", st)
                
                st.markdown("---")

else:
    st.header("📤 Single Image Analysis")
    uploaded_file = st.file_uploader(
        "📸 Upload a dermatoscopic image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG (max 10MB)"
    )
    
    if uploaded_file is not None:
        # Validate image
        image, error = validate_image(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Please upload a valid image file (JPG, PNG) with minimum dimensions of 50x50 pixels and maximum size of 10MB.")
        else:
            # Display original
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process and display results
            results = process_image(image, uploaded_file.name)
            if results:
                display_results(results, uploaded_file.name)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>HAM10000 Skin Lesion Classes:</strong></p>
        <p>MEL (Melanoma) | NV (Melanocytic Nevus) | BCC (Basal Cell Carcinoma) | 
        AKIEC (Actinic Keratosis) | BKL (Benign Keratosis) | DF (Dermatofibroma) | VASC (Vascular Lesion)</p>
    </div>
""", unsafe_allow_html=True)
st.caption("⚕️ Trained on HAM10000 dataset — Samsung Innovation Camp AI Project")
st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Always consult a healthcare professional for medical diagnosis.")
