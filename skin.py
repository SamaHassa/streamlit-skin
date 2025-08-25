import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ---------------------------
# Load the YOLOv8 model
# ---------------------------
@st.cache_resource  # cache model so it doesn’t reload every time
def load_model():
    try:
        model = YOLO("best_skin.pt")  # make sure this file is in the same folder
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_model()

# ---------------------------
# Streamlit App
# ---------------------------
st.title("🩺 Skin Cancer Classification Demo")
st.write("Upload a dermoscopy image and the model will predict the cancer type.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open and show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Running inference...")

    # Run prediction
    results = model(np.array(image))  # inference
    result = results[0]

    # Check if classification output exists
    if result.probs is not None:
        top_idx = result.probs.top1
        class_name = result.names[top_idx]
        confidence = float(result.probs.top1conf)

        st.success(f"**Prediction:** {class_name} ({confidence * 100:.2f}%)")

        # Optionally show top-5 predictions
        st.subheader("Top-5 Predictions")
        for idx in result.probs.top5:
            st.write(f"- {result.names[idx]}: {result.probs.data[idx].item() * 100:.2f}%")

    else:
        st.error("⚠️ This model does not return classification results. Please check if `best_skin.pt` is a classification model.")
