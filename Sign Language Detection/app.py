# =============================================================
# 🤟 ASL Sign Language Detection – Streamlit App (Clean + Modern UI)
# =============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from collections import deque, Counter


# =============================================================
# ⚙️ PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="ASL Sign Detector",
    page_icon="🤟",
)


# =============================================================
# 📌 CONSTANTS
# =============================================================
MODEL_PATH = "asl_sign_model.keras"
IMG_SIZE = (224, 224)

CLASSES = [
    'A','B','C','D','E','F','G','H','I',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y'
]


# =============================================================
# 📚 SIDEBAR – PROJECT INFORMATION
# =============================================================
st.sidebar.title("🤟 ASL Detection Project")

st.sidebar.markdown("""
### 📖 About
Real-time **American Sign Language (ASL)** alphabet detection using Deep Learning.

The model predicts hand signs from images.

---

### 📦 Dataset
**Kaggle – ASL Alphabet Dataset**  
24 classes (A–Y excluding J & Z)

---

### 🧠 Model
• MobileNetV2 (Transfer Learning)  
• Pretrained on ImageNet  
• Fine-tuned on ASL dataset  
• Input size: 224 × 224 RGB

---

### 📊 Performance
• Accuracy ≈ **99%**  
• Fast real-time inference  
• Stable predictions using smoothing

---

### 🛠 Tech Stack
NumPy • Matplotlib • Seaborn • Scikit-Learn • TensorFlow • Keras • Streamlit 
""")


# =============================================================
# 🧠 LOAD MODEL (cached)
# =============================================================
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model_cached()


# =============================================================
# 🔄 PREPROCESSING
# =============================================================
def preprocess_frame(frame):
    """Resize + convert for model input"""
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img


# =============================================================
# 🔮 PREDICTION FUNCTION
# =============================================================
def predict_frame(frame):
    """Return predicted class + confidence"""
    x = preprocess_frame(frame)
    preds = model.predict(x, verbose=0)[0]

    idx = int(np.argmax(preds))
    label = CLASSES[idx]
    confidence = float(preds[idx])

    return label, confidence


# =============================================================
# 🎯 MAIN TITLE
# =============================================================
st.title("🤟 ASL Sign Language Detector")
st.caption("Recognize hand signs instantly using Deep Learning")


# =============================================================
# 🎛 MODE SELECTION
# =============================================================
# mode = st.radio(
#     'Choose',
#     ["📤 Upload Image"],
#     horizontal=True
# )


# =============================================================
# 📤 IMAGE UPLOAD MODE
# =============================================================


uploaded_file = st.file_uploader(
    "Upload an image of a hand sign",
    type=["jpg", "jpeg", "png"]
    )

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        preview = image.copy()
        preview.thumbnail((400, 400))
        st.image(preview, caption="Preview", use_container_width=False)

    with col2:
        if st.button("🔍 Predict"):
            frame = np.array(image)
            label, conf = predict_frame(frame)

            st.success(f"Prediction: {label}")
            st.metric("Confidence", f"{conf*100:.2f}%")


# =============================================================
# 🧾 FOOTER
# =============================================================
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow + Streamlit | MobileNetV2 Transfer Learning")
