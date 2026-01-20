import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# ---------------- Load Model & Scaler ----------------
model = load_model('trained_model.keras')

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------- App Config ----------------
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🩺 Breast Cancer Detection App")
st.write("Predict whether a tumor is **Benign** or **Malignant** based on input features.")

st.sidebar.header("About Dataset")

st.sidebar.write("""
**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)

This dataset is used for detecting breast cancer based on measurements extracted
from digitized images of Fine Needle Aspirate (FNA) of breast masses.

**Total Samples:** 569  
**Total Features:** 30 numerical features  
**Target Variable:** Diagnosis of tumor

**Target Classes:**
- 🟢 **Benign (0)** – Non-cancerous tumor  
- 🔴 **Malignant (1)** – Cancerous tumor  

---

### 🔍 Feature Description
Each tumor is described using 30 real-valued features, which are grouped into
three main categories:

**1. Mean Features**
Average value of tumor characteristics such as:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal Dimension

**2. Error Features**
Standard deviation (variation) of each tumor feature, indicating irregularity
and uncertainty in measurements.

**3. Worst Features**
The largest (worst) observed value of each feature, representing the most severe
characteristics of the tumor.

---

### ⚙️ Data Preprocessing
- Target labels converted into numerical format  
- Feature scaling applied using **StandardScaler**  
- Dataset split into training and testing sets  
- Scaled features used as input to the Deep Learning model  

---

### 🧠 Model Information
- **Model Type:** Deep Learning (Keras / TensorFlow)
- **Input:** 30 standardized numerical features
- **Output:** Probability of tumor being Benign or Malignant
- **Prediction Method:** Class with highest probability (Softmax + Argmax)

---

### 🎯 Purpose of This App
This application helps demonstrate how Deep Learning models can assist in
early detection of breast cancer by analyzing tumor characteristics.
It is intended for **educational and research purposes only**.
""")

# ---------------- Feature Inputs ----------------
st.subheader("Please provide the following details")

# Organize 30 features into 3 columns for readability
col1, col2, col3 = st.columns(3)

# Column 1
mean_radius = col1.number_input('Mean Radius', 0.0)
mean_texture = col1.number_input('Mean Texture', 0.0)
mean_perimeter = col1.number_input('Mean Perimeter', 0.0)
mean_area = col1.number_input('Mean Area', 0.0)
mean_smoothness = col1.number_input('Mean Smoothness', 0.0)
mean_compactness = col1.number_input('Mean Compactness', 0.0)
mean_concavity = col1.number_input('Mean Concavity', 0.0)
mean_concave_points = col1.number_input('Mean Concave Points', 0.0)
mean_symmetry = col1.number_input('Mean Symmetry', 0.0)
mean_fractal_dimension = col1.number_input('Mean Fractal Dimension', 0.0)

# Column 2
radius_error = col2.number_input('Radius Error', 0.0)
texture_error = col2.number_input('Texture Error', 0.0)
perimeter_error = col2.number_input('Perimeter Error', 0.0)
area_error = col2.number_input('Area Error', 0.0)
smoothness_error = col2.number_input('Smoothness Error', 0.0)
compactness_error = col2.number_input('Compactness Error', 0.0)
concavity_error = col2.number_input('Concavity Error', 0.0)
concave_points_error = col2.number_input('Concave Points Error', 0.0)
symmetry_error = col2.number_input('Symmetry Error', 0.0)
fractal_dimension_error = col2.number_input('Fractal Dimension Error', 0.0)

# Column 3
worst_radius = col3.number_input('Worst Radius', 0.0)
worst_texture = col3.number_input('Worst Texture', 0.0)
worst_perimeter = col3.number_input('Worst Perimeter', 0.0)
worst_area = col3.number_input('Worst Area', 0.0)
worst_smoothness = col3.number_input('Worst Smoothness', 0.0)
worst_compactness = col3.number_input('Worst Compactness', 0.0)
worst_concavity = col3.number_input('Worst Concavity', 0.0)
worst_concave_points = col3.number_input('Worst Concave Points', 0.0)
worst_symmetry = col3.number_input('Worst Symmetry', 0.0)
worst_fractal_dimension = col3.number_input('Worst Fractal Dimension', 0.0)

# ---------------- Prediction ----------------
if st.button('Predict'):

    # Prepare input array
    user_input = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                            mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
                            mean_symmetry, mean_fractal_dimension, radius_error, texture_error,
                            perimeter_error, area_error, smoothness_error, compactness_error,
                            concavity_error, concave_points_error, symmetry_error, fractal_dimension_error,
                            worst_radius, worst_texture, worst_perimeter, worst_area,
                            worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
                            worst_symmetry, worst_fractal_dimension]])

    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Model prediction
    prediction_prob = model.predict(user_input_scaled)
    predicted_class = np.argmax(prediction_prob, axis=1)[0]

    # Map label
    labels = {0: "Benign", 1: "Malignant"}

    # Display result
    if predicted_class == 1:
        st.error(f"🔴 Malignant Tumor ({prediction_prob[0][1]*100:.2f}% probability)")
    else:
        st.success(f"🟢 Benign Tumor ({prediction_prob[0][0]*100:.2f}% probability)")

    # Display probability bar chart
    st.subheader("Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(['Benign', 'Malignant'], prediction_prob[0], color=['green', 'red'])
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    st.pyplot(fig)