import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("fraud_detection_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Merchant Category Mapping
# -----------------------------
merchant_category_map = {
    "Grocery": 0,
    "Electronics": 1,
    "Fuel": 2,
    "Online Services": 3
}

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write("Predict whether a transaction is **Fraudulent** or **Legitimate**")

# -----------------------------
# Sidebar - About Model
# -----------------------------
st.sidebar.title("📌 About the Model")
st.sidebar.markdown("""
### 🔍 Problem Type
**Binary Classification**  
Detect whether a credit card transaction is fraudulent or legitimate.

### 🧠 Model Used
**Artificial Neural Network (ANN)**  
- Input Layer: 8 transaction features  
- Hidden Layers: Dense layers with ReLU activation  
- Output Layer: 1 neuron with **Sigmoid** activation  

### ⚙️ Training Details
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Evaluation Metrics:** Accuracy, Precision, Recall  

### 📊 Why Neural Network?
- Handles complex feature interactions  
- Performs well on non-linear fraud patterns  
- Scales effectively for large transaction data  

### ⚠️ Important Note
This model is trained on historical data and should be used as a **decision-support tool**, not a final authority.
""")

# -----------------------------
# User Inputs
# -----------------------------
st.subheader("🔢 Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

transaction_hour = st.slider(
    "Transaction Hour (0–23)", 0, 23, 12
)

merchant_category_name = st.selectbox(
    "Merchant Category",
    options=list(merchant_category_map.keys())
)
merchant_category = merchant_category_map[merchant_category_name]

foreign_transaction = st.selectbox(
    "Foreign Transaction", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

location_mismatch = st.selectbox(
    "Location Mismatch", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

device_trust_score = st.slider(
    "Device Trust Score", 0.0, 1.0, 0.8
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours", min_value=0, value=3
)

cardholder_age = st.number_input(
    "Cardholder Age", min_value=18, max_value=100, value=35
)

# -----------------------------
# Input Data (8 Features)
# -----------------------------
input_data = np.array([[
    amount,
    transaction_hour,
    merchant_category,
    foreign_transaction,
    location_mismatch,
    device_trust_score,
    velocity_last_24h,
    cardholder_age
]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Fraud"):
    scaled_input = scaler.transform(input_data)
    fraud_prob = model.predict(scaled_input)[0][0]

    st.subheader("🔎 Prediction Result")

    if fraud_prob >= 0.5:
        st.error(f"🚨 **Fraud Detected**\n\nProbability: **{fraud_prob:.2f}**")
    else:
        st.success(f"✅ **Legitimate Transaction**\n\nProbability: **{1 - fraud_prob:.2f}**")
