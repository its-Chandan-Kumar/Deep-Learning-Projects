# 🤟 ASL Sign Language Detection using Deep Learning

Real-time **American Sign Language (ASL) Alphabet Recognition** web app built with **TensorFlow + Streamlit**.

The system detects hand signs from images and predicts the corresponding ASL alphabet letter using a **MobileNetV2 Transfer Learning model**.

---

## 🚀 Live Demo
Deployed on Render  
👉 https://sign-language-detection-soy5.onrender.com/

---

## 📌 Features

✅ Upload hand sign image  
✅ Instant prediction  
✅ Confidence score  
✅ Lightweight & fast inference  
✅ Clean modern UI  
✅ Cloud deployable (Render compatible)  
✅ ~99% accuracy  

---

## 📦 Dataset

**ASL Alphabet Dataset (Kaggle)**  
https://www.kaggle.com/datasets/dorukdemirci/asl-alphabet-dataset

### Dataset Details
- 24 classes
- A–Y (excluding J & Z)
- Thousands of labeled images
- RGB images
- Hand gesture signs

---

## 🧠 Model Architecture

### MobileNetV2 (Transfer Learning)

Why MobileNetV2?
- Lightweight
- Fast inference
- High accuracy
- Perfect for real-time apps

### Training Strategy
- Pretrained on ImageNet
- Transfer learning
- Fine-tuned on ASL dataset
- Input size: **224 × 224 × 3**

### Performance
| Metric | Value |
|-------|-------|
| Accuracy | ~99% |
| Inference | Real-time |
| Model Size | Small |

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow
- OpenCV
- Render (deployment)
