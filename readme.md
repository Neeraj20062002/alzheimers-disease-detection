# 🧠 Alzheimer’s Disease Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green)
![Status](https://img.shields.io/badge/Project-Active-success)

---

## 🧩 Project Overview

Alzheimer’s Disease is one of the most common neurodegenerative disorders affecting millions worldwide.  
This project leverages **Convolutional Neural Networks (CNN)** to automatically detect and classify different stages of Alzheimer’s Disease from **MRI brain scans**.

A **Tkinter-based Graphical User Interface (GUI)** allows users to upload MRI images, run the trained deep learning model, and visualize the prediction results interactively.

> 💡 *Goal:* Assist medical professionals and researchers by providing an AI-powered tool for early detection and stage classification of Alzheimer’s Disease.

---

## 🚀 Key Features

✅ **Automated Detection:** Classifies MRI brain scans into disease stages (e.g., Mild, Moderate, Severe).  
✅ **Deep Learning Model:** CNN architecture trained on preprocessed MRI data.  
✅ **Interactive GUI:** Simple Tkinter interface for image upload and prediction.  
✅ **Multi-View Support:** Handles Axial, Coronal, and Sagittal views of MRI scans.  
✅ **Evaluation Metrics:** Displays model performance — Accuracy, Precision, Recall, and F1-Score.  
✅ **Lightweight Deployment:** Works on any system with Python and TensorFlow installed.

---

## 🧠 Model Architecture

The model follows a **Convolutional Neural Network (CNN)** pipeline:

1. **Input Layer:** Preprocessed MRI images (resized and normalized).  
2. **Conv2D Layers:** Feature extraction using multiple convolution + ReLU blocks.  
3. **Pooling Layers:** Spatial dimensionality reduction.  
4. **Flatten & Dense Layers:** Fully connected neural network for classification.  
5. **Output Layer:** Softmax activation for multi-class stage prediction.

🧪 *Optimizer:* Adam  
🎯 *Loss Function:* Categorical Crossentropy  
📈 *Metrics:* Accuracy, Precision, Recall  

---

## 📊 Dataset Details

- **Source:** Kaggle – Alzheimer’s MRI Dataset  
- **Classes:**  
  - Mild Demented  
  - Moderate Demented  
  - Non-Demented  
  - Very Mild Demented  
- **Size:** ~6,400 MRI images  
- **Preprocessing:**  
  - Normalization and resizing  
  - Augmentation (rotation, zoom, shear) for generalization

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow · Keras |
| **Data Handling** | NumPy · Pandas |
| **Image Processing** | OpenCV |
| **Visualization** | Matplotlib |
| **GUI** | Tkinter |
| **Environment** | Virtualenv / Anaconda |

---

## ⚙️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/Neeraj20062002/alzheimers-disease-detection.git
cd alzheimers-disease-detection

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python Main.py

---

## 🔮 Future Improvements

- Add model explainability with Grad-CAM visualizations.
- Improve GUI design with modern themes.
- Integrate cloud-based model serving (TensorFlow Lite / ONNX).

---

### 📅 Update Log
- **Oct 25, 2025:** Added future improvements section and updated documentation.

