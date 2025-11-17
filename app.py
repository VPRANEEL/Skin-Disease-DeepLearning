
#  Streamlit Skin Disease analyzer


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from sklearn.preprocessing import LabelEncoder
import os


# Configurationstrea

MODEL_PATH = "skin_disease_model.keras"  # Path to your trained model

# Load Model

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_trained_model()


# Streamlit UI

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🩺", layout="centered")

st.title("Skin Disease Analyzer Using Deep Learning")
st.write("Upload a skin image to predict the disease")

st.sidebar.subheader("Instructions :")
st.sidebar.write("1. Upload an Image with good resolution and intensity")
st.sidebar.write("2. Click on upload")

uploaded_file = st.file_uploader("Upload a skin image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Predict
    with st.spinner("🔍 Analyzing image..."):
        preds = model.predict(img_batch)
        confidence = np.max(preds) * 100
        pred_label = LABELS[np.argmax(preds)]

    # Confidence Check

    if confidence < 40:
        st.warning(f"⚠️ Image quality seems low. (Confidence: {confidence:.2f}%)")
        st.info("Please upload a clearer image for better prediction.")
    else:
        st.success(f"✅ Predicted Disease: **{pred_label}**")
        st.write(f"**Model Confidence:** {confidence:.2f}%")
else:
    st.info("⬆️ Please upload an image to start prediction.")

