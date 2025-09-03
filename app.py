import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib

# Page config
st.set_page_config(page_title="Psoriasis Detector", layout="centered")

# Inject custom CSS for animation background
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .main {
        background-color: rgba(0, 0, 0, 0.4);
        padding: 2rem;
        border-radius: 15px;
    }

    h1 {
        color: #fff;
        text-align: center;
        text-shadow: 1px 1px 2px black;
    }

    .css-1cpxqw2 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔬 Psoriasis Skin Disease Detection")

# Load trained model
model = joblib.load("psoriasis_model.pkl")

# Upload section
uploaded_file = st.file_uploader("📷 Upload an image of skin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", width=300)

    if st.button("🚀 Predict"):
        img = np.array(image)
        img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).flatten() / 255.0
        result = model.predict([gray])[0]

        if result == 0:
            st.error("⚠️ Psoriasis Detected")
        else:
            st.success("✅ Normal Skin - No Psoriasis")
