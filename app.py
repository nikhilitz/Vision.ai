# app.py
import streamlit as st
from inference.inference import run_inference

st.set_page_config(page_title="Vision.ai", layout="centered")

st.title("👁️ Vision.ai — Image Captioning for the Visually Impaired")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with open("temp_uploaded_image.png", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp_uploaded_image.png", caption="Uploaded Image", use_column_width=True)
    st.write("Generating caption...")

    caption = run_inference("temp_uploaded_image.png")
    st.success(f"🧠 Caption: {caption}")
