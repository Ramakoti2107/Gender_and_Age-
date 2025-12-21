# Install: pip install streamlit deepface

import streamlit as st
from deepface import DeepFace
import cv2

st.title("Age and Gender Prediction")
st.write("Upload a face image and get predictions")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Show image
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    # Predict
    try:
        result = DeepFace.analyze(img_path="temp.jpg", actions=["age", "gender"], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]

        age = result['age']
        gender = result['dominant_gender']

        st.success(f"Predicted Age: {age}")
        st.success(f"Predicted Gender: {gender.capitalize()}")

    except Exception as e:
        st.error(f"Error: {e}")
