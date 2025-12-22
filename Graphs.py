import streamlit as st
import cv2
import cvlib as cv
from PIL import Image
import numpy as np

st.title("Age and Gender Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    faces, confidences = cv.detect_face(img)

    for face in faces:
        (startX, startY, endX, endY) = face
        face_crop = img[startY:endY, startX:endX]

        gender, confidence = cv.detect_gender(face_crop)

        st.success(f"Gender: {gender[0]} ({confidence[0]*100:.2f}%)")
