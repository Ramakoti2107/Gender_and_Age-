import streamlit as st
from deepface import DeepFace

st.title("Age and Gender Prediction")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    try:
        with st.spinner("Analyzing..."):
            result = DeepFace.analyze(
                img_path="temp.jpg",
                actions=["age", "gender"],
                detector_backend="opencv",   # cloud-safe backend
                enforce_detection=False      # avoids crash if face not detected
            )

        # DeepFace may return list — pick first element
        if isinstance(result, list):
            result = result[0]

        st.success(f"Predicted Age: {result['age']}")
        st.success(f"Predicted Gender: {result['dominant_gender']}")

    except Exception as e:
        st.error("Prediction failed. Please upload a clear face image.")
