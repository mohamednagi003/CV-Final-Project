import os
import sys

print("🟡 Current Working Directory:", os.getcwd())
print("🟡 Directory Contents:", os.listdir())
print("🟡 Python Path:", sys.path)


import streamlit as st
from PIL import Image
#from models.classifier import load_classification_model, classify_image
from models.object_detector import load_yolo_model, detect_objects
import tempfile

# Load models
classifier_model = load_classification_model()
yolo_model = load_yolo_model()

st.title("Image Classification & Object Detection App")
option = st.selectbox("Choose Task", ["Image Classification", "Object Detection"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if option == "Image Classification":
        prediction = classify_image(classifier_model, image)
        st.write(f"Predicted Class: {prediction}")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            result_image = detect_objects(yolo_model, tmp.name)
            st.image(result_image, caption="Detected Objects", use_column_width=True)
