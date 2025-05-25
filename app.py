import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Dog vs Cat Classifier",
    layout="centered",
)

st.title("Dog vs Cat Image Classifier")
st.markdown(
    "Upload an image and the model will classify it as either a **dog** or a **cat**, "
    "along with the prediction confidence."
)

# --------------------------------------------------
# Sidebar – upload + about
# --------------------------------------------------
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a JPG / JPEG / PNG file", type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Load the model (cached for efficiency)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogs_vs_cats_model.keras")

model = load_model()

# --------------------------------------------------
# Helper — preprocess image
# --------------------------------------------------
def preprocess(pil_image: Image.Image) -> np.ndarray:
    pil_image = pil_image.convert("RGB").resize((256, 256))
    arr = keras_image.img_to_array(pil_image) / 255.0
    return np.expand_dims(arr, axis=0)

# --------------------------------------------------
# Main logic
# --------------------------------------------------
if uploaded_file:
    with st.spinner("Analyzing image..."):
        pil_img = Image.open(uploaded_file)
        tensor = preprocess(pil_img)
        pred = model.predict(tensor)[0][0]
        label = "Dog" if pred > 0.5 else "Cat"
        confidence = pred if label == "Dog" else 1 - pred

    # Layout: image and result
    col_img, col_res = st.columns([2, 1], gap="large")

    with col_img:
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)


    with col_res:
        st.subheader(f"Prediction: {label}")
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(confidence))  
else:
    st.info("Upload an image from the sidebar to get started.")
