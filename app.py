import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('dogs_vs_cats_model.keras')

st.title("Dog vs Cat Image Classifier")
st.header("Upload an image to classify if it's a dog or a cat")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
  image = Image.open(uploaded_file).resize((256, 256))
  img_array = np.array(image)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0

  st.image(image, caption='Uploaded Image.', use_column_width=True)

  prediction = model.predict(img_array)

  if prediction[0][0] > 0.5:
    st.write("Prediction: Dog")
  else:
    st.write("Prediction: Cat")