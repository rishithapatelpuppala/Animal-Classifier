import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("animal_classifier_model.h5")

class_labels = [
    "Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant",
    "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

st.title("Animal Image Classifier 🐾")
st.write("Upload an image of an animal to classify it into one of 15 classes.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    predicted_class = class_labels[np.argmax(pred)]
    confidence = np.max(pred)

    st.success(f"Prediction: **{predicted_class}** ({confidence * 100:.2f}% confidence)")
