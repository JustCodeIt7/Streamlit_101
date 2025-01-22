# Import necessary libraries
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

# Load the MobileNetV2 model
model = MobileNetV2(weights="imagenet")


# Function to load and preprocess the image
def load_image(img):
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    return img_array


# Function to make predictions
def make_prediction(img_array):
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return predictions


# Streamlit app
st.title("Image Recognition App")
st.header("Upload an image to classify it")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an Image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess and make prediction
    with st.spinner("Processing..."):
        img_array = load_image(img)
        predictions = make_prediction(img_array)

    # Get the top prediction
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display predictions
    st.subheader("Predictions:")
    for i, (imagenetID, label, prob) in enumerate(decoded_predictions):
        st.write(f"{i+1}. {label}: {prob * 100:.2f}%")
else:
    st.warning("Please upload an image file to classify.")
