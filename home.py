import streamlit as st
import numpy as np
from PIL import Image
import pickle as pkl

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    # Open the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = np.array(img)  # Convert to NumPy array
    img_array = 255 - img_array  # Invert colors (MNIST uses white digits on black background)
    img_array = img_array / 255.0  # Normalize to range [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dimensions
    return img_array

with open("cnn_model.pkl", "rb") as file:
        model = pkl.load(file)

st.title("MNIST Digit Recogonizer")
st.subheader("Upload image")
img = st.file_uploader("1", label_visibility="collapsed")
if img is not None:
    st.image(img,width=128)
if st.button("Submit"):
    if img is not None:
        img = preprocess_image(img)
        prediction = model.predict(img)
        predict = int(np.argmax(prediction))
        st.subheader(f'Prediction: {predict}')
    else:
        st.subheader(f'Prediction: Null')