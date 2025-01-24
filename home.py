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


import streamlit as st

# Custom title with font size and style
st.markdown(
    """
    <h1 style='text-align: center; color: #00FFFF; font-family: Arial; font-size: 40px;'>
    MNIST Digit Recognizer
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <h3 style='text-align: center; color: magenta; font-family: Arial; font-size: 30px;'>
    Upload image
    </h3>
    """,
    unsafe_allow_html=True
)




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