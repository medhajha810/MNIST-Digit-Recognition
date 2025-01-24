from fastapi import FastAPI
from fastapi import UploadFile, File
import numpy as np
from PIL import Image
import pickle as pkl

app = FastAPI()

# Load the model once at the start
with open("cnn_model.pkl", "rb") as file:
    model = pkl.load(file)

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

@app.post("/tester")
async def predict(file:UploadFile=File(...)):

    # Read the uploaded image
    content = await file.read()
    with open("temp_image.jpg", "wb") as temp_file:
        temp_file.write(content)

    img = preprocess_image("temp_image.jpg")

    with open("cnn_model.pkl", "rb") as file:
        model = pkl.load(file)

    prediction = model.predict(img)
    return {"prediction": int(np.argmax(prediction))}
    