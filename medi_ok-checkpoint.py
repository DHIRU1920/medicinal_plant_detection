import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Define constants
img_height, img_width = 150, 150
categories = ['Category1', 'Category2', 'Category3']  # Replace with actual category names

# Load your model (assuming the model was saved in 'model.h5')
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])
model.load_weights('model.h5')  # Ensure the model weights are saved as 'model.h5'

def predict_image_class(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = categories[np.argmax(predictions)]
    
    return predicted_class

st.title('Plant Disease Detection')
st.write('Upload an image of a plant leaf to detect its disease.')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Classifying...")
    img = img.resize((img_height, img_width))
    predicted_class = predict_image_class(img)
    st.write(f'This is the detected plant disease: {predicted_class}')
