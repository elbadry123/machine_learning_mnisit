import streamlit as st 
from PIL import Image
import numpy as np
import joblib

svm_model = joblib.load('svm_model.pkl')

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')  
    image_array = np.array(image).astype('float32') / 255.0
    image_vector = image_array.flatten()
    return image_vector

st.title("Digit Recognition using SVM")
st.write("Upload an image of a handwritten digit (0-9) and let the model predict it!")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    predicted_label = svm_model.predict([processed_image])

    st.write(f"**Predicted Label:** {predicted_label[0]}")
