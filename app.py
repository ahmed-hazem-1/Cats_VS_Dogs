import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶")

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = tf.keras.models.load_model('model_v2.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Convert to grayscale
    img = image.convert('L')
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    # Reshape for model input
    img_array = np.reshape(img_array, (1, 128, 128, 1))
    return img_array

# Main app
st.title("Cat vs Dog Classifier 🐱🐶")
st.write("Upload an image of a cat or dog, and the model will predict which one it is.")

# Load model
model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    
    with col2:
        st.write("Processing...")
        if model is not None:
            prediction = model.predict(processed_image)
            confidence = float(prediction[0][0])
            
            # Display results
            if confidence > 0.5:
                result = f"Dog (Confidence: {confidence:.2%})"
                st.success(result)
            else:
                result = f"Cat (Confidence: {(1-confidence):.2%})"
                st.success(result)
            
            # Show prediction gauge
            st.write("Prediction Confidence:")
            st.progress(confidence if confidence > 0.5 else 1-confidence)
        else:
            st.error("Model could not be loaded. Please check if the model file exists.")

# Add instructions on how to run the app
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a CNN model trained on the Microsoft Cats vs Dogs dataset.
    
    The model was trained to distinguish between images of cats and dogs with grayscale images.
    
    Upload a picture to see if the model correctly identifies it!
    """
)

# How to run instructions
st.sidebar.title("How to Run")
st.sidebar.code("streamlit run app.py")
