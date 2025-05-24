import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

# Set page configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶")

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        # Set memory growth to avoid memory allocation issues
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Check if model file exists
        if os.path.exists('model_v2.h5'):
            model = tf.keras.models.load_model('model_v2.h5')
            return model
        else:
            st.error("Model file 'model_v2.h5' not found.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    # Convert to RGB
    img = image.convert('RGB')
    # Resize to 128x128
    img = img.resize((128, 128))
    # Convert to numpy array
    img_array = np.array(img)
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    # Reshape for model input
    img_array = np.reshape(img_array, (1, 128, 128, 3))
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
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image and make prediction
    processed_image = preprocess_image(image)
    
    with col2:
        st.write("Processing...")
        if model is not None:
            try:
                prediction = model.predict(processed_image)
                
                # Enhanced debugging information
                st.write("Raw prediction shape:", prediction.shape)
                st.write("Raw prediction values:", prediction)
                
                # Check if we need to interpret indices differently
                # Try flipping the interpretation of the indices
                cat_probability = float(prediction[0][0])
                dog_probability = float(prediction[0][1])
                
                st.write("Interpreted as:")
                st.write(f"- Class 0 (Cat) probability: {cat_probability:.4f}")
                st.write(f"- Class 1 (Dog) probability: {dog_probability:.4f}")
                
                # Display results with flipped interpretation
                if dog_probability > cat_probability:
                    result = f"Dog (Confidence: {dog_probability:.2%})"
                    st.success(result)
                else:
                    result = f"Cat (Confidence: {cat_probability:.2%})"
                    st.success(result)
                
                # Show prediction gauge with the higher confidence value
                st.write("Prediction Confidence:")
                confidence = max(cat_probability, dog_probability)
                st.progress(confidence)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.error("Model could not be loaded. Please check the logs for details.")

# Add instructions on how to run the app
st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a CNN model trained on the Microsoft Cats vs Dogs dataset.
    
    The model was trained to distinguish between images of cats and dogs with RGB images.
    
    Upload a picture to see if the model correctly identifies it!
    """
)

# How to run instructions
# st.sidebar.title("How to Run")
# st.sidebar.code("streamlit run app.py")

# # Add troubleshooting info
# st.sidebar.title("Troubleshooting")
# st.sidebar.info(
#     """
#     If you encounter issues with the model loading:
    
#     1. Make sure the model file (model_v2.h5) is in the same directory as the app.py file
#     2. Check that TensorFlow is properly installed
#     3. Try refreshing the page
#     """
# )
