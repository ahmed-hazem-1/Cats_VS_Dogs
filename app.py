import streamlit as st
import numpy as np
from PIL import Image
import io
import os

# Set page configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱🐶")

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    if not TENSORFLOW_AVAILABLE:
        st.error("TensorFlow is not available in this environment. This is a demo mode.")
        return None
        
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

if not TENSORFLOW_AVAILABLE:
    st.warning("""
    ⚠️ TensorFlow is not available in this environment. 
    
    This app is running in demo mode with limited functionality.
    
    The app requires TensorFlow which is not compatible with the current Python version on Streamlit Cloud.
    
    Please visit the GitHub repository to download and run the app locally for full functionality.
    """)
    
    # Display sample images
    st.subheader("Sample Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://i.imgur.com/JiEcPh6.jpg", caption="Sample Cat Image")
        st.success("Cat (Confidence: 95%)")
    with col2:
        st.image("https://i.imgur.com/KYdSbYf.jpg", caption="Sample Dog Image")
        st.success("Dog (Confidence: 92%)")
        
    st.subheader("How to Run Locally")
    st.code("""
    # Clone the repository
    git clone https://github.com/your-username/cats_vs_dogs.git
    
    # Install dependencies
    pip install tensorflow numpy pillow streamlit
    
    # Run the app
    streamlit run app.py
    """)
    
else:
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
                try:
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
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("Model could not be loaded. Please check the logs for details.")

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

# Add troubleshooting info
st.sidebar.title("Troubleshooting")
st.sidebar.info(
    """
    If you encounter issues with the model loading:
    
    1. Make sure the model file (model_v2.h5) is in the same directory as the app.py file
    2. Check that TensorFlow is properly installed
    3. For models created in Google Colab:
       - Ensure the TensorFlow version is compatible (Python 3.10 or lower recommended)
       - Consider downloading the app and running it locally
    4. Try refreshing the page
    """
)
