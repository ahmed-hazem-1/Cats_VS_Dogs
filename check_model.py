import tensorflow as tf
import os

def check_model_compatibility():
    """Check model compatibility with current TensorFlow version"""
    print(f"TensorFlow version: {tf.__version__}")
    
    model_path = 'model_v2.h5'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' not found!")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully!")
        print(f"Model summary:")
        model.summary()
        print("\nModel input shape:", model.input_shape)
        print("Model output shape:", model.output_shape)
        print("\nModel successfully verified with current TensorFlow version.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible fixes:")
        print("1. Save model in Colab using model.save('model_v2.h5', save_format='h5')")
        print("2. Try converting the model using the TensorFlow SavedModel format")
        print("3. Make sure the TensorFlow version is compatible")

if __name__ == "__main__":
    check_model_compatibility()
