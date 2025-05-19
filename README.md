# Cat vs Dog Classifier

This Streamlit app allows you to upload an image of a cat or dog and uses a trained CNN model to predict which animal it is.

## Online Demo

The online demo on Streamlit Cloud runs in a limited mode because TensorFlow is not compatible with the Python 3.13 environment used by Streamlit Cloud.

## Local Setup

For full functionality with model predictions, run this app locally:

1. Clone this repository:
   ```
   git clone https://github.com/your-username/cats_vs_dogs.git
   cd cats_vs_dogs
   ```

2. Install the required dependencies:
   ```
   pip install streamlit tensorflow numpy pillow
   ```

3. Make sure the model file (`model_v2.h5`) is in the same directory as the app.py file.

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## How It Works

1. The app loads a pre-trained CNN model that was trained on the Microsoft Cats vs Dogs dataset.
2. When you upload an image, it is:
   - Converted to grayscale
   - Resized to 128x128 pixels
   - Normalized
   - Passed through the model for prediction
3. The app displays the prediction result along with the confidence level.

## Model Information

- The model was created in Google Colab using TensorFlow
- CNN architecture with multiple convolutional layers
- Trained on grayscale images (128x128 pixels)
- Binary classification: cat vs dog
