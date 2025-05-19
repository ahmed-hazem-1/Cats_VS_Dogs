# Cat vs Dog Classifier App

This Streamlit app allows you to upload an image of a cat or dog and uses a trained CNN model to predict which animal it is.

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the model file (`model_v2.h5`) is in the same directory as the app.py file.

3. Run the Streamlit app:
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

## Model Architecture

The model uses a deep CNN architecture with several convolutional layers, batch normalization, max pooling, and dropout for regularization.
