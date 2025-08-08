from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
import io
import uuid
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Load the MNIST model
MODEL_PATH = 'models'
model = None

def load_mnist_model():
    global model
    try:
        h5_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.h5')]
        if not h5_files:
            print("Warning: No .h5 model file found in model folder")
            return None
        
        model_file = h5_files[0]  # Use the first .h5 file found
        model_path = os.path.join(MODEL_PATH, model_file)
        model = keras.models.load_model(model_path)
        print(f"MNIST model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading MNIST model: {e}")
        return None

def preprocess_image_for_mnist(image):
    """
    Preprocess the image for MNIST model prediction
    - Convert to grayscale
    - Pad to square if needed
    - Resize to 28x28
    - Normalize pixel values
    """
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Pad to square
        width, height = image.size
        if width != height:
            # Find the longer side
            max_side = max(width, height)
            # Create a new square image (black background)
            new_image = Image.new('L', (max_side, max_side), 0)
            # Paste the original image centered
            paste_x = (max_side - width) // 2
            paste_y = (max_side - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image

        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(image)
        img_array = img_array.astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def predict_digit(image):
    """
    Predict the digit using the loaded MNIST model
    """
    if model is None:
        print("Model not loaded")
        return None, None
    
    try:
        # Preprocess the image
        processed_image = preprocess_image_for_mnist(image)
        if processed_image is None:
            return None, None
        
        predictions = model.predict(processed_image, verbose=0)
        
        predicted_digit = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return predicted_digit, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Load the model at startup
load_mnist_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()
        
        image_data = data['image']
        
        if image_data.startswith('data:image/png;base64,'):
            image_data = image_data[22:]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        predicted_digit, confidence = predict_digit(image)
        
        return jsonify({
            'success': True,
            'digit': int(predicted_digit) if predicted_digit is not None else None,
            'confidence': float(confidence) if confidence is not None else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
