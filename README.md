# MNIST Digit Recognition MLP Interactive App

A simple web application for handwritten digit recognition using the MNIST dataset and a Multi-Layer Perceptron (MLP) model. The app is built with TensorFlow/Keras and Flask, providing an interactive interface for real-time digit prediction.

## Features
- Draw digits on a canvas and get instant predictions
- Uses a trained MLP model on the MNIST dataset
- Real-time confidence score for each prediction
- No images are saved; predictions are made in-memory
- Modern UI

## Project Structure
```
├── app.py                # Main Flask application
├── models/
│   └── mnist_model.h5    # Trained MLP model
├── templates/
│   └── index.html        # Web interface
├── images/
│   └── digits.jpg        # Notebook Image
├── notebook styles/
│   └── styles.py         # Custom styles for notebook
├── README.md             # Project documentation
```

## Getting Started
### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone the repository:
   ```powershell
   git clone <repo-url>
   cd MNIST-Digit-Recognition-MLP-Interactive-App
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the app:
   ```powershell
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000`

## Usage
- Use your mouse to draw a digit on the canvas.
- The app will predict the digit and show the confidence score.
- Clear the canvas to draw a new digit.

## Model
- The MLP model is trained on the MNIST dataset using TensorFlow/Keras.
- The trained model is saved as `models/mnist_model.h5`.

## License
This project is licensed under the MIT License.

## Acknowledgements
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)

# Using a virtual environment (recommended)

To avoid conflicts with other Python projects, it's recommended to use a virtual environment:

1. Create a virtual environment named `env`:
   ```powershell
   python -m venv env
   ```
2. Activate the virtual environment:
   ```powershell
   .\env\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
