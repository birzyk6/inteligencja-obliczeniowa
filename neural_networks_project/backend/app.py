from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64
import io
from PIL import Image
import cv2
import redis
import json
from model.neural_network import NeuralNetworkVisualizer
from utils.image_processing import preprocess_image
from utils.visualization import create_layer_visualization

app = Flask(__name__)
CORS(app)

# Initialize Redis client
try:
    redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("Connected to Redis")
except:
    print("Redis connection failed, running without cache")
    redis_client = None

# Initialize neural network
nn_visualizer = NeuralNetworkVisualizer()


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Neural Network API is running"})


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    """Get information about the neural network model"""
    try:
        model_info = nn_visualizer.get_model_info()
        return jsonify(model_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict_digit():
    """Predict the digit from a drawn image"""
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1]  # Remove data:image/png;base64, prefix
        image_bytes = base64.b64decode(image_data)

        # Process image
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)

        # Make prediction
        prediction, confidence = nn_visualizer.predict(processed_image)

        response = {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "probabilities": nn_visualizer.get_probabilities(processed_image).tolist(),
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/visualize", methods=["POST"])
def visualize_network():
    """Get detailed visualization of neural network processing"""
    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data["image"].split(",")[1]
        image_bytes = base64.b64decode(image_data)

        # Process image
        image = Image.open(io.BytesIO(image_bytes))
        processed_image = preprocess_image(image)

        # Get layer activations
        layer_outputs = nn_visualizer.get_layer_outputs(processed_image)

        # Create visualization data
        visualization_data = create_layer_visualization(
            layer_outputs, nn_visualizer.model
        )

        # Add prediction info
        prediction, confidence = nn_visualizer.predict(processed_image)
        visualization_data["prediction"] = {
            "digit": int(prediction),
            "confidence": float(confidence),
            "probabilities": nn_visualizer.get_probabilities(processed_image).tolist(),
        }

        return jsonify(visualization_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train-info", methods=["GET"])
def get_training_info():
    """Get information about the training process"""
    try:
        training_info = nn_visualizer.get_training_history()
        return jsonify(training_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
