# Neural Network Visualization App

A full-stack application that demonstrates how neural networks work by allowing users to draw images and see real-time visualization of how the network processes them for recognition.

## Features

-   **Interactive Drawing Canvas**: Draw digits (0-9) on a canvas
-   **Real-time Neural Network Visualization**: See how data flows through the network layers
-   **Layer-by-layer Analysis**: Visualize activations, weights, and transformations
-   **Handwritten Digit Recognition**: MNIST-trained model for digit classification
-   **Modern UI**: Clean, responsive React interface
-   **RESTful API**: Flask backend with Redis caching

## Tech Stack

-   **Frontend**: React, Canvas API, Chart.js for visualizations
-   **Backend**: Flask, TensorFlow/Keras, NumPy
-   **Database**: Redis for caching
-   **Containerization**: Docker & Docker Compose
-   **ML Model**: Convolutional Neural Network trained on MNIST

## Getting Started

### Prerequisites

-   Docker and Docker Compose installed
-   Git

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd neural_networks_project
```

2. Build and run with Docker Compose:

```bash
docker-compose up --build
```

3. Access the application:

-   Frontend: http://localhost:3000
-   Backend API: http://localhost:5000

## How It Works

1. **Draw**: Use the canvas to draw a digit (0-9)
2. **Process**: The image is sent to the neural network
3. **Visualize**: See how each layer processes the input
4. **Predict**: Get the final classification result

## API Endpoints

-   `POST /api/predict` - Predict drawn digit
-   `POST /api/visualize` - Get layer-by-layer network visualization
-   `GET /api/model-info` - Get model architecture information

## Project Structure

```
neural_networks_project/
├── backend/
│   ├── app.py
│   ├── model/
│   ├── utils/
│   └── Dockerfile
├── frontend/
│   ├── src/
│   ├── public/
│   └── Dockerfile
└── docker-compose.yml
```
