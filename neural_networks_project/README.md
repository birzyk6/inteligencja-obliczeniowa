# 🧠 Neural Network Visualization App

<div align="center">

![Neural Network](https://img.shields.io/badge/Neural%20Network-CNN-blue?style=for-the-badge&logo=tensorflow)
![React](https://img.shields.io/badge/React-18.0-61DAFB?style=for-the-badge&logo=react)
![Flask](https://img.shields.io/badge/Flask-Python-green?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker)

**AI-Powered Digit Recognition with Real-Time Neural Network Visualization**

_An interactive full-stack application that demonstrates how Convolutional Neural Networks process handwritten digits with live visualization of each layer's activations_

</div>

---

## 🎯 Demo & Screenshots

### 🎬 Live Application Demo

<div align="center">

**Watch the Neural Network in Action**

<<<<<<< HEAD
<video controls>
  <source src="https://www.youtube.com/watch?v=ucLcYAWXpMU" type="video/mp4">
</video>
=======
[https://www.youtube.com/watch?v=ucLcYAWXpMU&ab_channel=BartoszIrzyk](https://github.com/user-attachments/assets/29ab24e1-caec-4767-8c62-52e19633a9b4)
>>>>>>> 3cb5632107d358fbc4bc1a3dc333abf01e515a1f

_Complete application walkthrough showing real-time digit recognition, neural network visualization, and interactive features_

</div>

### 🖥️ Main Application Interface

<div align="center">

![Application Interface](plots/9_app_screen.png)

_Interactive web interface featuring dark theme design with real-time digit recognition and neural network visualization_

</div>

### 🔍 Neural Network Layer Visualizations

<table>
<tr>
<td width="50%">

**Convolutional Layers**
![Conv Layer Visualization](plots/9_conv1_feature_maps.png)
_Feature maps showing edge detection and pattern recognition_

</td>
<td width="50%">

**Dense Layer Activations**
![Dense Layer Visualization](plots/9_dense1.png)
_Abstract feature representations in fully connected layers_

</td>
</tr>
</table>

### 📊 Training Performance Analysis

<table>
<tr>
<td width="33%">

![Training Loss](plots/training_plots/training_loss.png)
_Loss convergence over training epochs_

</td>
<td width="33%">

![Training Accuracy](plots/training_plots/training_accuracy.png)
_Accuracy improvement during training_

</td>
<td width="33%">

![Learning Rate](plots/training_plots/learning_rate.png)
_Adaptive learning rate scheduling_

</td>
</tr>
</table>

---

## ✨ Key Features

<table>
<tr>
<td width="25%" align="center">

### 🎨 Interactive Drawing

Draw digits (0-9) on HTML5 canvas with real-time prediction

</td>
<td width="25%" align="center">

### 🧠 Neural Network Visualization

Live visualization of activations through all network layers

</td>
<td width="25%" align="center">

### 📈 Performance Analytics

Comprehensive training analysis with Polish documentation

</td>
<td width="25%" align="center">

### 🌙 Modern UI

Dark theme interface with gradient effects and responsive design

</td>
</tr>
</table>

---

## 🚀 Tech Stack

<div align="center">

|                                       **Frontend**                                        |                                         **Backend**                                          |                                                **ML/AI**                                                 |                                              **DevOps**                                               |
| :---------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
| ![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=white) |  ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)   | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) |     ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)      |
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |        ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)         | ![Compose](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat&logo=docker&logoColor=white) |
|  ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)   |  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)   |       ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)       |          ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)          |

</div>

---

## 🏗️ Architecture

### 🧠 Neural Network Architecture

```mermaid
graph TD
    A[Input: 28x28 Grayscale Image] --> B[Conv2D: 32 filters, 3x3]
    B --> C[BatchNormalization]
    C --> D[Conv2D: 32 filters, 3x3]
    D --> E[MaxPooling2D: 2x2]
    E --> F[Dropout: 25%]
    F --> G[Conv2D: 64 filters, 3x3]
    G --> H[BatchNormalization]
    H --> I[Conv2D: 64 filters, 3x3]
    I --> J[MaxPooling2D: 2x2]
    J --> K[Dropout: 25%]
    K --> L[Conv2D: 128 filters, 3x3]
    L --> M[BatchNormalization]
    M --> N[Conv2D: 128 filters, 3x3]
    N --> O[Dropout: 25%]
    O --> P[Flatten]
    P --> Q[Dense: 512 neurons]
    Q --> R[BatchNormalization]
    R --> S[Dropout: 50%]
    S --> T[Dense: 256 neurons]
    T --> U[Dropout: 50%]
    U --> V[Dense: 10 classes - Softmax]
```

### 📊 Model Performance

|       **Metric**        | **Value**  |
| :---------------------: | :--------: |
|  **Training Accuracy**  |   99.39%   |
| **Validation Accuracy** | **99.61%** |
|  **Total Parameters**   |   ~2.3M    |
|   **Training Epochs**   |     18     |
| **Final Learning Rate** |  0.000125  |

---

## 🚀 Quick Start

### 📋 Prerequisites

-   [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)
-   [Git](https://git-scm.com/downloads)

### ⚡ Installation

1. **Clone the repository**

```bash
git clone git@github.com:birzyk6/inteligencja-obliczeniowa.git
cd neural_networks_project
```

2. **Launch with Docker Compose**

```bash
docker-compose up --build
```

3. **Access the application**
    - 🌐 **Frontend**: [http://localhost:3000](http://localhost:3000)
    - 🔗 **Backend API**: [http://localhost:5000](http://localhost:5000)

---

## 🎮 How to Use

<div align="center">

### 1️⃣ Draw → 2️⃣ Process → 3️⃣ Visualize → 4️⃣ Analyze

</div>

1. **🎨 Draw a digit** on the interactive canvas
2. **⚡ Real-time processing** by the neural network
3. **🔍 Visualize activations** through each layer
4. **📊 Analyze predictions** with confidence scores

---

## 🔌 API Endpoints

| **Method** |   **Endpoint**    | **Description**                               |
| :--------: | :---------------: | :-------------------------------------------- |
|   `POST`   |  `/api/predict`   | 🎯 Predict drawn digit with confidence scores |
|   `POST`   | `/api/visualize`  | 🔍 Get layer-by-layer network visualization   |
|   `GET`    | `/api/model-info` | ℹ️ Retrieve model architecture information    |

---

## 📁 Project Structure

```
neural_networks_project/
├── 📱 frontend/                 # React application
│   ├── src/
│   │   ├── App.js              # Main application component
│   │   ├── components/         # React components
│   │   │   ├── DrawingCanvas.js
│   │   │   ├── PredictionDisplay.js
│   │   │   ├── NetworkVisualization.js
│   │   │   └── ModelInfo.js
│   │   └── index.js
│   ├── public/
│   └── Dockerfile
├── 🔧 backend/                  # Flask API server
│   ├── app.py                  # Main Flask application
│   ├── model/                  # Neural network implementation
│   │   ├── neural_network.py   # CNN model definition
│   │   ├── trained_model.h5    # Pre-trained weights
│   │   └── training_history.json
│   ├── utils/                  # Utility functions
│   │   ├── image_processing.py
│   │   └── visualization.py
│   └── Dockerfile
├── 📊 plots/                    # Generated visualizations
│   ├── training_plots/         # Training analysis charts
│   └── *_feature_maps.png      # Layer activation visualizations
├── 📈 training_analysis.ipynb   # Jupyter notebook with analysis
├── 📋 WNIOSKI.md               # Polish conclusions document
├── 🐳 docker-compose.yml       # Container orchestration
└── 📖 README.md                # This file
```

---

## 🏆 Key Achievements

-   **🎯 99.61% Validation Accuracy** - Superior performance on MNIST dataset
-   **🚫 Zero Overfitting** - Validation accuracy exceeds training accuracy
-   **⚡ Real-time Inference** - Instant digit recognition and visualization
-   **🎨 Modern UI/UX** - Professional dark theme with gradient effects
-   **📊 Comprehensive Analysis** - Complete training metrics and visualizations
-   **🐳 Containerized Deployment** - Easy setup with Docker Compose

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
