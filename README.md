# 🧠 Inteligencja Obliczeniowa - Computational Intelligence Projects

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![NumPy](https://img.shields.io/badge/NumPy-Data-013243?style=for-the-badge&logo=numpy)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-FF6F00?style=for-the-badge&logo=tensorflow)

**A comprehensive collection of computational intelligence projects demonstrating various AI and optimization techniques**

</div>

---

## 📋 Table of Contents

-   [🧠 Inteligencja Obliczeniowa - Computational Intelligence Projects](#-inteligencja-obliczeniowa---computational-intelligence-projects)
    -   [📋 Table of Contents](#-table-of-contents)
    -   [🎯 Project Overview](#-project-overview)
    -   [🚀 Projects](#-projects)
        -   [1. 🎲 Monte Carlo π Estimation](#1--monte-carlo-π-estimation)
        -   [2. 🗺️ Traveling Salesman Problem (Komiwojażer)](#2-️-traveling-salesman-problem-komiwojażer)
        -   [3. 📊 Fuzzy Logic Investment System](#3--fuzzy-logic-investment-system)
        -   [4. 🤖 Neural Network MNIST Recognition](#4--neural-network-mnist-recognition)
    -   [📋 Prerequisites](#-prerequisites)
    -   [🔧 Installation](#-installation)
        -   [Quick Setup (All Projects)](#quick-setup-all-projects)
        -   [Project-Specific Setup](#project-specific-setup)
            -   [Monte Carlo Project](#monte-carlo-project)
            -   [Traveling Salesman Project](#traveling-salesman-project)
            -   [Fuzzy Logic Project](#fuzzy-logic-project)
            -   [Neural Network Project](#neural-network-project)
    -   [🎮 Usage](#-usage)
        -   [Running Individual Projects](#running-individual-projects)
    -   [📊 Results](#-results)
        -   [Monte Carlo](#monte-carlo)
        -   [Traveling Salesman](#traveling-salesman)
        -   [Fuzzy Logic](#fuzzy-logic)
        -   [Neural Network](#neural-network)
    -   [🤝 Contributing](#-contributing)
    -   [📝 License](#-license)
    -   [📧 Contact](#-contact)

---

## 🎯 Project Overview

This repository contains four distinct computational intelligence projects, each demonstrating different AI techniques and methodologies:

1. **Monte Carlo Methods** - Statistical simulation for π estimation
2. **Genetic Algorithms** - Optimization for the Traveling Salesman Problem
3. **Fuzzy Logic Systems** - Investment risk assessment
4. **Neural Networks** - Deep learning for digit recognition

Each project showcases practical applications of computational intelligence in solving real-world problems.

---

## 🚀 Projects

### 1. 🎲 Monte Carlo π Estimation

**Location:** `lab1 - Monte Carlo/`

A statistical simulation project that estimates the value of π using Monte Carlo methods.

**Key Features:**

-   Random point generation and geometric probability
-   Animated visualization of the estimation process
-   Convergence analysis and accuracy metrics
-   Interactive Jupyter notebook implementation

**Files:**

-   `montecarlo.ipynb` - Main notebook with implementation and visualization
-   `monte_carlo_pi.gif` - Animation of the estimation process

**How it works:**

-   Generates random points in a unit square
-   Counts points falling inside a quarter circle
-   Uses the ratio to estimate π: `π ≈ 4 × (points inside circle / total points)`

---

### 2. 🗺️ Traveling Salesman Problem (Komiwojażer)

**Location:** `komiwoj/`

An optimization project solving the classic Traveling Salesman Problem using genetic algorithms.

**Key Features:**

-   Genetic algorithm implementation with multiple selection/crossover methods
-   Parameter optimization using grid search
-   Route visualization and evolution animation
-   Comprehensive performance analysis

**Main Files:**

-   `src/ga.py` - Genetic algorithm implementation
-   `src/optimizer.py` - Parameter optimization module
-   `src/visualization.py` - Plotting and animation functions
-   `src/main.py` - Main execution script

**Capabilities:**

-   Multiple crossover types (order, PMX, cycle)
-   Various selection methods (tournament, roulette, rank)
-   Adaptive mutation rates
-   Performance comparison with/without optimization

**Sample Results:**

-   Best route visualizations
-   Evolution animations (GIF format)
-   Parameter sensitivity analysis
-   Convergence plots

---

### 3. 📊 Fuzzy Logic Investment System

**Location:** `fuzzy_logic_project/`

A decision support system using fuzzy logic for investment risk assessment.

**Key Features:**

-   Fuzzy membership functions for investment variables
-   Rule-based inference system
-   Multi-dimensional risk visualization
-   Comprehensive market analysis

**Input Variables:**

-   Market Volatility (low → high)
-   Company Financial Health (poor → excellent)
-   Industry Growth Potential (declining → booming)

**Output:**

-   Investment Risk Assessment (very safe → very risky)

**Visualizations:**

-   Membership function plots
-   3D risk surface plots
-   Contour plots for decision boundaries
-   Example case studies

---

### 4. 🤖 Neural Network MNIST Recognition

**Location:** `neural_networks_project/`

A full-stack web application for handwritten digit recognition with real-time neural network visualization.

**Key Features:**

-   Convolutional Neural Network (CNN) implementation
-   Interactive web interface with drawing canvas
-   Real-time layer activation visualization
-   Docker containerization for easy deployment

**Architecture:**

-   **Frontend:** React.js with interactive drawing canvas
-   **Backend:** Flask API with TensorFlow/Keras
-   **Model:** CNN trained on MNIST dataset
-   **Deployment:** Docker Compose setup

**Unique Features:**

-   Live visualization of each CNN layer's activations
-   Interactive digit drawing and immediate recognition
-   Feature map visualization for convolutional layers
-   Modern dark-theme web interface

**Technical Stack:**

-   TensorFlow/Keras for deep learning
-   React.js for frontend interface
-   Flask for REST API backend
-   Docker for containerization

---

## 📋 Prerequisites

-   Python 3.8 or higher
-   pip package manager
-   Docker (for neural network project)
-   Jupyter Notebook (for Monte Carlo project)

---

## 🔧 Installation

### Quick Setup (All Projects)

```bash
# Clone the repository
git clone <repository-url>
cd inteligencja-obliczeniowa

# Install common dependencies
pip install numpy matplotlib pandas scikit-learn jupyter
```

### Project-Specific Setup

#### Monte Carlo Project

```bash
cd "lab1 - Monte Carlo"
jupyter notebook montecarlo.ipynb
```

#### Traveling Salesman Project

```bash
cd komiwoj
pip install -r requirements.txt
python src/main.py
```

#### Fuzzy Logic Project

```bash
cd fuzzy_logic_project
pip install -r requirements.txt
python src/fuzzy_investment.py
```

#### Neural Network Project

```bash
cd neural_networks_project

# Using Docker (Recommended)
docker-compose up --build

# Or manual setup
pip install -r backend/requirements.txt
cd frontend && npm install
```

---

## 🎮 Usage

### Running Individual Projects

1. **Monte Carlo π Estimation:**

    ```bash
    cd "lab1 - Monte Carlo"
    jupyter notebook montecarlo.ipynb
    ```

2. **Traveling Salesman:**

    ```bash
    cd komiwoj
    python src/main.py
    ```

3. **Fuzzy Logic Investment:**

    ```bash
    cd fuzzy_logic_project
    python src/fuzzy_investment.py
    ```

4. **Neural Network App:**
    ```bash
    cd neural_networks_project
    docker-compose up
    # Then visit http://localhost:3000
    ```

---

## 📊 Results

Each project generates comprehensive results and visualizations:

### Monte Carlo

-   π estimation convergence plots
-   Animation of point distribution
-   Accuracy analysis over iterations

### Traveling Salesman

-   Optimized route visualizations
-   Evolution animations showing improvement over generations
-   Parameter sensitivity analysis
-   Performance comparisons

### Fuzzy Logic

-   Investment risk assessment plots
-   3D surface visualizations
-   Membership function graphs
-   Decision boundary analysis

### Neural Network

-   Real-time digit recognition
-   Layer activation visualizations
-   Feature map displays
-   Interactive web interface

---

## 🤝 Contributing

This repository is part of a computational intelligence course. Each project demonstrates different aspects of AI and optimization techniques suitable for educational purposes.

---

## 📝 License

This project is created for educational purposes as part of computational intelligence coursework.

---

## 📧 Contact

For questions about any of the projects, please refer to the individual README files in each project directory for more detailed information.

---

<div align="center">

**🧠 Exploring the Frontiers of Computational Intelligence 🧠**

</div>
