#!/bin/bash

# Neural Network Visualizer Setup Script

echo "🧠 Neural Network Visualizer Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Build and start the application
echo "🚀 Building and starting the application..."
docker-compose up --build -d

echo ""
echo "🎉 Application is starting up!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5000"
echo "🗄️ Redis: localhost:6379"
echo ""
echo "⏱️  Please wait a moment for the services to fully start..."
echo "The neural network model will be trained on first startup (may take a few minutes)."
echo ""
echo "To stop the application: docker-compose down"
echo "To view logs: docker-compose logs -f"
