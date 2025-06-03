@echo off
REM Neural Network Visualizer Setup Script for Windows

echo 🧠 Neural Network Visualizer Setup
echo ==================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are installed

REM Build and start the application
echo 🚀 Building and starting the application...
docker-compose up --build -d

echo.
echo 🎉 Application is starting up!
echo.
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:5000
echo 🗄️ Redis: localhost:6379
echo.
echo ⏱️  Please wait a moment for the services to fully start...
echo The neural network model will be trained on first startup (may take a few minutes).
echo.
echo To stop the application: docker-compose down
echo To view logs: docker-compose logs -f
echo.
pause
