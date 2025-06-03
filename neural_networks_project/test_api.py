#!/usr/bin/env python3
"""
Simple test script to verify the Neural Network API functionality
"""

import requests
import base64
import numpy as np
from PIL import Image, ImageDraw
import io
import json

def create_test_digit(digit=7, size=(280, 280)):
    """Create a simple test image of a digit"""
    # Create white canvas
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple digit 7
    if digit == 7:
        # Draw lines to make a "7"
        draw.line([(50, 50), (230, 50)], fill='black', width=20)  # Top horizontal line
        draw.line([(230, 50), (130, 230)], fill='black', width=20)  # Diagonal line
    elif digit == 1:
        # Draw a simple "1"
        draw.line([(140, 50), (140, 230)], fill='black', width=20)  # Vertical line
    
    return img

def image_to_base64(img):
    """Convert PIL image to base64 data URL"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def test_health():
    """Test the health endpoint"""
    print("Testing /api/health endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/health')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\nTesting /api/model-info endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/model-info')
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Total parameters: {data.get('total_params')}")
        print(f"Number of layers: {len(data.get('layers', []))}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\nTesting /api/predict endpoint...")
    try:
        # Create test image
        test_img = create_test_digit(7)
        img_data = image_to_base64(test_img)
        
        # Make prediction request
        payload = {"image": img_data}
        response = requests.post('http://localhost:5000/api/predict', 
                               json=payload,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Predicted digit: {data.get('prediction')}")
            print(f"Confidence: {data.get('confidence', 0):.4f}")
            print(f"Top 3 probabilities:")
            probs = data.get('probabilities', [])
            for i, prob in enumerate(probs):
                if prob > 0.01:  # Only show probabilities > 1%
                    print(f"  Digit {i}: {prob:.4f}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_visualization():
    """Test the visualization endpoint"""
    print("\nTesting /api/visualize endpoint...")
    try:
        # Create test image
        test_img = create_test_digit(1)
        img_data = image_to_base64(test_img)
        
        # Make visualization request
        payload = {"image": img_data}
        response = requests.post('http://localhost:5000/api/visualize', 
                               json=payload,
                               headers={'Content-Type': 'application/json'})
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Visualization data keys: {list(data.keys())}")
            if 'prediction' in data:
                pred = data['prediction']
                print(f"Predicted digit: {pred.get('digit')}")
                print(f"Confidence: {pred.get('confidence', 0):.4f}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_training_info():
    """Test the training info endpoint"""
    print("\nTesting /api/train-info endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/train-info')
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Training epochs: {data.get('epochs')}")
            print(f"Final accuracy: {data.get('final_accuracy', 0):.4f}")
            print(f"Final validation accuracy: {data.get('final_val_accuracy', 0):.4f}")
        else:
            print(f"Error response: {response.text}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Neural Network API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Training Info", test_training_info),
        ("Prediction", test_prediction),
        ("Visualization", test_visualization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:20} : {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall Status: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()
