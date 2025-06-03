import numpy as np
import cv2
from PIL import Image


def preprocess_image(pil_image):
    """
    Preprocess the drawn image for neural network input
    Convert to grayscale, resize to 28x28, and normalize
    """
    # Convert PIL image to numpy array
    img_array = np.array(pil_image)

    # If image has alpha channel, remove it
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Invert colors (white background to black, black drawing to white)
    img_array = 255 - img_array

    # Resize to 28x28
    img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize to 0-1 range
    img_array = img_array.astype("float32") / 255.0

    # Add batch dimension
    return np.expand_dims(img_array, axis=0)


def center_image(image):
    """Center the drawn digit in the image using center of mass"""
    # Find center of mass
    moments = cv2.moments(image)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # Calculate shift needed to center
        h, w = image.shape
        shift_x = w // 2 - cx
        shift_y = h // 2 - cy

        # Create transformation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Apply translation
        centered = cv2.warpAffine(image, M, (w, h))
        return centered

    return image


def enhance_image(image):
    """Apply enhancement techniques to improve recognition"""
    # Apply Gaussian blur to smooth edges
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply threshold to create binary image
    _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    return thresholded
