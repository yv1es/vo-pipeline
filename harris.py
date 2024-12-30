import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_harris_keypoints(
    image: np.ndarray, block_size=2, ksize=3, k=0.04, threshold=0.01
):
    """
    Extract keypoints using the Harris corner detection algorithm.

    Args:
        image (np.ndarray): Input image (grayscale).
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture size of the Sobel derivative.
        k (float): Harris detector free parameter.
        threshold (float): Threshold for detecting corners (relative to max value).

    Returns:
        keypoints (list): List of (x, y) keypoints.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Harris corner detection
    harris_response = cv2.cornerHarris(image, block_size, ksize, k)

    # Normalize and threshold the response
    harris_response = cv2.normalize(
        harris_response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    keypoints = np.argwhere(harris_response > threshold * harris_response.max())

    # Convert keypoints to list of tuples (x, y)
    keypoints = [(int(y), int(x)) for x, y in keypoints]

    return keypoints
