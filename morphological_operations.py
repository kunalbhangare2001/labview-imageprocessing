import cv2
import numpy as np
import os

def morphological_operations(image_path, output_dir):
    image = cv2.imread(image_path, 0)  # Load as grayscale
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    kernel = np.ones((5, 5), np.uint8)
    
    # Dilation
    dilated = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "dilated.png"), dilated)
    
    # Erosion
    eroded = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "eroded.png"), eroded)
    
    # Opening
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, "opening.png"), opening)
    
    # Closing
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, "closing.png"), closing)
    
    return f"{output_dir}"
