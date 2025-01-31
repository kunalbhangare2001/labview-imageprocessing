import cv2
import numpy as np
import os

def segment_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding for segmentation
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "binary_segmentation.png"), binary)
    
    # Watershed Segmentation
    ret, markers = cv2.connectedComponents(binary)
    markers = markers + 1
    markers[binary == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    cv2.imwrite(os.path.join(output_dir, "watershed_segmentation.png"), image)
    
    return f"{output_dir}"
