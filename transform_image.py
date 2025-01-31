import cv2
import numpy as np
import os

def transform_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    rows, cols = image.shape[:2]
    
    # Rotation
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    cv2.imwrite(os.path.join(output_dir, "rotated.png"), rotated)
    
    # Translation
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 50]])
    translated = cv2.warpAffine(image, translation_matrix, (cols, rows))
    cv2.imwrite(os.path.join(output_dir, "translated.png"), translated)
    
    # Scaling
    scaled = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, "scaled.png"), scaled)
    
    return f"{output_dir}"
