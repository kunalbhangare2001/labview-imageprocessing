import cv2
import numpy as np
import os

def perspective_transform(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    rows, cols = image.shape[:2]
    
    # Define source and destination points for perspective transformation
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([[0, 0], [cols-1, 0], [int(0.33*cols), rows-1], [int(0.67*cols), rows-1]])
    
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, matrix, (cols, rows))
    cv2.imwrite(os.path.join(output_dir, "perspective_transformed.png"), transformed)
    
    return f"{output_dir}"
