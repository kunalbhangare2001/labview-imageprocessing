import cv2
import numpy as np
import os

def sharpen_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Define sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    cv2.imwrite(os.path.join(output_dir, "sharpened.png"), sharpened)
    
    return f"{output_dir}"
