import cv2
import numpy as np
import os

def augment_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    cv2.imwrite(os.path.join(output_dir, "bright.png"), bright)
    
    # Add noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    cv2.imwrite(os.path.join(output_dir, "noisy.png"), noisy)
    
    return f"{output_dir}"
