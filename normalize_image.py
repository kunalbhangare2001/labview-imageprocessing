import cv2
import numpy as np
import os

def normalize_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Min-Max Scaling
    min_max = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(output_dir, "min_max.png"), min_max)
    
    # Z-score Normalization
    z_score = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        channel = image[:, :, i]
        mean, std = np.mean(channel), np.std(channel)
        z_score[:, :, i] = (channel - mean) / (std + 1e-8)
    z_score = cv2.normalize(z_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "z_score.png"), z_score)
    
    # Histogram Equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(output_dir, "hist_eq.png"), hist_eq)
    
    return f"{output_dir}"
