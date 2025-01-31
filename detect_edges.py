import cv2
import numpy as np
import os

def detect_edges(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(sobel / sobel.max() * 255)
    cv2.imwrite(os.path.join(output_dir, "sobel.png"), sobel)
    
    # Canny
    canny = cv2.Canny(gray, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "canny.png"), canny)
    
    # Laplacian of Gaussian
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    log = np.uint8(np.absolute(log))
    cv2.imwrite(os.path.join(output_dir, "log.png"), log)
    
    return f"{output_dir}"
