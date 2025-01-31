import cv2
import os

def denoise_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "gaussian.png"), gaussian)
    
    # Median Blur
    median = cv2.medianBlur(image, 5)
    cv2.imwrite(os.path.join(output_dir, "median.png"), median)
    
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.imwrite(os.path.join(output_dir, "bilateral.png"), bilateral)
    
    return f"{output_dir}"
