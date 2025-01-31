import cv2
import os

def blur_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Average Blurring
    avg_blur = cv2.blur(image, (5, 5))
    cv2.imwrite(os.path.join(output_dir, "average_blur.png"), avg_blur)
    
    # Gaussian Blurring
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "gaussian_blur.png"), gaussian_blur)
    
    # Median Blurring
    median_blur = cv2.medianBlur(image, 5)
    cv2.imwrite(os.path.join(output_dir, "median_blur.png"), median_blur)
    
    return f"{output_dir}"
