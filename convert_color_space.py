import cv2
import os

def convert_color_space(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, "hsv.png"), hsv)
    
    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(output_dir, "lab.png"), lab)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "grayscale.png"), gray)
    
    return f"{output_dir}"
