import cv2
import os

def track_objects(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "contours.png"), result_image)
    
    return f"{output_dir}"
