import cv2
import os

def binarize_image(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Global thresholding (Otsu's method)
    _, global_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "global_thresh.png"), global_thresh)
    
    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_dir, "adaptive_thresh.png"), adaptive_thresh)
    
    return f"{output_dir}"
