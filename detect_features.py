import cv2
import os

def detect_features(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # ORB Feature Detector
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray, None)
    orb_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite(os.path.join(output_dir, "orb_features.png"), orb_image)
    
    # Harris Corner Detector
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris = cv2.dilate(harris, None)
    image[harris > 0.01 * harris.max()] = [0, 0, 255]
    cv2.imwrite(os.path.join(output_dir, "harris_corners.png"), image)
    
    return f"{output_dir}"
