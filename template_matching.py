import cv2
import os

def match_template(image_path, template_path, output_dir):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)
    if image is None or template is None:
        return "Error: Failed to load image or template"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Draw rectangle around matched region
    h, w = template.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "template_matched.png"), image)
    
    return f"{output_dir}"
