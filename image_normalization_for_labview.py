import cv2
import numpy as np
import os

def normalize_image(image, output_dir):
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

def augment_image(image, output_dir):
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    cv2.imwrite(os.path.join(output_dir, "bright.png"), bright)
    
    # Add noise
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    cv2.imwrite(os.path.join(output_dir, "noisy.png"), noisy)

def denoise_image(image, output_dir):
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, "gaussian.png"), gaussian)
    
    # Median Blur
    median = cv2.medianBlur(image, 5)
    cv2.imwrite(os.path.join(output_dir, "median.png"), median)
    
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.imwrite(os.path.join(output_dir, "bilateral.png"), bilateral)

def detect_edges(image, output_dir):
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

def binarize_image(image, output_dir):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Global thresholding (Otsu's method)
    _, global_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, "global_thresh.png"), global_thresh)
    
    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_dir, "adaptive_thresh.png"), adaptive_thresh)

# Main function for LabVIEW Python Node
def process_images_labview(image_path, output_dir):
    """
    Main function to be called from LabVIEW Python Node.
    
    Parameters:
    image_path (str): Path to the input image
    output_dir (str): Path to the output directory
    
    Returns:
    str: Status message indicating success or failure
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return f"Error: Failed to load image from {image_path}"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process the image
        normalize_image(image, output_dir)
        augment_image(image, output_dir)
        denoise_image(image, output_dir)
        detect_edges(image, output_dir)
        binarize_image(image, output_dir)
        
        return f"{output_dir}"
    
    except Exception as e:
        return f"Error: {str(e)}"

# For testing outside LabVIEW
if __name__ == "__main__":
    test_image_path = r"path_to_your_test_image.jpg"
    test_output_dir = r"path_to_your_output_directory"
    result = process_images_labview(test_image_path, test_output_dir)
    print(result)