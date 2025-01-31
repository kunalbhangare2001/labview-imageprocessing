import cv2
import numpy as np
import os

def calculate_histogram(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Failed to load image from {image_path}"
    
    # Calculate histograms for each channel
    colors = ('b', 'g', 'r')
    histogram_data = {}
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        histogram_data[color] = hist
    
    # Save histogram data to file
    hist_path = os.path.join(output_dir, "histogram_data.txt")
    with open(hist_path, "w") as f:
        for color, hist in histogram_data.items():
            f.write(f"{color} channel:\n{hist.flatten().tolist()}\n\n")
    
    return f"Histogram data saved at {hist_path}"
