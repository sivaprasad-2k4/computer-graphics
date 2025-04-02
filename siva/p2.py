import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # I. Perform an inverse transformation (negative image)
    inverse_image = cv2.bitwise_not(image)
    
    # II. Enhance the image using contrast stretching
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(grayscale)
    contrast_stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # III. Generate a histogram-equalized image to improve contrast
    hist_eq_image = cv2.equalizeHist(grayscale)
    
    # IV. Detect edges using Canny Edge Detection
    edges = cv2.Canny(grayscale, 100, 200)
    
    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Inverse Image", inverse_image)
    cv2.imshow("Contrast Stretched Image", contrast_stretched)
    cv2.imshow("Histogram Equalized Image", hist_eq_image)
    cv2.imshow("Edge Detection", edges)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "img1.jpeg"  # Replace with the actual image path
process_image(image_path)