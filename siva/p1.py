import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    # I. Print size and shape of the image
    print(f"Image Dimensions: {image.shape}")
    print(f"Image Size: {image.size} bytes")
    
    # II. Convert to grayscale and binary
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY)
    
    # III. Scale the image (reduce size by half)
    scaled = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    
    # IV. Remove noise using Gaussian Blur
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", grayscale)
    cv2.imshow("Binary Image", binary)
    cv2.imshow("Scaled Image", scaled)
    cv2.imshow("Denoised Image", denoised)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "img1.jpeg"  # Replace with the actual image path
process_image(image_path)