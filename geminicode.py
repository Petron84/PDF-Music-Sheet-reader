import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def geminihelper():
    # 1. Setup the path
    # Use the current directory where main.py is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    # CHANGE THIS: Put a sample sheet music image in your folder
    filename = 'media\\blanktreble.png' 
    img_path = os.path.join(base_path, filename)

    # 2. Load Image
    img = cv.imread(img_path)

    if img is None:
        print(f"Error: Could not find image at {img_path}")
        print("Make sure to place a 'sheet_music_sample.jpg' in the project folder!")
    else:
        # 3. Pre-processing for Music Reading
        
        # A. Convert to Grayscale
        # We don't need color to identify notes
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # B. Gaussian Blur (Noise Reduction)
        # This slightly smoothes the image to remove small 'speckles' 
        # that aren't notes, making the thresholding cleaner.
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # C. Adaptive Thresholding (Crucial for Documents/Music)
        # This turns the image into pure Black (0) and White (255).
        # It calculates the threshold for a small region around each pixel.
        # 11 is the block size (neighbor area), 2 is a constant subtracted from the mean.
        binary = cv.adaptiveThreshold(blurred, 255, 
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv.THRESH_BINARY_INV, 11, 2)
        # Note: I used THRESH_BINARY_INV so notes/lines become White (255) 
        # and background becomes Black (0). This is often easier for detection algorithms.

        # 4. Display Results using Matplotlib
        plt.figure(figsize=(12, 6))

        # Original (Grayscale)
        plt.subplot(1, 2, 1)
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Input")
        plt.axis('off')

        # Binary (Thresholded)
        plt.subplot(1, 2, 2)
        plt.imshow(binary, cmap='gray')
        plt.title("Adaptive Threshold (Binary)\n(Ready for detection)")
        plt.axis('off')

        plt.show()