import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
#from geminicode import geminihelper

def readPic():
    img = cv.imread("media\\twinkle star.png") 
    if img is None:
        print("Image not found")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)

    height, width = binary.shape
    instaff = False
    
    # Store detected staff boundaries as (top, bottom) tuples
    potential_staves = []
    current_start = 0

    # Search zone for the clef
    start_x = int(width * 0.05)
    end_x = int(width * 0.15)

    for i in range(height):
        black_pixel_count = np.sum(binary[i, start_x:end_x]) / 255

        if not instaff and black_pixel_count > 10:
            instaff = True
            current_start = i 

        elif instaff and black_pixel_count < 2:
            instaff = False
            current_end = i
            # ADDED: Only keep segments that are tall enough (e.g., > 60 pixels)
            if (current_end - current_start) > 30:
                potential_staves.append((current_start, current_end))

    # Now draw the lines only for the valid staves
    for (top, bottom) in potential_staves:
        # Draw blue line at the top
        img[top, :] = [255, 0, 0]
        # Draw green line at the bottom
        img[bottom, :] = [0, 255, 0]
        
        print(f"Detected staff at Y-range: {top} to {bottom} (Height: {bottom-top})")

    cv.imshow("Cleaned Staff Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def dynamic_readPic():
    # this is a dynamic version of readPic, with the value of "tallness" be determined
    # by the height of the first detected staff. This is to accomodate for different sizes of sheet music.
    img = cv.imread("media\\twinkle star.png")
    if img is None:
        print("Image not found")
        return
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
    height, width = binary.shape
    instaff = False

    potential_staves = []
    current_start = 0
    start_x = int(width * 0.05)
    end_x = int(width * 0.15)

    first_staff_height = None
    for i in range(height):
        black_pixel_count = np.sum(binary[i, start_x:end_x]) / 255

        if not instaff and black_pixel_count > 10:
            instaff = True
            current_start = i 

        # allow some noise, if there are too many white pixels, we consider it as the end of the staff
        # this value of 0.75 is just a guess value. aka a ratio of black pixels to total pixels in the search zone.
        # adjustments may be needed based on the quality of the image and the size of the search zone.
        elif instaff and black_pixel_count <= 0.75: 
            instaff = False
            current_end = i
            staff_height = current_end - current_start
            
            if first_staff_height is None:
                first_staff_height = staff_height
            
            # Use the first detected staff height as a threshold
            if staff_height > (first_staff_height * 0.5):  # Allow some variation
                potential_staves.append((current_start, current_end))
    for (top, bottom) in potential_staves:
        img[top, :] = [255, 0, 0]
        img[bottom, :] = [0, 255, 0]
        print(f"Detected staff at Y-range: {top} to {bottom} (Height: {bottom-top})")
    cv.imshow("Dynamic Staff Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
def findFirstBlack(image,segment):
     pass
        
    
    
if __name__ == "__main__":
    dynamic_readPic()

    #geminihelper()
