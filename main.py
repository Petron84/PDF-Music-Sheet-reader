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
    
    #resize image to a fixed dimension
    img = cv.resize(img, (800, 1000))
    cv.imshow("Dynamic Staff Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_clef(img_path, template_path):
    # this function just tests out if the program can detect the clef, similar to detect_multi_scale
    img = cv.imread(img_path)
    template = cv.imread(template_path, 0) # the big clef crop
    if img is None or template is None:
        print("Missing files!")
        return
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    result = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
    (startX, startY) = maxLoc
    (tH, tW) = template.shape[:2]
    (endX, endY) = (startX + tW, startY + tH)
    cv.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv.imshow("Clef Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def detect_multi_scale():
    img = cv.imread("media\\twinkle star.png")
    template = cv.imread("template\\treble_clef.png", 0) # Your small clef crop
    
    if img is None or template is None:
        print("Missing files!")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tH, tW = template.shape[:2]
    found = None

    # Loop over the scales of the image
    # We test 30 different sizes from 100% down to 20%
    for scale in np.linspace(0.1, 2, 100)[::-1]:
        # Resize the image according to the scale
        resized = cv.resize(gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
        ratio = gray.shape[1] / float(resized.shape[1])

        # If the resized image is smaller than the template, break the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # Apply template matching
        result = cv.matchTemplate(resized, template, cv.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)

        # If we found a new maximum correlation value, update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, ratio)

    # Unpack the "best" result and calculate coordinates
    (maxVal, maxLoc, ratio) = found
    (startX, startY) = (int(maxLoc[0] * ratio), int(maxLoc[1] * ratio))
    (endX, endY) = (int((maxLoc[0] + tW) * ratio), int((maxLoc[1] + tH) * ratio))

    # Draw the boundary on the original image
    cv.rectangle(img, (0, startY), (img.shape[1], endY), (255, 0, 0), 2)
    
    cv.imshow("Scale-Invariant Result", img)
    cv.waitKey(0)
        
    
    
if __name__ == "__main__":
    detect_clef("media\\twinkle star.png", "template\\treble_clef.png")

    #geminihelper()
