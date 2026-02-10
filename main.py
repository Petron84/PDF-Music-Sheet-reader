import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
#import sys
#sys.path.append("/main-0203")
#from geminicode import geminihelper

def readPic():
    img = cv.imread("media\\blanktreble.png") # Modified to accommodate the way my files are organized and the virtual environment system
    cv.waitKey(0)
    
    # img.shape returns a tuple: (Height, Width, Channels)
    height, width, channels = img.shape
    print(img.shape)
    #cv.imshow("Displayed Image", img)
    #segmentsOfFive = width/5

    #listDepths = []
    #for i in range(0,width,int(segmentsOfFive)):
        #listDepths.append(findFirstBlack(img, i))
    
    instaff = False
    for i in range(1,height,3):
        curRowSet = [0,0,0]
        for j in range (0,width):
            for z in range(-1,2):
                curRowSet[0] += int(img[i+z,j][0])
                curRowSet[1] += int(img[i+z,j][1])
                curRowSet[2] += int(img[i+z,j][2])
        for rowSetIn in range(0,3):
            curRowSet[rowSetIn] = curRowSet[rowSetIn] / (width*3)
        if (not instaff) and (sum(curRowSet)/len(curRowSet) < 192):
            instaff = True
            for j in range (0,width):
                img[i,j] = [255,0,0]
        # else:
            # print(sum(curRowSet)/len(curRowSet))
        if (instaff) and (sum(curRowSet)/len(curRowSet) > 254):
            instaff = False
            for j in range (0,width):
                img[i-3,j] = [0,255,0]
    cv.imshow("Displayed Image", img)

    # Wait for a key press before closing the window
    cv.waitKey(0)
    cv.destroyAllWindows()


#Phat's code here
def readPic2():
    img = cv.imread("media\\blanktreble.png") # Using your uploaded file name
    if img is None:
        print("Image not found")
        return

    height, width, _ = img.shape
    instaff = False
    
    # We only need to check the left side of the image where the clef is
    # Checking the first 20% of the width is usually enough
    clef_zone = int(width * 0.2)

    for i in range(1, height - 1):
        # Find the darkest pixel in the current row (within the clef zone)
        # 0 is black, 255 is white.
        min_val = np.min(img[i, 0:clef_zone])

        # If we find a dark pixel (< 128) and we aren't "in staff" yet
        if not instaff and min_val < 128:
            instaff = True
            # Draw blue line at the very top of the clef
            img[i, :] = [255, 0, 0] 

        # If the entire row is white (> 250) and we were "in staff"
        elif instaff and np.min(img[i, 0:clef_zone]) > 250:
            instaff = False
            # Draw green line at the very bottom of the clef
            img[i-1, :] = [0, 255, 0]

    cv.imshow("Detected Boundaries", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
 
 
def findFirstBlack(image,segment):
     print("Hello World")
        
    
    
if __name__ == "__main__":
    readPic2()

    #geminihelper()
