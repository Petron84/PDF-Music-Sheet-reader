import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
#import sys
#sys.path.append("/main-0203")
#from geminicode import geminihelper

def readMyPic():
    img = cv.imread(".venv\\main-0203\\media\\blanktreble.png")
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
    
 
 
def findFirstBlack(image,segment):
     print("Hello World")
        
    
    
if __name__ == "__main__":
    readMyPic()
    #geminihelper()