import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from geminicode import geminihelper

def readMyPic():
    img = cv.imread('media\\blanktreble.png')
    
    # img.shape returns a tuple: (Height, Width, Channels)
    height, width, channels = img.shape
    cv.imshow("Displayed Image", img)
    segmentsOfFive = width/5

    listDepths = []
    for i in range(0,width,int(segmentsOfFive)):
        listDepths.append(findFirstBlack(img, i))
    
    
 
    # Wait for a key press before closing the window
    cv.waitKey(0)
    cv.destroyAllWindows()
    
 
 
def findFirstBlack(image,segment):
     print("Hello World")
        
    
    
if __name__ == "__main__":
    readMyPic()
    #geminihelper()