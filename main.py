import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from tutorial import firstContours, firstContoursOnOriginal
from pdf2image import convert_from_path

def openTwinkleAsPDF():
    pages = convert_from_path('media\\twinkle-twinkle-little-star-piano-solo.pdf', dpi=300)
    print(len(pages),'size of pages')
    
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
    
        
    
    
if __name__ == "__main__":
    imagepath ='media\\blanktreble.png'
    #imagepath ='media\\silentnight.png'
    #imagepath ='media\\twinkle star.png'
    #imagepath ="media\\basetemplate.png"

    #firstContours(imagepath)
    firstContoursOnOriginal(imagepath)
