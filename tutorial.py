import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from geminicode import geminihelper

def readMyPic():
    img = cv.imread('media\\blanktreble.png')
    
    # img.shape returns a tuple: (Height, Width, Channels)
    height, width, channels = img.shape
    #cv.imshow("Displayed Image", img)
    # Wait for a key press before closing the window
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    segmentsOfFive = width/5
    drawRedLines(imgRGB,segmentsOfFive,height,width)
    plt.figure()
    plt.title("My Image with Redlines as Boxes")
    plt.imshow(imgRGB)    
    plt.show()
    
    drawRedLinesAgain(imgRGB,segmentsOfFive,height,width)
    plt.figure()
    plt.title("My Image with Redlines filled by pixel")
    plt.imshow(imgRGB)    
    plt.show()
    
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    makeFirstBlackLineRed(imgRGB, height,width)
    plt.figure()
    plt.title("My Image Red First Black Line")
    plt.imshow(imgRGB)    
    plt.show()
    
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    findTrebleAfterFirstLine(imgRGB, height,width)
    plt.figure()
    plt.title("My Image Red First Black Line")
    plt.imshow(imgRGB)    
    plt.show()
    
def drawRedLines(imgRGB,segmentsOfFive,height,width):
    """
    Draws three vertical lines across the image.
    """
    ones = np.ones((height,1))
    zeros = np.zeros((height,1))
    redpart = cv.merge((200*ones,zeros,zeros))
    for i in range(int(segmentsOfFive),width,int(segmentsOfFive)):
        try:
            imgRGB[0:height,i:i+1] =redpart  
        except:
            pass
    
def drawRedLinesAgain(imgRGB,segmentsOfFive,height,width):
    """
    Draws three vertical lines across the image.
    """
    ones = np.ones((height,5))
    zeros = np.zeros((height,5))
    redpart = cv.merge((200*ones,zeros,zeros))
    for i in range(int(segmentsOfFive),width,int(segmentsOfFive)):
        for j in range(height):
            try:
                imgRGB[j,i] =(200,0,0)
            except:
                pass
    
def makeFirstBlackLineRed(imgRGB, height,width):
    """
        In the image, we will scan down the middle of the music score, find the first black line and put a redlines through it.
    """
    middle = int(width/2)
    ones = np.ones((1,width))
    zeros = np.zeros((1,width))
    redpart = cv.merge((200*ones,zeros,zeros))
    
    foundHeight = 0
    
    while foundHeight<height:
        red, green, blue = imgRGB[middle,foundHeight]
        if red<225 and green<225 and blue<225:
            break
        foundHeight = foundHeight+1
    
    imgRGB[foundHeight-1:foundHeight,0:width] =redpart

    
def findTrebleAfterFirstLine(imgRGB, height,width):
    """
        This will scan the center of the score and then go to the left looking for the treble clef
    """
    middle = int(width/2)
    
    foundHeight = 0
    
    while foundHeight<height:
        red, green, blue = imgRGB[middle,foundHeight]
        if red<225 and green<225 and blue<225:
            break
        foundHeight = foundHeight+1
    print(foundHeight)
    
    #foundHeight = foundHeight-3
    
    foundWidth = middle
    
    while foundWidth>0:
        red, green, blue = imgRGB[foundWidth,foundHeight]
        print(foundWidth,red,green,blue)
        if red<50 and green<50 and blue<50:
            break
        foundWidth = foundWidth -1 
        
    ones = np.ones((10,10))
    zeros = np.zeros((10,10))
    redpart = cv.merge((200*ones,zeros,zeros))
    
    imgRGB[foundHeight-5:foundHeight+5,foundWidth-5:foundWidth+5] = redpart

def firstContours(imagepath:str)->None:
    img = cv.imread(imagepath)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #cv.drawContours(img, contours, -1, (0, 255, 0), 3)
    # hierarchy[0] contains the actual data
    inner_only = np.zeros_like(img)
    
    # print ("hierarchy", hierarchy)
    #print ("contour", contours)
    current = hierarchy[0][0]
    currentindex = 0
    # first find an outermost contour, need to check index3 of current
    while current[3]!=-1:
        currentindex = current[3]
        current = hierarchy[0][currentindex]
    
    # the outermost contour ends up being the rim of the image so we will needto go to its first child
    currentindex = current[2]
    current = hierarchy[0][currentindex]
     
    maxperimetercontour = contours[currentindex]  
    maxperimeterindex = currentindex
  
    while currentindex!=-1:
        if cv.arcLength(contours[currentindex], True) > cv.arcLength(maxperimetercontour, True):
            maxperimetercontour = contours[currentindex]
            maxperimeterindex = currentindex

        currentindex = hierarchy[0][currentindex][0]
        current = hierarchy[0][currentindex]
        
    cv.drawContours(inner_only, contours, maxperimeterindex, (255,255,255),2) 

    img_h , img_w, _ = inner_only.shape
    resized_im = cv.resize(inner_only, (int(img_w/2), int(img_h/2)), interpolation=cv.INTER_LINEAR)
    cv.imshow('Contours', resized_im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def firstContoursOnOriginal(imagepath: str)->None:
    img = cv.imread(imagepath)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    current = hierarchy[0][0]
    currentindex = 0
    # first find an outermost contour, need to check index3 of current
    while current[3]!=-1:
        currentindex = current[3]
        current = hierarchy[0][currentindex]
    
    # the outermost contour ends up being the rim of the image so we will needto go to its first child
    currentindex = current[2]
    current = hierarchy[0][currentindex]
     
    maxperimetercontour = contours[currentindex]  
    maxperimeterindex = currentindex

    while currentindex!=-1:
        if cv.arcLength(contours[currentindex], True) > cv.arcLength(maxperimetercontour, True):
            maxperimetercontour = contours[currentindex]
            maxperimeterindex = currentindex
        currentindex = hierarchy[0][currentindex][0]
        current = hierarchy[0][currentindex]
        
    cv.drawContours(img, contours, maxperimeterindex, (0,0,255),2) 
    img_h , img_w, _ = img.shape
    resized_im = cv.resize(img, (int(img_w/2), int(img_h/2)), interpolation=cv.INTER_LINEAR)
    cv.imshow('Contours', resized_im)
    cv.waitKey(0)
    cv.destroyAllWindows()