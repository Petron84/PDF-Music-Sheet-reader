import torch
import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from geminicode import geminihelper
from statdicer import LineContour
import garbagemodeltruthbuilder
import actionmodelbuilder

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
    # imagepath ='media\\blanktreble.png'
    # imagepath ='media\\silentnight.png'
    # imagepath ='media\\twinkle star.png'
    # imagepath ="media\\basetemplate.png"
    # firstContours(imagepath)
    # imagepath = "media\\pdf2png(1).png"
    # imagepath = "media\\pdf2png(2).png"
    # imagepath = "media\\pdf2png(2).png"
    # imagepath = "media\\pdf2png(2).png"
    # imagepath = "media\\Enfantilligis_2.png"
    # imagepath = "media\\pdf2png(3).png"
    # lineContour = LineContour(imagepath)
    # lineContour.drawFirstLayerContours()
    # 
    # actionmodelbuilder.deleteTooSmall()
    # imagepath = "media\\linenotes\\1_random1_treble_pdf2png(1)_v3_24.png"
    # image = cv.imread(imagepath)
    # w, h, c = image.shape
    # print(f"Width: {w}, Height: {h}, Channels: {c}")
    # cv.imshow("Displayed Image", image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # garbagemodeltruthbuilder.setupgame()
    # garbagemodeltruthbuilder.getStats(os.listdir('media\\actionmodeldataset_v2'))
    # garbagemodeltruthbuilder.getMaxShape(os.listdir('media\\actionmodeldataset_v2'))

    #actionmodelbuilder.train_action_model()
    #lineContour = LineContour(imgp)

    dirlist = os.listdir('media\\ds')
    #LineContour('media\\ds\\lg-2267728-aug-beethoven--page-2.png')
    cv.imshow("Test", cv.imread('media\\ds\\lg-2267728-aug-beethoven--page-2.png'))
    print(cv.waitKey(0))
    
    """
    for fileindex in range(len(dirlist)):
        if dirlist[fileindex].endswith('.png'):
            path = 'media\\ds\\' + dirlist[fileindex]
            linec = LineContour(path)
            print(f"Processed {fileindex} out of {len(dirlist)} files.")
    """

    #for file in dirlist:
    #    if file.endswith('.png'):
    #        lineContour = LineContour(f'media\\ds\\{file}')
"""
    for fileindex in range(len(dirlist)):
        dirlistlen = len(dirlist)
        if dirlist[fileindex].endswith('.png'):
            tempimg = cv.imread(f'media\\lines\\{dirlist[fileindex]}', cv.IMREAD_GRAYSCALE)
            cv.imshow(f'sorting {fileindex}/{dirlistlen}', cv.resize(tempimg, (0,0), fx=0.5, fy=0.5))

            match cv.waitKey(0):
                case 48: # 0 key
                    print("skip")

                case 49: # 1 key
                    imggo = 'media\\linesproper\\' + dirlist[fileindex][:-4] + '.png'
                    cv.imwrite(imggo, tempimg)
                    print(os.path.exists(imggo))
                
                case 50: # 2 key
                    splitimg_h = int(tempimg.shape[0] / 2)
                    topimg = tempimg[:splitimg_h, :]
                    bottomimg = tempimg[splitimg_h:, :]
                    topimggo = 'media\\linesproper\\' + dirlist[fileindex][:-4] + '_top.png'
                    bottomimggo = 'media\\linesproper\\' + dirlist[fileindex][:-4] + '_bottom.png'
                    cv.imwrite(topimggo, topimg)
                    cv.imwrite(bottomimggo, bottomimg)

                case 51: # 3 key
                     print("skip")
                
                case _:
                    print("Invalid key, skipping")
            cv.destroyAllWindows()
"""