import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np


class LineContour():
    def __init__(self, imagepath = 'media\\twinkle star.png'):
        self.image =cv.imread(imagepath)
        self.imagepath = imagepath[6:-4] # removing the .png part of the path
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        self.contours, self.hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.lineContourIndeces = self._setLineContourIndices()
        self._breakLines()
    def _setLineContourIndices(self):
        """This method uses the hierarchy and the contours attributes of the LineContour object to find the layer under the 
        outermost contour. Then it will collect all the contours of that layer into a list and return it. The hypothesis is, 
        will be the lines of the music sheet."""
        
        # Initialize the lineContourIndeces collection.
        lineContourIndeces = []
        
        # set current conditions before we navigate it to find the outermost contour.
        # Recall that contours are a collection of point. 
        # Recall that in hierarchy there are 4 values [Next, Previous, First_Child, Parent]
        current = self.hierarchy[0][0] #Technically, the hierarchy is [[Next, Previous, First_Child, Parent],[Next, Previous, First_Child, Parent],[Next, Previous, First_Child, Parent],[Next, Previous, First_Child, Parent]].
        currentindex = 0

        # first find an outermost contour, need to check index3 of current
        # The meaning is, if current's 3rd index is -1, then it has no parent and is the outermost contour.
        while current[3]!=-1:
            currentindex = current[3] # moving the two pointers so that they point to the parent of the current one
            current = self.hierarchy[0][currentindex]
        
        currentindex = current[2] # This is first child. At this point current is pointing at the outermost contour. 
        current = self.hierarchy[0][currentindex] # After this all the pointers are shifted to the first child of the outermost contour.
        
        # let's move the pointers until we find a contour that does not have a "previous sibling"
        while current[1]!=-1: # when this index =-1, contour is the first contour of that layer
            currentindex = current[1] # moving the two pointers so that they point to the previous sibling of the current one
            current = self.hierarchy[0][currentindex]

        # chosenIndex, chosenContour = currentindex, self.contours[currentindex] # initializing with dummy values
        # when it exits that loop, I'm at the first contour of that layer.
        # now I will traverse them one by one, appending the indeces to my collection until I find 
        # a contour that does not have a "next sibling"
        while current[0]!=-1: # when this index =-1, contour is the last contour of that layer
            # chose 10000 as the threshold of contour area             
            if cv.contourArea(self.contours[currentindex])>=10000.0:
                lineContourIndeces.append((current,currentindex)) # adds them to my collection.
                # chosenIndex, chosenContour = currentindex, self.contours[currentindex]
            currentindex = current[0] # moving the two pointers so that they point to the next sibling of the current one
            current = self.hierarchy[0][currentindex]
        
        #when it exits that loops, I will have visited all contours in that layer and collected them.
        return lineContourIndeces


    def _breakLines(self):
        """This method will extract the lines from the image."""
        count = 1
        for line in self.lineContourIndeces:
            print("Processing line contour at index: ", self.contours[line[1]])
            x,y,w,h = cv.boundingRect(self.contours[line[1]])
            lineImage = self.image[y:y+h, x:x+w]
            cv.imwrite(f'media\\lines\\trebble_{self.imagepath}_{count}.png', lineImage)
            count+=1
            # cv.imshow('Line Image', lineImage)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
    
    
    
    def drawFirstLayerContours(self):
        print("The index count is ", len(self.lineContourIndeces))
        for line in self.lineContourIndeces:
            cv.drawContours(self.image, self.contours, line[1], (0,0,255),2)
        cv.imshow('Contours', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
def firstContours(imagepath:str)->None:
    

    
    # the outermost contour ends up being the rim of the image so we will need to go to its first child
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
    
    
    
    # I'm going to draw one more contour
    print("next is: ", hierarchy[0][maxperimeterindex][0])




    img_h , img_w, _ = img.shape
    resized_im = cv.resize(img, (int(img_w/2), int(img_h/2)), interpolation=cv.INTER_LINEAR)
    cv.imshow('Contours', resized_im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    