import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
            # print("Processing line contour at index: ", self.contours[line[1]])
            x,y,w,h = cv.boundingRect(self.contours[line[1]])
            lineImage = self.image[y:y+h, x:x+w]
            cv.imwrite(f'media\\lines\\{count}_{self.imagepath}.png', lineImage)
            LineProcessor(f'media\\lines\\{count}_{self.imagepath}.png')
            count+=1
            # cv.imshow('Line Image', lineImage)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
    
    
    
    def drawFirstLayerContours(self):
        print("The index count is ", len(self.lineContourIndeces))
        for line in self.lineContourIndeces:
            cv.drawContours(self.image, self.contours, line[1], (0,0,255),2)
        # cv.imshow('Contours', self.image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    
    
    
    
class LineProcessor():
    def __init__(self, imagepath = 'media\\lines\\treble_twinkle star_5.png'):
        self.lineImage = cv.imread(imagepath, cv.IMREAD_GRAYSCALE)
        self.colorImage = cv.imread(imagepath) # this is the color version of the line image, we will use it to draw on it and visualize our results.
        self.imagepath = imagepath[12:-4] # removing the .png part of
        #self.clef = self._identifyClef(self.lineImage)
        #self.imagepath = self.clef + "_" + self.imagepath
        self.height, self.width = self.lineImage.shape
        print('LineProcessor: ',self.imagepath)
        #self.clefsize, self.clefsignatures = self._identifyClefSignature(self.lineImage)
        self.individualLines = []
        self._splitLines()
    
    def _identifyClef(self, lineImage):
        # Implementation for identifying clef
        # Hardcoding trebble celf for now, will implement the actual logic later.
        return "treble"
    
    def _identifyClefSignature(self, lineImage):
        # Implementation for getting the size of the clef signature and the other markings near the clef
        # Hardcoding these, we will dynamically find them later.
        return 33, 0 # clefsize, clefsignatures
    
    def _splitLines(self):
        """This method will analyze our image, assess if it needs to be split up and then adds it to self.indivisualLines for processing
        """
        ret, thresh_img = cv.threshold(self.lineImage, 220, 255, cv.THRESH_BINARY)
        #startingPoint = self.clefsize + self.clefsignatures 
        #Because of when this method is called, we have to begin at the very beginning
        startingPoint = 0
  
        # The y buckets are the pixel count from the shape function
        y_buckets = [0] * self.height
        
        # Now we will go vertical line by vertical line and look for not 0 pixels
        for y in range(self.height):
            for x in range(startingPoint, self.width):
                if thresh_img[y][x] != 255:
                    y_buckets[y] += 1        
        
        cv.imshow(self.imagepath, thresh_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # self.visulatize(y_buckets)
        
        # Let's find the max of the y_buckets
        max_y = max(y_buckets) -10

        # I'm going to say things can be 10 pixels off so subtracting 10 from the max.
        # Now we will find the index of the maxes.
        max_indices = []
        for i in range(len(y_buckets)):
            if y_buckets[i] >= max_y:
                max_indices.append(i)
                
        # Now we're trying to find the start and end of each white space
        white_spaces = []
        for i in range(len(max_indices)-1):
            if max_indices[i+1] - max_indices[i] > 1: # if the difference is more than 1, then there is a white space between them.
                white_spaces.append((max_indices[i], max_indices[i+1]))
        # print("White spaces are: ", white_spaces)
        
        #now we will find the minimum size of the white spaces this will be width of our box
        min_white_space = self.height
        for space in white_spaces:
            space_size = space[1] - space[0]
            if space_size < min_white_space:
                min_white_space = space_size
        # print("Minimum white space is: ", min_white_space)
        
        #Let me draw it so we see something.
        # We will store top_left_corner and botton_right_corner as a list of tuples.
        corners = []
        
        # This draws into the white spaces. In Green
        for space in white_spaces:
            top_left_corner = (startingPoint, space[0]+1)
            bottom_right_corner = (startingPoint+min_white_space, space[1]-1)
            corners.append((top_left_corner, bottom_right_corner))
            #cv.rectangle(self.colorImage, top_left_corner, bottom_right_corner, (0,255,0), 1)
            
        # self.visulatize(y_buckets)
            
        # # This draws on the lines in Blue
        # for line in range(len(max_indices)):
        #     if line==0:
        #         if max_indices[line]-max_indices[line-1]==1:
        #             continue
        #     top_left_corner = (startingPoint, max_indices[line]-int(min_white_space/2))
        #     bottom_right_corner = (startingPoint+min_white_space, max_indices[line]+int(min_white_space/2))
        #     corners.append((top_left_corner, bottom_right_corner))
        #     cv.rectangle(self.colorImage, top_left_corner, bottom_right_corner, (255,0,0), 1)
        
        #print(f"The decision for len(corners)/4 is: {len(corners)/4}")        
        if len(corners)/4 > 1: # there are 4 boxes if there is a single line. If there are more than that, then they must be split
            self._breakintoSingleLines(max_indices, thresh_img)
        else:
            self.individualLines.append(thresh_img) # if there is only one line, then we will just add the whole line image to our individual lines collection.
            
        # print ("Number of individual lines is: ", len(self.individualLines))
        for line in self.individualLines:
            cv.imshow('Line Image', line)
            cv.waitKey(0)
            cv.destroyAllWindows()
            self._lineSeparatorConvolution(line)
            
            # self._horizontalAnalysis(line)
            # self._lookfornotes(corners, max_indices, line , 10)
            
    
    def _lineSeparatorConvolution(self, line):
        """This method will perform a line separation convolution on the line image."""
        clefofline = self._identifyClef(line)
        clefsize, clefsignatures = self._identifyClefSignature(line)
        
        # This is the kernel for the vertical character
        verKernel = np.array([ # It's called Sobel Kernel
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]], dtype=np.float32)
        
        beforeimg = line.astype(np.float32) # Convolution function requires float32 input
        vertimg = cv.filter2D(beforeimg, -1, verKernel) # Performing the convolution. The -1 means the output image will have the same depth as the input image.
        vertimg = np.abs(vertimg).astype(np.uint8) # Taking the absolute value and converting back to uint8 for visualization
        cv.imshow('After Vertical', vertimg)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        opimage = 255 - vertimg # this is the image after inverting the colors. The idea is that the lines will be detected by the kernels and will be white in vertimg, so when we invert it, they will be black and the notes will be white.
        x_ranges = self._getRanges(opimage, clefsize+clefsignatures) # this will get the x ranges of the lines. We will use these to split the line into individual lines.
        notecount = 0
        for (start, end) in x_ranges:
            lineImage = line[:, start:end]
            notecount +=1
            cv.imwrite(f'media\\linenotes\\{clefofline}_{notecount}_{self.imagepath}.png', lineImage)
            print(f'media\\linenotes\\{clefofline}_{notecount}_{self.imagepath}.png')
        # # I tried other kernels and I found that they don't really work. So we will work with the veritcal filter only.
        # posKernel = np.array([ 
        #     [ 0,  1, 2],
        #     [-1,  0, 1],
        #     [-2, -1, 0]], dtype=np.float32)
        # posimg = cv.filter2D(beforeimg, -1, posKernel) # Performing the convolution. The -1 means the output image will have the same depth as the input image.
        # posimg = np.abs(posimg).astype(np.uint8) # Taking the absolute value and converting back to uint8 for visualization
        # cv.imshow('After Positive', posimg) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        # negKernel = np.array([ 
        #     [ 2,  1, 0],
        #     [ 1,  0, -1],
        #     [ 0, -1, -2]], dtype=np.float32)
        # negimg = cv.filter2D(beforeimg, -1, negKernel) # Performing the convolution. The -1 means the output image will have the same depth as the input image.
        # negimg = np.abs(negimg).astype(np.uint8) # Taking the absolute value and converting back to uint8 for visualization
        # cv.imshow('After Negative', negimg) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        # horKernel = np.array([ 
        #     [ -1, -2, -1],
        #     [  0,  0,  0],
        #     [  1,  2,  1]], dtype=np.float32)
        # horimg = cv.filter2D(beforeimg, -1, horKernel) # Performing the convolution. The -1 means the output image will have the same depth as the input image.
        # horimg = np.abs(horimg).astype(np.uint8)
        # cv.imshow('After Horizontal', horimg) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        # afterimage = beforeimg + posimg + negimg + horimg # this is the image after subtracting the lines. The idea is that the lines will be detected by the kernels and subtracted from the original image, leaving only the notes and other markings.
        # afterimage = np.clip(afterimage, 0, 255).astype(np.uint8) # Clipping the values to be between 0 and 255 and converting back to uint8 for visualization
        # cv.imshow('Subtraction', afterimage) 
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
    def _getRanges(self, opimage, startingPoint):
        # Implementation for getting x ranges
        h, w = opimage.shape
        x_buckets = [0] * (w - startingPoint)
        #print("len of x_buckets: ", len(x_buckets))
        for x in range(startingPoint, w):
            #print("height: ", h, "width: ", w)
            for y in range(h):
                #print("y: ", y, "x: ", x, )
                if opimage[y][x] != 255:
                    x_buckets[x-startingPoint] += 1 
                    
        # lets draw them
        white_image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        for x in range(len(x_buckets)):
            for y in range(x_buckets[x]):
                white_image[y][x+startingPoint] = (0,0,0)
        
        cv.imshow('Squished Image', white_image)
        cv.waitKey(0)   
        cv.destroyAllWindows()
        
        ranges = []
        x =0
        while x<len(x_buckets):
            if x_buckets[x]>0: # So this means there was a black pixel in the column
                print("here")
                start = x
                while start<len(x_buckets) and x_buckets[start]>0:
                    start+=1
                end = start +2
                ranges.append((x+startingPoint-2, end+startingPoint))
                print(f"Found a note from {x+startingPoint-2} to {end+startingPoint}")
                x = end
            else:
                x+=1
        return ranges

    def _horizontalAnalysis(self, line):
        """We will look at x frequency and then look at what the image is"""        
        
        (max_y, max_x) = line.shape
        print("Horizontal analysis, max_y: ", max_y, "max_x: ", max_x)
        x_buckets = [0] * max_x
        
        # Now we will go horizontal line by horizontal line and look for not 0 pixels
        for x in range(max_x):
            for y in range(max_y):
                # print(f"line[{y}][{x}] = {line[y][x]}")
                if line[y][x] != 255:
                    x_buckets[x] += 1        
        
        # Let's find the max of the x_buckets
        # I'm going to say things can be 10 pixels off so subtracting 10 from the max.
        max_x = max(x_buckets) -10
        
        # Now we will find the index of the maxes.
        max_indices = []
        for i in range(len(x_buckets)):
            if x_buckets[i] >= max_x:
                max_indices.append(i)
        
        print("Max indeces are: ", max_indices)
        white_image = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        
        for x in range(len(x_buckets)):
            for y in range(x_buckets[x]):
                white_image[y][x] = (0,0,0) 
        
        cv.imshow('Visualization', white_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return max_indices
        
    def _breakintoSingleLines(self, max_indices, thresh_img):
        """"
        This method is designed to break the line into single lines, including the possibility of more than two lines,
        ASSUMING that splitting halfway between lines is a good strategy.
        """
        averageDistance = 0
        
        for i in range(len(max_indices)-1):
            distance = max_indices[i+1]-max_indices[i]
            averageDistance+=distance
            
        averageDistance = averageDistance/(len(max_indices)-1)
        
        #will save these in a list of tuples, (distance, (first_y, second_y))
        distances = []
        for i in range(len(max_indices)-1):
            distance = max_indices[i+1]-max_indices[i]
            if distance>averageDistance:
                distances.append(int((max_indices[i] + max_indices[i+1]) / 2))
        
        lastDistance = 0

        for distance in distances:
            lineImage = thresh_img[lastDistance:distance, :]
            self.individualLines.append(lineImage)
            lastDistance = distance
            # cv.imshow('Line Image', lineImage)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        lineImage = thresh_img[lastDistance:self.height,:]
        self.individualLines.append(lineImage)
       
            
            
            
    def _lookfornotes_original(self, corners, blacklines, thresh_img, sliver_width):
        """This mehtod is now deprecated
        
        This method will look for notes and save an image of the sliver in a predefined way. 
        The parameter blacklines is the list of y values of the black lines and are the black values we will ignore.
        We will be using numpy methods."""
        
        notecount = 0
        startingX = self.clefsize + self.clefsignatures 
        sliver = startingX # this is the x offset that slides down the line.
        while sliver+sliver_width<= self.width:  
            presence = []
            # First we will look above the first black line which are less than the first value of blackline
            # roi = thresh_img[y1:y2, x1:x2]
            roi = thresh_img[0:blacklines[0], sliver:sliver+sliver_width]
            if 0 in roi: # if there is a non white pixel, then there is a note in that sliver
                presence.append(True)
               
            # This will traverse our boxes and look for notes in them.   
            for topleft, bottomright in corners:
                # topleft is  (33, 18) bottomright is  (40, 24)
                roi = thresh_img[topleft[1]:bottomright[1], sliver:sliver+sliver_width]
                if 0 in roi:
                    presence.append(True)# if the sliver is within the box of the white space or the line
            
            # This will check after the last blackline
            roi = thresh_img[blacklines[-1]+1:self.height, sliver:sliver+sliver_width]
            if 0 in roi:
                presence.append(True) # if there is a non white pixel, then there is a note in that sliver
            
            print("Presence is: ", presence)
            if len(presence)>0:
                notecount +=1
                sliverimage = thresh_img[:, sliver:sliver+sliver_width]
                cv.imwrite(f'media\\linenotes\\{notecount}_{self.imagepath}.png', sliverimage)
                print(f'media\\linenotes\\{notecount}_{self.imagepath}.png')
            
            sliver += sliver_width # move the sliver to the right by the width of the white space.
            
        
        
    def _lookfornotes(self, corners, blacklines, thresh_img, sliver_width):
        """"
        This method will look for notes and save an image of the sliver in a predefined way. 
        The parameter blacklines is the list of y values of the black lines and are the black values we will ignore.
        We will be using numpy methods."""
        
        notecount = 0
        startingX = self.clefsize + self.clefsignatures 
        sliver = startingX # this is the x offset that slides down the line.
        while sliver+sliver_width<= self.width:  
            presence = []
            # First we will look above the first black line which are less than the first value of blackline
            # roi = thresh_img[y1:y2, x1:x2]
            roi = thresh_img[0:blacklines[0], sliver:sliver+sliver_width]
            if 0 in roi: # if there is a non white pixel, then there is a note in that sliver
                presence.append(True)
               
            # This will traverse our boxes and look for notes in them.   
            for topleft, bottomright in corners:
                # topleft is  (33, 18) bottomright is  (40, 24)
                roi = thresh_img[topleft[1]:bottomright[1], sliver:sliver+sliver_width]
                if 0 in roi:
                    presence.append(True)# if the sliver is within the box of the white space or the line
            
            # This will check after the last blackline
            roi = thresh_img[blacklines[-1]+1:self.height, sliver:sliver+sliver_width]
            if 0 in roi:
                presence.append(True) # if there is a non white pixel, then there is a note in that sliver
            
            print("Presence is: ", presence)
            if len(presence)>0:
                notecount +=1
                sliverimage = thresh_img[:, sliver:sliver+sliver_width]
                cv.imwrite(f'media\\linenotes\\{notecount}_{self.imagepath}.png', sliverimage)
                print(f'media\\linenotes\\{notecount}_{self.imagepath}.png')
            
            sliver += sliver_width # move the sliver to the right by the width of the white space.
        
        
        
        
        
                
                
                
                
                
                
    def visulatize(self,y_buckets:list[int]):
        """Designed to visualize the y_buckets in graphic format"""
        if(len(y_buckets)==0):
            print("No data to visualize")
        elif(len(y_buckets)!=self.height):
            print("Data length does not match image height. There is a problem with the data.")
        
        white_image = np.full((self.height, self.width-self.clefsize-self.clefsignatures, 3), 255, dtype=np.uint8)
        
        for y in range(len(y_buckets)):
            for x in range(y_buckets[y]):
                white_image[y][x] = (0,0,0) 
        
        cv.imshow('Visualization', white_image)
        cv.waitKey(0)
        cv.destroyAllWindows()