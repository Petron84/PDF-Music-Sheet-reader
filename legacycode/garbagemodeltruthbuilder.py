import cv2 as cv
import os
from pathlib import Path

from sympy import Max

def setupgame():
    '''A simple game where the player looks at images of notes and designates them as "garbage" or "not garbage".'''
    subfolder = 'media\\amds_balanced_v2'
    # Create a path object for the subfolder relative to your current location
    subfolder_path = Path(".") / subfolder
    # Check if it exists to avoid errors
    if subfolder_path.exists() and subfolder_path.is_dir():
        # List all files (excluding sub-directories)
        files = [f.name for f in subfolder_path.iterdir() if f.is_file()]
        startgame(files)
    else:
        print("Folder not found!")
        files = os.listdir('media\\amds_balanced_v2')
        print(len(files))
        
    # getStats(files)
    
def startgame(files):
    lastone = open('media\\groundtruth\\lastfile.txt', 'r').read().strip()
    fileindex = 0
    
    if len(lastone) != 0:
        while files[fileindex] != lastone:
            fileindex += 1
        fileindex += 1
    
    for i in range(fileindex, len(files)):
        if i == len(files)-1:
            continue
        file = files[i]
        img = cv.imread('media\\amds_balanced_v2\\' + file)
        cv.imshow("Displayed Image", img)
        key = cv.waitKey(0)  # Wait for a key press
        while not (key == ord('0') or key == ord('1') or key == ord('2') or key == ord('3')):
            print("Invalid key pressed. Please press '0' for Skip, '1' for widen left, '2' for widen right, '3' for Save.")
            key = cv.waitKey(0)  # Wait for a key press again
        key = int(chr(key))
        print(key)
        logintotext(file, key)
        cv.destroyAllWindows()
        
def logintotext(file, key):
    with open('actionlog.txt', 'a') as f:
        f.write(f"{file}, {key}\n")
    
    with open('media\\groundtruth\\lastfile.txt', 'w') as f:
        f.write(file)
    
    
def getStats(files):

    with open('actionlog.txt', 'r') as f:
        lines = f.readlines() 
    garbage_count = 0
    widen_left_count = 0
    widen_right_count = 0
    picture_count = 0
    
    nongarbage = []
   
    for line in lines:
        if line.strip():  # Check if the line is not empty
            im, label = line.strip().split(', ')
            print(f"Image: {im}, Label: {label}")
            if label == '1':
                widen_left_count += 1
                nongarbage.append(im)
            elif label == '0':
                garbage_count += 1
            elif label == '2':
                widen_right_count += 1
                nongarbage.append(im)
            elif label == '3':
                picture_count += 1
                nongarbage.append(im)
    
    print(f"Nongarbage size is {len(nongarbage)}")
    print(f"Garbage: {garbage_count}")
    print(f"Widen Left: {widen_left_count}")
    print(f"Widen Right: {widen_right_count}")
    print(f"Picture: {picture_count}")
    
    allShapes = []
    for im in nongarbage:
        img = cv.imread('media\\amds_balanced_v2\\' + im)
        allShapes.append(img.shape)
        
    max_height = max(shape[0] for shape in allShapes)
    max_width = max(shape[1] for shape in allShapes)
        
    with open('garbagestats.txt', 'w') as f:
        f.write(f"Garbage: {garbage_count}\n")
        f.write(f"Widen Left: {widen_left_count}\n")
        f.write(f"Widen Right: {widen_right_count}\n")
        f.write(f"Picture: {picture_count}\n")
        f.write(f"Max height of non-garbage images: {max_height}\n")
        f.write(f"Max width of non-garbage images: {max_width}\n")
        
        
def getMaxShape(files):
    allShapes = []
    for im in files:
        img = cv.imread('media\\amds_balanced_v2\\' + im)
        allShapes.append(img.shape)
        
    max_height = max(shape[0] for shape in allShapes)
    max_width = max(shape[1] for shape in allShapes)
    
    print(f"Max height: {max_height}")
    print(f"Max width: {max_width}") 
    
    ave_height = sum(shape[0] for shape in allShapes) / len(allShapes)
    ave_width = sum(shape[1] for shape in allShapes) / len(allShapes)
    
    print(f"Average height: {ave_height}")
    print(f"Average width: {ave_width}")
    
    median_height = sorted(shape[0] for shape in allShapes)[len(allShapes) // 2]
    median_width = sorted(shape[1] for shape in allShapes)[len(allShapes) // 2] 
    
    print(f"Median height: {median_height}")
    print(f"Median width: {median_width}")
    
    # count the number of images with width greater than 76
    count_greater_than_76 = sum(1 for shape in allShapes if shape[1] > 76)
    print(f"Number of images with width greater than 76: {count_greater_than_76}")
    # count the number of images with width less than or equal to 76
    count_less_than_or_equal_to_76 = sum(1 for shape in allShapes if shape[1] <= 76)
    print(f"Number of images with width less than or equal to 76: {count_less_than_or_equal_to_76}")
    
    with open('garbagestats.txt', 'a') as f:
        f.write(f"Max height: {max_height}\n")
        f.write(f"Max width: {max_width}\n")
        f.write(f"Average height: {ave_height}\n")
        f.write(f"Average width: {ave_width}\n")
        f.write(f"Median height: {median_height}\n")
        f.write(f"Median width: {median_width}\n")

def _modifyAllToMakeSize(folderpath='media\\amds_balanced_v2', targetheight=395, targetwidth=169):
    #Max height: 395
    #Max width: 169
    #These were measured previously using getMaxShape()
    with open('actionlog.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip():  # Check if the line is not empty
            im, label = line.strip().split(', ')
            img = cv.imread(folderpath + '\\' + im)
            h, w, c = img.shape
            print(f"Original shape of {im}: {img.shape}")
            # I need to see if the height needs adjusting
            if h < targetheight:
                padding_needed = targetheight - h
                top_padding = padding_needed // 2
                bottom_padding = padding_needed - top_padding
                img = cv.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
            
            #checking if width padding is needed    
            if w < targetwidth:
                #If label is widen left, add padding to the right. If label is widen right, add padding to the left. 
                # If label is picture or garbage, add padding equally to both sides.
                padding_needed = targetwidth - w
                if label == '1':  # widen left
                    img = cv.copyMakeBorder(img, 0, 0, 0, padding_needed, cv.BORDER_CONSTANT, value=[255, 255, 255])
                    #save this to the same file name, overwriting the old one
                    cv.imwrite(folderpath + '\\' + im, img)
                elif label == '2':  # widen right
                    img = cv.copyMakeBorder(img, 0, 0, padding_needed, 0, cv.BORDER_CONSTANT, value=[255, 255, 255])
                    #save this to the same file name, overwriting the old one
                    cv.imwrite(folderpath + '\\' + im, img)
                else:  # picture or garbage
                    left_padding = padding_needed // 2
                    right_padding = padding_needed - left_padding
                    img = cv.copyMakeBorder(img, 0, 0, left_padding, right_padding, cv.BORDER_CONSTANT, value=[255, 255, 255])
                    #save this to the same file name, overwriting the old one
                    cv.imwrite(folderpath + '\\' + im, img)
                    
                    
def getAverageandStandardDev(folderpath='media\\actionmodeldataset'):
    allShapes = []
    for im in os.listdir(folderpath):
        img = cv.imread(folderpath + '\\' + im)
        allShapes.append(img.shape)
        
    heights = [shape[0] for shape in allShapes]
    widths = [shape[1] for shape in allShapes]
    
    average_height = sum(heights) / len(heights)
    average_width = sum(widths) / len(widths)
    
    std_dev_height = (sum((h - average_height) ** 2 for h in heights) / len(heights)) ** 0.5
    std_dev_width = (sum((w - average_width) ** 2 for w in widths) / len(widths)) ** 0.5
    
    print(f"Average height: {average_height}")
    print(f"Average width: {average_width}")
    print(f"Standard deviation of height: {std_dev_height}")
    print(f"Standard deviation of width: {std_dev_width}")