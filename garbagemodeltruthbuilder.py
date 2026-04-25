import cv2 as cv
import os
from pathlib import Path

def setupgame():
    '''A simple game where the player looks at images of notes and designates them as "garbage" or "not garbage".'''
    subfolder = 'media\\actionmodeldataset'
    # Create a path object for the subfolder relative to your current location
    subfolder_path = Path(".") / subfolder
    # Check if it exists to avoid errors
    if subfolder_path.exists() and subfolder_path.is_dir():
        # List all files (excluding sub-directories)
        files = [f.name for f in subfolder_path.iterdir() if f.is_file()]
        startgame(files)
    else:
        print("Folder not found!")
        files = os.listdir('media\\actionmodeldataset')
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
        img = cv.imread('media\\actionmodeldataset\\' + file)
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
        img = cv.imread('media\\actionmodeldataset\\' + im)
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
        
        
    