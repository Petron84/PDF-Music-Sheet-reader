import cv2 as cv
import os
from pathlib import Path

def setupgame():
    '''A simple game where the player looks at images of notes and designates them as "garbage" or "not garbage".'''
    subfolder = 'media\\garbagefilterdataset'
    # Create a path object for the subfolder relative to your current location
    subfolder_path = Path(".") / subfolder
    # Check if it exists to avoid errors
    if subfolder_path.exists() and subfolder_path.is_dir():
        # List all files (excluding sub-directories)
        files = [f.name for f in subfolder_path.iterdir() if f.is_file()]
        startgame(files)
    else:
        print("Folder not found!")
        files = os.listdir('media\\garbagefilterdataset')
        print(len(files))
    
def startgame(files):
    lastone = open('media\\groundtruth\\lastfile.txt', 'r').read().strip()
    fileindex = 0
    
    if len(lastone) != 0:
        while files[fileindex] != lastone:
            fileindex += 1
    
    for i in range(fileindex, len(files)):
        file = files[i]
        img = cv.imread('media\\garbagefilterdataset\\' + file)
        cv.imshow("Displayed Image", img)
        key = cv.waitKey(0)  # Wait for a key press
        while not (key == ord('0') or key == ord('1')):
            print("Invalid key pressed. Please press '0' for garbage or '1' for not garbage.")
            key = cv.waitKey(0)  # Wait for a key press again
        key = int(chr(key))
        print(key)
        logintotext(file, key)
        cv.destroyAllWindows()
        
def logintotext(file, key):
    with open('garbagelog.txt', 'a') as f:
        f.write(f"{file}, {key}\n")
    
    with open('media\\groundtruth\\lastfile.txt', 'w') as f:
        f.write(file)
    