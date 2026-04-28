
import os
import cv2 as cv


def move_all_pictures(picture_dir, target_dir):
    """We will iterate through the file names in actionlog.txt,
    we will comma separate each line to title and label,
    when the label is 3 we will open that picture in picture_dir using cv
    then we will save that same picture into target_dir, keeping the same name."""
    with open('actionlog.txt', 'r') as f:
        for line in f:
            title, label = line.strip().split(',')
            print(label)
            if int(label) == 3:
                print(title, label)
                img_path = os.path.join(picture_dir, title)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
                if img is not None:
                    target_path = os.path.join(target_dir, title)
                    cv.imwrite(target_path, img)
                    with open('asidelog.txt', 'a') as log_file:
                        log_file.write(f"{title},{label}\n")
                else:
                    print(f"Warning: Could not read image {img_path}")
                    
def double_check_pictures(target_dir):
    """We will iterate through the file names in asidelog.txt,
    we will comma separate each line to title and label,
    when the label is 3 we will open that picture in target_dir using cv.
    If it does not ope, we will print the title and label to the console."""
    with open('asidelog.txt', 'r') as f:
        for line in f:
            title, label = line.strip().split(',')
            if int(label) == 3:
                img_path = os.path.join(target_dir, title)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
                if img is None:
                    print(f"Warning: Could not read image {img_path}")

def check_Otherway(target_dir):
    """First, we will open asidelog.txt and store the titles in a set. 
    Then we will go through evevery file in the target_dir and check that
    its title is in the set. If not, we will print the title to the console."""
    with open('asidelog.txt', 'r') as f:
        asidelog_titles = {line.strip().split(',')[0] for line in f}

    for filename in os.listdir(target_dir):
        if filename not in asidelog_titles:
            print(f"Warning: File {filename} is in target_dir but not in asidelog.txt")

def main():
    picture_dir = 'media\\actionmodeldataset_v2\\'
    target_dir = 'media\\aside\\'
    # double_check_pictures(target_dir)
    # check_Otherway(target_dir)
    move_all_garbage(picture_dir, target_dir)

def move_all_garbage(picture_dir, target_dir):
    """We will iterate through the file names in actionlog.txt,
    we will comma separate each line to title and label,
    when the label is 0 we will open that picture in picture_dir using cv
    then we will save that same picture into target_dir, keeping the same name."""
    with open('actionlog.txt', 'r') as f:
        for line in f:
            title, label = line.strip().split(',')
            print(label)
            if int(label) == 0:
                print(title, label)
                img_path = os.path.join(picture_dir, title)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
                if img is not None:
                    target_path = os.path.join(target_dir, title)
                    cv.imwrite(target_path, img)
                    with open('garbagelog.txt', 'a') as log_file:
                        log_file.write(f"{title},{label}\n")
                else:
                    print(f"Warning: Could not read image {img_path}")


main()