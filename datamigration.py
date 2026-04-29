
import os
import cv2 as cv
import random


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
    target_dir = 'media\\onlycategory3\\'
    # double_check_pictures(target_dir)
    # check_Otherway(target_dir)
    #move_all_garbage(picture_dir, target_dir)
    #add_to_category3(target_dir)
    #getsubsampleintoproject('media\\aside\\', 'media\\onlycategory2\\', 2, 'onlyrightlog.txt')
    # move_all_widenleft(picture_dir, 'media\\aside\\')
    # move_to_category1('media\\onlycategory1\\')
    # move_all_widenright(picture_dir, 'media\\aside\\')
    assemble_balanced_datasate()

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
                    with open('asidelog.txt', 'a') as log_file:
                        log_file.write(f"{title},{label}\n")
                else:
                    print(f"Warning: Could not read image {img_path}")

def add_to_category3(target_dir):
    """First, we will open picturelog.txt and store the titles in a set. 
    Then we will go through every file in the target_dir and check that
    its title is in the set. If not, we will print the title to the console 
    then open picturelog and then append  file name, 3."""
    with open('picturelog.txt', 'r') as f:
        asidelog_titles = {line.strip().split(',')[0] for line in f}

    for filename in os.listdir(target_dir):
        if filename not in asidelog_titles:
            print(f"Warning: File {filename} is in target_dir but not in picturelog.txt")
            with open('picturelog.txt', 'a') as log_file:
                log_file.write(f"{filename},3\n")


def getsubsampleintoproject(source_dir, target_dir, label_to_move, logfile_name):
    """We will go to source_dir and get all file names into a list. 
    Then from that list we will randomly select 1550 file names.
    We will open the file name from the source_dir using opencv,
    then we will save that same picture into target_dir, keeping the same name.
    We will also open logfile_name and write the file name and label to it."""
    all_files = os.listdir(source_dir)
    if(len(all_files) < 1550):
        print(f"Warning: Not enough files in {source_dir} to select 1550. Only {len(all_files)} files available.")
        return
    selected_files = random.sample(all_files, 1550)

    for filename in selected_files:
        img_path = os.path.join(source_dir, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_dir, filename)
            cv.imwrite(target_path, img)
            with open(logfile_name, 'a') as log_file:
                log_file.write(f"{filename},{label_to_move}\n")
        else:
            print(f"Warning: Could not read image {img_path}")


def move_all_widenleft(picture_dir, target_dir):
    """We will iterate through the file names in actionlog.txt,
    we will comma separate each line to title and label,
    when the label is 1 we will open that picture in picture_dir using cv
    then we will save that same picture into target_dir, keeping the same name."""
    with open('actionlog.txt', 'r') as f:
        for line in f:
            title, label = line.strip().split(',')
            if int(label) == 1:
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


def move_to_category1(target_dir):
    """
    We will open asidelog.txt. On each line we will comma separate the title and label.
    If opening the picture with opencv fails, we will continue to the next file.
    If it succeeds, we will save a copy of the picture in target_dir, keeping the same name. 
    We will also log the title and label in onlyleftlog.txt.
    """
    with open('asidelog.txt', 'r') as f:
        asidelog_titles = {line.strip().split(',')[0] for line in f}

    for title in asidelog_titles:
        img_path = os.path.join('media\\aside\\', title)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_dir, title)
            cv.imwrite(target_path, img)
            with open('onlyleftlog.txt', 'a') as log_file:
                log_file.write(f"{title},1\n")
        else:
            print(f"Warning: Could not read image {img_path}")

def move_all_widenright(picture_dir, target_dir):
    """We will iterate through the file names in actionlog.txt,
    we will comma separate each line to title and label,
    when the label is 2 we will open that picture in picture_dir using cv
    then we will save that same picture into target_dir, keeping the same name."""
    with open('actionlog.txt', 'r') as f:
        for line in f:
            title, label = line.strip().split(',')
            if int(label) == 2:
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

def assemble_balanced_datasate():
    all_picture_names = os.listdir('media\\onlycategory3\\')
    all_garbage_names = os.listdir('media\\onlycategory0\\')
    all_left_names = os.listdir('media\\onlycategory1\\')
    all_right_names = os.listdir('media\\onlycategory2\\')
    
    target_folder = 'media\\balanced_dataset\\'
    
    minimum_length = min(len(all_picture_names), len(all_garbage_names), len(all_left_names), len(all_right_names))
    
    for i in range(minimum_length):
        img_path = os.path.join('media\\onlycategory3\\', all_picture_names[i])
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_folder, all_picture_names[i])
            cv.imwrite(target_path, img)
            with open('balancedlog.txt', 'a') as log_file:
                log_file.write(f"{all_picture_names[i]},3\n")
        else:
            print(f"Warning: Could not read image {img_path}")
        
        img_path = os.path.join('media\\onlycategory0\\', all_garbage_names[i])
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_folder, all_garbage_names[i])
            cv.imwrite(target_path, img)
            with open('balancedlog.txt', 'a') as log_file:
                log_file.write(f"{all_garbage_names[i]},0\n")
        else:
            print(f"Warning: Could not read image {img_path}")
        
        img_path = os.path.join('media\\onlycategory1\\', all_left_names[i])
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_folder, all_left_names[i])
            cv.imwrite(target_path, img)
            with open('balancedlog.txt', 'a') as log_file:
                log_file.write(f"{all_left_names[i]},1\n")
        else:
            print(f"Warning: Could not read image {img_path}")
        
        img_path = os.path.join('media\\onlycategory2\\', all_right_names[i])
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
        if img is not None:
            target_path = os.path.join(target_folder, all_right_names[i])
            cv.imwrite(target_path, img)
            with open('balancedlog.txt', 'a') as log_file:
                log_file.write(f"{all_right_names[i]},2\n")
        else:
            print(f"Warning: Could not read image {img_path}")
            
    if len(all_picture_names) > minimum_length:
        for i in range(minimum_length, len(all_picture_names)):
            img_path = os.path.join('media\\onlycategory3\\', all_picture_names[i])
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
            if img is not None:
                target_path = os.path.join(target_folder, all_picture_names[i])
                cv.imwrite(target_path, img)
                with open('balancedlog.txt', 'a') as log_file:
                    log_file.write(f"{all_picture_names[i]},3\n")
            else:
                print(f"Warning: Could not read image {img_path}")
                
    if len(all_garbage_names) > minimum_length:
        for i in range(minimum_length, len(all_garbage_names)):
            img_path = os.path.join('media\\onlycategory0\\', all_garbage_names[i])
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
            if img is not None:
                target_path = os.path.join(target_folder, all_garbage_names[i])
                cv.imwrite(target_path, img)
                with open('balancedlog.txt', 'a') as log_file:
                    log_file.write(f"{all_garbage_names[i]},0\n")
            else:
                print(f"Warning: Could not read image {img_path}")

    if len(all_left_names) > minimum_length:
        for i in range(minimum_length, len(all_left_names)):
            img_path = os.path.join('media\\onlycategory1\\', all_left_names[i])
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
            if img is not None:
                target_path = os.path.join(target_folder, all_left_names[i])
                cv.imwrite(target_path, img)
                with open('balancedlog.txt', 'a') as log_file:
                    log_file.write(f"{all_left_names[i]},1\n")
            else:
                print(f"Warning: Could not read image {img_path}")

    if len(all_right_names) > minimum_length:
        for i in range(minimum_length, len(all_right_names)):
            img_path = os.path.join('media\\onlycategory2\\', all_right_names[i])
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE) 
            if img is not None:
                target_path = os.path.join(target_folder, all_right_names[i])
                cv.imwrite(target_path, img)
                with open('balancedlog.txt', 'a') as log_file:
                    log_file.write(f"{all_right_names[i]},2\n")
            else:
                print(f"Warning: Could not read image {img_path}")





main()