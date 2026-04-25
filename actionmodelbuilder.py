from sympy import Max
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from PIL import Image, ImageOps
import os


class MusicNoteDataset(Dataset):
    def __init__(self, log_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [] # index 0 will be the image name, index 1 will be the label (0,1,2 or 3)
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    # Your log format uses ', ' as a separator
                    parts = line.strip().split(', ')
                    if len(parts) == 2:
                        self.samples.append((parts[0], int(parts[1])))
                    else:
                        print(f"Skipping malformed line: {line.strip()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (OpenCV reads BGR, PIL reads RGB - usually better for Torchvision)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Target: 
        # Max height: 395
        # Max width: 169
        # Calculate total padding needed
        total_pad_h = 395 - h
        total_pad_w = 169 - w
        
        # Split the padding for centering
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
        
        # PIL Padding: (left, top, right, bottom)
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        
        # Pad with white pixels (255, 255, 255)
        image = ImageOps.expand(image, padding, fill=(255, 255, 255)) 

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
    
    
    
def deleteTooSmall():
    """We will go through the files in linenotes and then delete the ones that are more narrow than 15 pixels because they are not useful for training the model
    """
    deleted_files = 0
    folder_path = 'media\\linenotes'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Image.open() is "lazy"; it only reads the header
            with Image.open(file_path) as img:
                width, height = img.size
                
                if width < 20:
                    print(f"Deleting {filename}: Width ({width}px) is too narrow.")
                    # We must close the image or exit the 'with' block 
                    # before deleting to release the file lock.
                    need_to_delete = True
                else:
                    need_to_delete = False
                    
        except (IOError, SyntaxError):
            # This handles non-image files or corrupted images
            print(f"Skipping {filename}: Not a valid image.")
            need_to_delete = False

        if need_to_delete:
            os.remove(file_path)
            deleted_files += 1
                
    print(f"Deleted {deleted_files} files that were too narrow.")        
