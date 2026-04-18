import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import os

class MusicNoteDataset(Dataset):
    def __init__(self, log_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = [] # index 0 will be the image name, index 1 will be the label (0 or 1)
        
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    # Your log format uses ', ' as a separator
                    parts = line.strip().split(', ')
                    if len(parts) == 2:
                        self.samples.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (OpenCV reads BGR, PIL reads RGB - usually better for Torchvision)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        # Target: 101 height, 51 width
        # Calculate total padding needed
        total_pad_h = 101 - h
        total_pad_w = 51 - w
        
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