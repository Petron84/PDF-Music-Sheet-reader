import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import os
from PIL import Image

class ActionModelDataset(Dataset):
    def __init__(self, txt_file='actionlog.txt', img_dir='media\\actionmodeldataset', transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with 'filename,label'.
            img_dir (string): Directory where all .png files are stored.
            transform (callable, optional): PyTorch transforms to apply.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        # Parse the text file
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma: e.g., "image1.png,0"
                file_name, label = line.split(',')
                self.samples.append((file_name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # OpenCV reads as Grayscale (H x W)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        # If transform is provided, apply it
        # ToTensor() will scale 0-255 to 0.0-1.0 automatically
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



def train_action_model():
    # Define the pipeline: Numpy -> PIL -> Grayscale Tensor
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    action_dataset = ActionModelDataset(
        txt_file='actionlog.txt', 
        img_dir='media\\actionmodeldataset',
        transform=data_transforms
    )

    # Create the Dataloader
    train_loader = DataLoader(
        dataset=action_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 0 for easier debugging on Windows
    )
    
    print(f"Dataset initialized with {len(action_dataset)} samples.")

    # --- Verification Block ---
    try:
        # Pull the first batch
        images, labels = next(iter(train_loader))
        
        print("-" * 30)
        print("Success! Data loaded successfully.")
        print(f"Batch Image Shape: {images.shape}") # Expected: [32, 1, H, W]
        print(f"Batch Label Shape: {labels.shape}") # Expected: [32]
        print(f"Labels in this batch: {labels.tolist()}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error during data loading: {e}")
    
    
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
