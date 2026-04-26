import torch
from torch.utils.data import Dataset, DataLoader, random_split
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

class ActionModel(torch.nn.Module):
    def __init__(self):
        super(ActionModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1) 
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 98 * 42, 32)  #targetheight=395, targetwidth=169 by half
        self.fc2 = torch.nn.Linear(32, 64) 
        self.fc3 = torch.nn.Linear(64, 128) 
        self.fc4 = torch.nn.Linear(128, 4) 
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)  # Reduce spatial dimensions by half
        x = torch.relu(self.conv2(x))
        x = self.pool(x)  # Reduce spatial dimensions by half
        x = x.view(-1, 32 * 98 * 42)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_action_model():
    # 1. Define Transforms (Crucial: Resize to match your Linear layer math)
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((395, 169)), # Ensuring all images match your CNN input
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    action_dataset = ActionModelDataset(
        txt_file='actionlog.txt', 
        img_dir='media\\actionmodeldataset',
        transform=data_transforms
    )
    
    # 3. Calculate 80-10-10 Split
    total_size = len(action_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # random_split shuffles the indices for you automatically
    train_set, val_set, test_set = random_split(action_dataset, [train_size, val_size, test_size])

    # 4. Create DataLoaders
    train_loader = DataLoader(train_set, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=0)
    val_loader = DataLoader(val_set, 
                            batch_size=32, 
                            shuffle=False, 
                            num_workers=0)
    test_loader = DataLoader(test_set, 
                             batch_size=32, 
                             shuffle=False, 
                             num_workers=0)
    
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
        
    # 5. Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training Loop (Skeleton)
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation Phase
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {100 * val_correct/len(val_set):.2f}%")
    
    # Save the model
    torch.save(model.state_dict(), "models//action_model.pth")
    print("Training Complete. Model saved.")
    
    # --- Test Block ---
    print("Evaluating on Test Set...")
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()

    print(f"Test Acc: {100 * test_correct/len(test_set):.2f}%")


    
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
