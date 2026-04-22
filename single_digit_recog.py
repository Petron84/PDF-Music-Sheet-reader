import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("PyTorch version:", torch.__version__)

transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

############################################################################
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

images, labels = next(iter(train_loader))
print(images.shape) # (64, 1, 28, 28) -> batch_size, channels, height, width
print(labels.shape) # (64,) -> batch_size
#############################################################################
import torch.nn as nn
##############################################################################
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#############################################################################
import torch.nn.functional as F
#############################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # architectire: input → (conv1→relu→pool) → (conv2→relu→pool) → flatten → (fc1→relu) → fc2

        #conv1 takes in 1 channel (grayscale) and outputs 16 channels, with a 3x3 kernel and padding of 1
        #to maintain the spatial dimensions
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        #conv2 takes in 16 channels and outputs 32 channels, with a 3x3 kernel and padding of 1 
        #to maintain the spatial dimensions
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        # after two rounds of conv+pool, the spatial dimensions are reduced 
        # from 28x28 to 7x7, and we have 32 channels
        self.fc1 = nn.Linear(32 * 7 * 7, 128)   
        self.fc2 = nn.Linear(128, 10)   #10 output classes for digits 0-9

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))   # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 → 7x7

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
#############################################################################
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
################################################################################
for epoch in range(5):
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)   # transfer stuff to gpu
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
################################################################################

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)   # transfer stuff to gpu

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total)
print("model used:", model)