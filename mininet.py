import cv2 as cv
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm


datacsv = pd.read_csv('media\\clefdata\\lines\\cleflabels.csv')
shuffledata = datacsv.sample(frac=1).reset_index(drop=True)
print(shuffledata.shape)
train_data = shuffledata[:3600]
test_data = shuffledata[3600:]

class ClefDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_dir):
        self.data = dataframe
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filenum = row['filenum']
        category = row['category']
        clefname = row['clefname']
        
        img_path = self.img_dir + str(filenum) + '.png' # f'media\\clefdata\\lines\\{clefname}\\{filenum}.png'
        
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, (50, 50), interpolation=cv.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        
        return torch.tensor(image), torch.tensor(category)

train_dataset = ClefDataset(dataframe=train_data
                            , img_dir='media\\clefdata\\lines\\combined\\')
test_dataset = ClefDataset(dataframe=test_data
                            , img_dir='media\\clefdata\\lines\\combined\\')

train_loader = torch.utils.data.DataLoader(train_dataset
                                           , batch_size=32
                                           , shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset
                                           , batch_size=32
                                             , shuffle=False)

images, labels = next(iter(train_loader))
print("Batch of images shape: ", images.shape)
print("Batch of labels shape: ", labels.shape)
print("Labels in this batch:", labels)

class ClefClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01), 
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.extractor(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClefClassifier().to(device)
crit = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in tqdm(train_loader, desc='Training'):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optim.zero_grad()
        predict = model(X_batch)
        
        loss = crit(predict, y_batch)
        epoch_loss += loss.item()

        loss.backward()
        optim.step()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")
    if(len(losses) <= 1):
        continue
    if(losses[-1] > (.99 * losses[-2])):
        print(f"Halting due to low loss decrease at epoch {epoch+1}")
        break

"""

dirlist_t = os.listdir('media\\clefdata\\lines\\treble')
dirlist_b = os.listdir('media\\clefdata\\lines\\bass')
print(dirlist_t[0])

col1_filenum = []
col2_category = []
col3_clefname = []

for fileindex in range(len(dirlist_t)):
    if fileindex >= 2200:
        break
    if dirlist_t[fileindex].endswith('.png'):
        col1_filenum.append(dirlist_t[fileindex][:-4]) # removes the .png part of the filename, leaving just the number.
        col2_category.append(0)
        col3_clefname.append('treble')
for fileindex in range(len(dirlist_b)):
    if fileindex >= 1800:
        break
    if dirlist_b[fileindex].endswith('.png'):
        col1_filenum.append(dirlist_b[fileindex][:-4]) # removes the .png part of the filename, leaving just the number.
        col2_category.append(1)
        col3_clefname.append('bass')

d = {"filenum": col1_filenum
     , "category": col2_category
     , "clefname": col3_clefname}

df = pd.DataFrame(d)
print(df.head())
print(df.tail())
df.to_csv('media\\clefdata\\lines\\cleflabels.csv', index=False)


img = cv.imread(f'media\\clefdata\\lines\\treble\\{dirlist[0]}'
                , cv.IMREAD_GRAYSCALE)

imgheight, imgwidth = img.shape

cropimg = img[:imgheight, :imgheight]

resizeimg = cv.resize(cropimg, (50,50), interpolation=cv.INTER_NEAREST)

cv.imshow("Displayed Image", resizeimg)
cv.waitKey(0)
cv.destroyAllWindows()
"""