import torch.nn as nn

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
