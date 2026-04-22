import torch
import torch.nn as nn
import cv2
import numpy as np

class NoteCNN(nn.Module):

    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
classes = torch.load("models/note_classes.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NoteCNN(num_classes=5)
model.load_state_dict(torch.load("models\\note_model.pth", map_location=device))
model.to(device)
model.eval()

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    img = img / 255.0

    #visualize the preprocessed image
    cv2.imshow("Preprocessed Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    tensor = torch.tensor(img).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor

def predict(image_path):
    tensor = preprocess(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    results = {}

    for i, cls in enumerate(classes):
        results[cls] = probabilities[i].item()

    return results

if __name__ == "__main__":
    image_path = "testnotes\\test6.png"
    results = predict(image_path)
    print("Prediction probabilities:\n")

    for cls, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {prob:.3f}")