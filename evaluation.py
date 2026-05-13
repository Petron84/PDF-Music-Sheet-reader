import torch
import torch.nn as nn
import cv2
# import numpy as np
import os

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
    # cv2.imshow("Preprocessed Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
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


# New data structure that attaches addtional info to each image that undergo recognition
class NoteImage:
    def __init__(self, image_name, image_path):
        self.image_name = image_name
        self.image_path = image_path
        self.type = None
        self.confidence = None
        self.pitch = None
        self.staff_type = None
        self.staff_idx = None
        self.note_idx = None

    def __repr__(self):
        return f"NoteImage(\nname= {self.image_name}, \
                            \ntype= {self.type}, \
                            \nconfidence= {self.confidence}, \
                            \npitch= {self.pitch}, \
                            \nstaff_type= {self.staff_type}, \
                            \nstaff_idx= {self.staff_idx}, \
                            \nnote_idx= {self.note_idx})"

# now that we have the data structure, we can modify the code block below to assign the predicted class 
# to the NoteImage object instead of just printing it out. This will allow us to keep track of 
# the predictions and use them later in the pipeline.
if __name__ == "__main__":
    test_folder = "media/linenotes"
    note_images = []
    for filename in os.listdir(test_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(test_folder, filename)
            results = predict(image_path)
            predicted_class = max(results, key=results.get)  # Get class with highest probability
            confidence = round(results[predicted_class], 4)  # Get confidence of the prediction
            
            note_image = NoteImage(image_name=filename, image_path=image_path)
            note_image.type = predicted_class  # Assign predicted class to the NoteImage object
            note_image.confidence = confidence  # Assign confidence to the NoteImage object

            # Assign the staff_type, staff_idx, and note_idx based on the filename:
            # Example filename format: "1_10_treble_2png(2)_v6.png"
            # This format is assumed to be: <staff_idx>_<note_idx>_<staff_type>_<song_name>.png
            parts = filename.split("_")
            if len(parts) >= 4:
                note_image.staff_idx = int(parts[0])
                note_image.note_idx = int(parts[1])
                note_image.staff_type = parts[2]

            note_images.append(note_image)
    
    for note_image in note_images:
        print(note_image)
        print("\n")

# store the data structure somewhere (like in a file) for other programs to use 
# (e.g. for pitch detection, or for later stages in the pipeline)
folder_name = "note_data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)


import json
def save_note_images(note_images, folder_name):

    # create one json file to dump all the note_images data into it
    data_list = []

    for note_image in note_images:
        data = {
            "image_name": note_image.image_name,
            "image_path": note_image.image_path,
            "type": note_image.type,
            "confidence": note_image.confidence,
            "pitch": note_image.pitch,
            "staff_type": note_image.staff_type,
            "staff_idx": note_image.staff_idx,
            "note_idx": note_image.note_idx
        }
        data_list.append(data)
    
    with open(os.path.join(folder_name, "note_images.json"), "w") as f:
        json.dump(data_list, f, indent=4)

# extra step: sort the order of the elements in the json file 
# based on the staff_idx and note_idx, this will help us to access the data later in the pipeline
note_images.sort(key=lambda x: (x.staff_idx, x.note_idx))
        
# save the note_images to json files
save_note_images(note_images, folder_name)

print(f"Saved {len(note_images)} note images to '{folder_name}/note_images.json'")