import os

import torch
import cv2 as cv
from actionmodelbuilder import ActionModel, pad_and_resize_transform,inference_transforms  # Import your architecture


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Initialize the model
model = ActionModel()

# 2. Load the saved weights from the .pth file
model.load_state_dict(torch.load('models\\action_model.pth', weights_only=True))

# 3. Set to evaluation mode for inference
model.eval()

for param in model.parameters():
    param.requires_grad = False
    
dir = 'media\\linenotes'
files = os.listdir(dir)
count=0
for file in files:
    print(file)
    # pulling and example image to see how it is performing.
    img = cv.imread(f'{dir}\\{file}')
    # img = pad_and_resize_transform(img)  # Apply the same padding and resizing as during training
    # if len(img.shape) == 3:
    #     img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_tensor = inference_transforms(img).unsqueeze(0).to(device)
    # img_tensor = torch.from_numpy(img).float()/255.0
    # img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    logits = model(img_tensor)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    print(probabilities)
    max_prob = torch.max(probabilities).item()
    if max_prob < 0.7:
        print(f"Warning: Model is guessing! Confidence is only {max_prob:.2f}")
    predicted_index = torch.argmax(probabilities, dim=1).item()
    print(predicted_index)
        
    window_name = f"{predicted_index} : {max_prob:.2f}"

    if predicted_index == 3:
        count+=1
        # cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        # cv.resizeWindow(window_name, 300, 300)
        # cv.imshow(window_name, img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
print(f"Total count of predicted index 3: {count}")
    

