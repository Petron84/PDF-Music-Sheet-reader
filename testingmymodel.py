import torch
import cv2 as cv
from actionmodelbuilder import ActionModel  # Import your architecture

# 1. Initialize the model
model = ActionModel()

# 2. Load the saved weights from the .pth file
model.load_state_dict(torch.load('models\\action_model.pth', weights_only=True))

# 3. Set to evaluation mode for inference
model.eval()

img = cv.imread('media\\linenotes\\4_29_treble_pdf2png(2)_v29.png')
#lets pad with white pixels
target_height = 395
target_width = 169
img = cv.copyMakeBorder(img, 0, target_height - img.shape[0], 0, target_width - img.shape[1], cv.BORDER_CONSTANT, value=[255, 255, 255]) 
img_tensor = torch.from_numpy(img).float()
img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
# Use the model for predictions
logits = model(img_tensor)

# Convert logits to probabilities (if needed)
probabilities = torch.nn.functional.softmax(logits, dim=1)
predicted_index = torch.argmax(probabilities, dim=1).item()
print(predicted_index)