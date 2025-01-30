import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Load trained model
MODEL_PATH = "fabric_defect_cnn.pth"
IMG_SIZE = 128

class FabricDefectCNN(nn.Module):
    def __init__(self):
        super(FabricDefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * (IMG_SIZE // 2) * (IMG_SIZE // 2), 128)
        self.fc2 = nn.Linear(128, 2)  # Two classes: Normal, Defective
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model = FabricDefectCNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open video stream
VIDEO_PATH = "fabric_conveyor_simulation.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('fabric_defect_evaluation.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to model input format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = "Defective" if predicted.item() == 1 else "Normal"
    
    # Display result
    cv2.putText(frame, f"Status: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if label == "Defective" else (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow("Fabric Defect Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
