import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import *
# Load trained model
MODEL_PATH = "fabric_defect_cnn.pth"
IMG_SIZE = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model
model = FabricDefectCNN().to(device)
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

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('fabric_defect_evaluation.avi', fourcc, 3, (IMG_SIZE, IMG_SIZE))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to model input format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        image = image.to(device)
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
