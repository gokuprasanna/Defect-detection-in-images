import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import *
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Load trained model
MODEL_PATH = "fabric_defect_cnn.pth"
IMG_SIZE = config["model"]["img_size"]

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

# Load test dataset
TEST_DIR = "fabric_dataset/test"
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Evaluate model
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy:.2f}%')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Defective"],
            yticklabels=["Normal", "Defective"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.savefig("conf_matrix.png")

# Visualizing Weights
conv1_weights = model.conv1.weight.data.cpu().numpy()
fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    if i < conv1_weights.shape[0]:
        ax.imshow(conv1_weights[i, 0], cmap='gray')
        ax.axis('off')
plt.suptitle("Conv1 Weights Visualization")
plt.show()
plt.savefig("conv1_weights.png")

# Generate evaluation video
VIDEO_PATH = "fabric_conveyor_simulation.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('fabric_defect_evaluation.mp4', fourcc, 10, (frame.shape[1], frame.shape[0]))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = "Defective" if predicted.item() == 1 else "Normal"

    cv2.putText(frame, f"Status: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if label == "Defective" else (0, 255, 0), 2)
    out.write(frame)
    cv2.imshow("Fabric Defect Evaluation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Evaluation visualization complete.")
