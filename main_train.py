import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from model import *
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Hyperparameters
BATCH_SIZE = config["model"]["batch_size"]
LEARNING_RATE = config["model"]["learning_rate"]
EPOCHS = config["model"]["epochs"]
IMG_SIZE = config["model"]["img_size"]  # Resizing images for CNN input

# Dataset paths
DATASET_DIR = "fabric_dataset"
TRAIN_DIR = f"{DATASET_DIR}/train"
TEST_DIR = f"{DATASET_DIR}/test"

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load datasets
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Initialize model, loss, and optimizer|
model = FabricDefectCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(model)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
         print(f"Device:{device}, Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "fabric_defect_cnn.pth")
print("Model training complete and saved!")
