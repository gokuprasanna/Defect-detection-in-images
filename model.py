import torch.nn as nn
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Hyperparameters

BATCH_SIZE = config["model"]["batch_size"]
LEARNING_RATE = config["model"]["learning_rate"]
EPOCHS = config["model"]["epochs"]
IMG_SIZE =  config["model"]["img_size"]  # Resizing images for CNN input


# Define CNN Model
class FabricDefectCNN(nn.Module):
    def __init__(self):
        super(FabricDefectCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, config["model"]["conv1_channels"], kernel_size=config["model"]["kernel_size"], stride=1, padding=1)
        self.conv2 = nn.Conv2d(config["model"]["conv1_channels"], config["model"]["conv2_channels"], kernel_size=config["model"]["kernel_size"], stride=1, padding=1)
        # self.conv3 = nn.Conv2d(config["model"]["conv2_channels"], config["model"]["conv3_channels"], kernel_size=config["model"]["kernel_size"], stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=config["model"]["pool_size"], stride=2, padding=0)
        self.fc1 = nn.Linear(config["model"]["conv2_channels"] * (IMG_SIZE // 2) * (IMG_SIZE // 2), config["model"]["fc1_units"])
        self.fc2 = nn.Linear(config["model"]["fc1_units"], 2)  # Two classes: Normal, Defective
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config["model"]["dropout_rate"])


    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        # x = self.pool(self.relu(self.conv2(x)))
        # x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x