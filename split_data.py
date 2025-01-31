import os
import shutil
import random

# Define dataset directories
DATASET_DIR = "fabric_dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
DEFECTIVE_DIR = os.path.join(DATASET_DIR, "defective")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Train-test split ratio
TRAIN_RATIO = 0.8

# Create train and test directories
for category in ["normal", "defective"]:
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)

# Function to split dataset
def split_data(source_dir, train_dest, test_dest, train_ratio=0.8):
    files = [f for f in os.listdir(source_dir) if f.endswith(".png") or f.endswith(".jpg")]
    random.shuffle(files)
    split_index = int(len(files) * train_ratio)
    train_files, test_files = files[:split_index], files[split_index:]
    
    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dest, file))
    
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dest, file))
    
    print(f"Processed {len(train_files)} train and {len(test_files)} test images for {source_dir}.")

# Split normal and defective datasets
split_data(NORMAL_DIR, os.path.join(TRAIN_DIR, "normal"), os.path.join(TEST_DIR, "normal"), TRAIN_RATIO)
split_data(DEFECTIVE_DIR, os.path.join(TRAIN_DIR, "defective"), os.path.join(TEST_DIR, "defective"), TRAIN_RATIO)

print("Train-test split completed!")
