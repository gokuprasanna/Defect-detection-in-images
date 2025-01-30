import cv2
import numpy as np
import os
import random
from model import *

# Directory structure
DATASET_DIR = "fabric_dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
DEFECTIVE_DIR = os.path.join(DATASET_DIR, "defective")

# Create directories if they don't exist
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DEFECTIVE_DIR, exist_ok=True)

# Image parameters
IMG_WIDTH, IMG_HEIGHT = 256, 256  # Image size
NUM_SAMPLES = 100  # Number of images per category
NUM_DEFECTS = 5  # Number of defect clusters per image
DEFECT_RADIUS = 10  # Max radius of each defect cluster
DEFECT_INTENSITY = 80  # Darker pixel intensity for defects
FABRIC_COLORS = [(200, 200, 200), (150, 100, 50), (50, 150, 200)]  # Gray, Brown, Blue


def generate_fabric_texture(color):
    """
    Generates a synthetic fabric-like texture with a given color.
    """
    texture = np.full((IMG_HEIGHT, IMG_WIDTH, 3), color, dtype=np.uint8)
    noise = np.random.randint(-10, 10, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.int8)
    texture = np.clip(texture + noise, 0, 255).astype(np.uint8)
    texture = cv2.GaussianBlur(texture, (7, 7), 2)
    return texture


def add_defects(image, num_defects=5, radius=10, defect_type="stain"):
    """
    Adds different types of defects (stains, scratches, patterns) to the fabric texture.
    """
    defective_img = image.copy()
    
    for _ in range(num_defects):
        x_center = random.randint(radius, IMG_WIDTH - radius)
        y_center = random.randint(radius, IMG_HEIGHT - radius)
        
        if defect_type == "stain":
            for _ in range(random.randint(5, 15)):
                x_offset, y_offset = random.randint(-radius, radius), random.randint(-radius, radius)
                x, y = np.clip(x_center + x_offset, 0, IMG_WIDTH - 1), np.clip(y_center + y_offset, 0, IMG_HEIGHT - 1)
                defective_img[y, x] = (0, 0, 0)
        
        elif defect_type == "scratch":
            thickness = random.randint(1, 3)
            length = random.randint(10, 30)
            angle = random.randint(0, 360)
            x_end = int(x_center + length * np.cos(np.radians(angle)))
            y_end = int(y_center + length * np.sin(np.radians(angle)))
            cv2.line(defective_img, (x_center, y_center), (x_end, y_end), (0, 0, 0), thickness)
        
        elif defect_type == "pattern":
            cv2.circle(defective_img, (x_center, y_center), radius, (0, 0, 0), -1)
    
    return defective_img


def save_images():
    """
    Generates and saves normal and defective fabric images.
    """
    for i in range(NUM_SAMPLES):
        fabric_color = random.choice(FABRIC_COLORS)
        fabric_texture = generate_fabric_texture(fabric_color)
        
        normal_path = os.path.join(NORMAL_DIR, f"fabric_{i}.png")
        cv2.imwrite(normal_path, fabric_texture)
        
        defect_type = random.choice(["stain", "scratch", "pattern"])
        defective_texture = add_defects(fabric_texture, NUM_DEFECTS, DEFECT_RADIUS, defect_type)
        defective_path = os.path.join(DEFECTIVE_DIR, f"fabric_defect_{i}.png")
        cv2.imwrite(defective_path, defective_texture)
    
    print(f"Dataset generation complete! {NUM_SAMPLES} normal and defective images saved.")


if __name__ == "__main__":
    save_images()