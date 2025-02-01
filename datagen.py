import cv2
import numpy as np
import os
import random
import shutil
import json

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Get video settings from config
FRAME_WIDTH = config["video"]["frame_width"]
FRAME_HEIGHT = config["video"]["frame_height"]
FPS = config["video"]["fps"]
VIDEO_DURATION = config["video"]["video_duration"]
FRAME_COUNT = FPS * VIDEO_DURATION
OUTPUT_VIDEO_PATH = config["video"]["output_video_path"]

# Directory structure
DATASET_DIR = "fabric_dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
DEFECTIVE_DIR = os.path.join(DATASET_DIR, "defective")
# Create directories if they don't exist
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DEFECTIVE_DIR, exist_ok=True)

# Directory for splitting data
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Train-test split ratio
TRAIN_RATIO = config["dataset"]["train_ratio"]

# Create train and test directories
for category in ["normal", "defective"]:
    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, category), exist_ok=True)


# Image parameters
IMG_WIDTH, IMG_HEIGHT = config["dataset"]["img_width"], config["dataset"]["img_height"] # Image size
NUM_SAMPLES = config["dataset"]["num_samples"]  # Number of images per category
NUM_DEFECTS = config["dataset"]["num_defects"]  # Number of defect clusters per image
DEFECT_RADIUS = config["dataset"]["defect_radius"]  # Max radius of each defect cluster
DEFECT_INTENSITY = config["dataset"]["defect_intensity"]  # Darker pixel intensity for defects
FABRIC_COLORS = config["dataset"]["fabric_colors"]  # Gray, Brown, Blue # we can change to any color here


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


    print("Train-test split completed!")

# Video settings
FRAME_WIDTH, FRAME_HEIGHT = config["video"]["frame_width"], config["video"]["frame_height"]
FPS = config["video"]["fps"]
VIDEO_DURATION = config["video"]["video_duration"]  # seconds
FRAME_COUNT = FPS * VIDEO_DURATION
OUTPUT_VIDEO_PATH = config["video"]["output_video_path"]

# Get images from dataset
def load_images(folder):
    return [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(".png")]

def make_video():
    normal_images = load_images(NORMAL_DIR)
    defective_images = load_images(DEFECTIVE_DIR)
    all_images = normal_images + defective_images
    if not all_images:
        raise ValueError("No images found in the dataset directories.")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    # Generate video frames
    for i in range(FRAME_COUNT):
        img_path = np.random.choice(all_images)  # Randomly select an image
        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        out.write(frame)

    # Release resources
    out.release()
    print(f"Video saved as {OUTPUT_VIDEO_PATH}")


def gen_data():
    # create images representing normal and defective fabric
    save_images()
    # Split normal and defective datasets
    split_data(NORMAL_DIR, os.path.join(TRAIN_DIR, "normal"), os.path.join(TEST_DIR, "normal"), TRAIN_RATIO)
    split_data(DEFECTIVE_DIR, os.path.join(TRAIN_DIR, "defective"), os.path.join(TEST_DIR, "defective"), TRAIN_RATIO)
    # make a video with the images that simulates the movement of fabric on a conveyor belt
    make_video()

if __name__ == "__main__":
    gen_data()