import os
import subprocess


def run_script(script_name):
    """Runs a Python script from the current directory."""
    print(f"Running {script_name}...")
    subprocess.run(["python3", script_name], check=True)


if __name__ == "__main__":
    # Step 1: Generate Toy Dataset, split the data into train and test, and generate Conveyor Belt Simulation Video
    run_script("datagen.py")

    # Step 2: Train the CNN Model
    run_script("main_train.py")

    # Step 3: Evaluate and Visualize Model Performance, To-Do: Run Real-Time Fabric Defect Detection
    run_script("main_eval.py")

    print("ML workflow completed successfully!")
