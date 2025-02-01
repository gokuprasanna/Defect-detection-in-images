# Defect-detection-in-images

After you clone the repository create a python virtual environment and install the required packages

## System config

OS: WSL2, Ubuntu 24.04.01

Trained on Nvidia RTX 2080ti (11gb VRAM)

intel core i5 9600k

32gb DDR4 RAM

## How to set up
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

## Start the work flow
python3 run.py

I recommend training for 5000 epochs, if GPU runs out of memory reduce the batch_size until the model fits in the GPU. 

If you have a trained model (.pth file of the model parameters) just run "python3 main_eval.py" to evaluate the model and get the results.


To-Do: Optimize workflow and test with different models.
