from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')  # You can choose other models like 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model on a custom dataset
# Replace 'coco8.yaml' with your dataset configuration file
# This file defines the paths to your training and validation images/labels and class names.
# You can create your own .yaml file or use one of the built-in datasets.
results = model.train(data='coco8.yaml', epochs=100, imgsz=640)

print("Training complete! Results are saved in runs/detect/train/")

"""This code snippet demonstrates loading a `yolov8n` model and initiating its training on a dataset specified by `coco8.yaml`. You will need to replace `coco8.yaml` with your own dataset's YAML configuration file, which should define the paths to your training and validation images, labels, and class names. If you don't have a dataset ready, you can start by exploring the `ultralytics` documentation for information on preparing datasets or using their example datasets."""

import os

# List the contents of the training run directory
run_dir = '/runs/detect/train'
print(f"Contents of {run_dir}:")
for item in os.listdir(run_dir):
    print(item)

from IPython.display import Image

# Display the main results plot
results_plot_path = os.path.join('/runs/detect/train', 'results.png')
if os.path.exists(results_plot_path):
    display(Image(filename=results_plot_path))
else:
    print(f"Results plot not found at {results_plot_path}")

