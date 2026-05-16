import os
from ultralytics import YOLO
from IPython.display import Image, display

# Load the best trained model weights
model = YOLO('E:\\GIT\\yolo-test\\runs\\detect\\train\\weights\\best.pt')

# Perform inference on an image
# You can replace this URL with a local image path or a video file
# Example image URL from Ultralytics docs
source_image = 'https://ultralytics.com/images/bus.jpg'
source_image = 'https://citizen.goapolice.gov.in/documents/10184/3819336/Trafficking_Day_3.JPG'  # Local image path
source_image = 'https://as2.ftcdn.net/v2/jpg/15/12/38/37/1000_F_1512383790_eIFuZzMS7yzxVbwnreA0yalN12SWlxqy.jpg'  # Local image path
source_image = 'https://www.shutterstock.com/image-photo/big-male-loin-600w-1030573348.jpg'  # Local image path
# Run inference
results = model.predict(source=source_image, save=True, conf=0.25)

# Display the predicted image
# The results are saved in a 'predict' folder, usually 'runs/detect/predict'
# We need to find the path to the saved image.
predicted_image_path = results[0].save_dir + '/' + source_image.split('/')[-1]
print(f"Predicted image saved to: {predicted_image_path}")
if os.path.exists(predicted_image_path):
    display(Image(filename=predicted_image_path))
else:
    print("Could not find the predicted image to display.")

"""The code above loads the `best.pt` weights, which represent the best performing model from our training run. It then uses this model to run predictions on a sample image. The `save=True` argument ensures the image with detected objects and bounding boxes is saved to a `predict` directory (e.g., `runs/detect/predict`). Finally, it displays the image with the predictions.

```yaml
# my_custom_dataset.yaml

# Define the root directory of your dataset
# This path is where your 'images' and 'labels' folders are located.
# For example, if your structure is: my_dataset/images/... and my_dataset/labels/...
# then 'path' would be 'my_dataset/' or '../my_dataset/' if this YAML is in a parent directory.
path: ../datasets/my_custom_dataset/ # Adjust this to your actual dataset root directory

# Define the relative paths to your training and validation image directories
# These paths are relative to the 'path' defined above.
train: images/train  # Path to training images
val: images/val      # Path to validation images
# test: images/test  # Optional: Path to test images (uncomment if you have a test set)

# Define the number of classes (nc) and their names
# The order of names must match the class_id integers (0, 1, 2, ...)
nc: 2 # Example: 2 classes (adjust this to your actual number of classes)
names: ['object_name_1', 'object_name_2']  # Example class names (replace with your object names)
# Example with more classes:
# names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']

# Optionally, you can add descriptions or other metadata
# version: 1.0
# description: "My first custom YOLO dataset for object detection."
```

**How to use this:**

1.  **Save this content** into a file named `my_custom_dataset.yaml` (or any `.yaml` name you prefer) in a location accessible by your Colab environment, ideally in your dataset's root directory.
2.  **Ensure your dataset directory structure matches** the `path`, `train`, `val` (and `test`) definitions.
3.  **Update `nc`** with the total number of unique object classes you have.
4.  **List your object `names`** in the exact order corresponding to the `class_id` integers (0, 1, 2, ...) used in your annotation `.txt` files.

Once this YAML file is ready, you can pass it to the `model.train()` function like this:

`results = model.train(data='path/to/my_custom_dataset.yaml', epochs=100, imgsz=640)`
"""