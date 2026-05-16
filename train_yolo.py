
# Make sure you have installed the ultralytics package:
# pip install ultralytics

from ultralytics import YOLO

# Path to your data.yaml file
data_yaml = 'data/training_data/dataset/data.yaml'
# Path to the YOLOv5 model config (e.g., yolov5s.yaml)
model_cfg = 'yolov5s.yaml'  # You can change to yolov5m.yaml, yolov5l.yaml, etc.

# Load a YOLOv5 model
model = YOLO(model_cfg)

# Train the model
results = model.train(data=data_yaml, epochs=50, batch=16)
# Save the trained model (best weights)
best_model_path = model.ckpt_path if hasattr(model, 'ckpt_path') else 'runs/detect/train/weights/best.pt'
print(f"Best model saved at: {best_model_path}")
# Optionally, export the model to ONNX, TorchScript, etc.
#model.export(format='onnx')
