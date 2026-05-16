from ultralytics import YOLO
import cv2
from PIL import Image
import sys
import os

# Path to the trained model (update if needed)
MODEL_PATH = 'runs/detect/train5/weights/best.pt'

def predict_image(model, image_path, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    results = model(image_path)
    results.save(save_dir)
    print(f"Prediction saved to {save_dir}")

def predict_video(model, video_path, save_dir='predictions'):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = os.path.join(save_dir, 'predicted_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        results = model(pil_img)
        # Get result image
        result_img = results[0].plot()
        # Convert back to BGR for OpenCV
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)
    cap.release()
    out.release()
    print(f"Predicted video saved to {out_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv5 Prediction Script')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.image:
        predict_image(model, args.image)
    elif args.video:
        predict_video(model, args.video)
    else:
        print('Please provide --image or --video argument.')

if __name__ == '__main__':
    main()
