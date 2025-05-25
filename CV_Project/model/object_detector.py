from ultralytics import YOLO
import cv2

def load_yolo_model():
    model = YOLO("path_to_yolov8.pt")  # Replace with actual path
    return model

def detect_objects(model, image_path):
    results = model(image_path)
    return results[0].plot()
