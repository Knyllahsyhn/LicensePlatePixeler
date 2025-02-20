import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11n.pt")

def detect_license_plates(image):
    results = model(image)
    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append((x1, y1, x2, y2))
    return plates
