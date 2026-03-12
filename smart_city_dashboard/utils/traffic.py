import os
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import numpy as np

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# Traffic Detection Function
# --------------------------------------------------
def detect_traffic(image_path):

    image = Image.open(image_path).convert("RGB")

    results = model.predict(
        source=image,
        conf=0.4,
        device="cpu",
        verbose=False
    )[0]

    counts = Counter()

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] += 1

    total = sum(counts.values())

    # congestion logic
    if total > 25:
        congestion = "High"
    elif total > 10:
        congestion = "Medium"
    else:
        congestion = "Low"

    annotated_img = results.plot()

    return dict(counts), total, congestion, annotated_img
