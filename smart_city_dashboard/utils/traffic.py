import os
from ultralytics import YOLO
from collections import Counter
from PIL import Image

# --------------------------------------------------
# Resolve absolute path to model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)

# --------------------------------------------------
# Traffic Detection Function
# --------------------------------------------------
def detect_traffic(image_path):

    # Open image using PIL instead of OpenCV
    img = Image.open(image_path)

    # Run YOLO detection
    results = model.predict(
        source=img,
        conf=0.4,
        device="cpu"
    )[0]

    # Count detected vehicles
    counts = Counter()
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] += 1

    total = sum(counts.values())

    # Traffic congestion logic
    if total > 25:
        congestion = "High"
    elif total > 10:
        congestion = "Medium"
    else:
        congestion = "Low"

    # Annotated output image
    annotated = results.plot()

    return dict(counts), total, congestion, annotated
