import os
import cv2
from ultralytics import YOLO
from collections import Counter
import tempfile

# --------------------------------------------------
# Resolve absolute path to model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# Traffic Detection Function
# --------------------------------------------------
def detect_traffic(image_path):
    # Force correct extension for YOLO
    temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = cv2.imread(image_path)
    cv2.imwrite(temp_img.name, img)

    results = model.predict(
        source=temp_img.name,
        conf=0.4,
        device="cpu"
    )[0]

    counts = Counter()
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        counts[label] += 1

    total = sum(counts.values())

    if total > 25:
        congestion = "High"
    elif total > 10:
        congestion = "Medium"
    else:
        congestion = "Low"

    annotated = results.plot()

    return dict(counts), total, congestion, annotated
