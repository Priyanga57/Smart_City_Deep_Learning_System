import os
from ultralytics import YOLO

# Get absolute path to project root (smart_city_dashboard)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8_best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found at {MODEL_PATH}"
    )

# Load YOLO on CPU (Streamlit Cloud safe)
model = YOLO(MODEL_PATH)
model.to("cpu")

VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike", "pickup-van", "microbus"]

def detect_traffic(image_path):
    results = model(image_path, device="cpu")[0]

    counts = {v: 0 for v in VEHICLE_CLASSES}

    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        if label in counts:
            counts[label] += 1

    total = sum(counts.values())

    congestion = (
        "High" if total > 50 else
        "Medium" if total > 20 else
        "Low"
    )

    return counts, total, congestion, results.plot()
