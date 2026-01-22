from ultralytics import YOLO

# Load YOLO model on CPU (MANDATORY for Streamlit Cloud)
model = YOLO("models/yolov8_best.pt")
model.to("cpu")

VEHICLE_CLASSES = ["car", "bus", "truck", "motorbike", "pickup-van", "microbus"]

def detect_traffic(image_path):
    results = model(image_path, device="cpu")[0]

    counts = {v: 0 for v in VEHICLE_CLASSES}

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        if label in counts:
            counts[label] += 1

    total = sum(counts.values())

    if total > 50:
        congestion = "High"
    elif total > 20:
        congestion = "Medium"
    else:
        congestion = "Low"

    annotated_img = results.plot()

    return counts, total, congestion, annotated_img
