import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

MAX_LEN = 150

# --------------------------------------------------
# Resolve absolute project path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "complaint_bilstm_model.h5")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# --------------------------------------------------
# Safety checks (prevents silent crash)
# --------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Complaint model not found at {MODEL_PATH}")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

if not os.path.exists(LABEL_ENCODER_PATH):
    raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# --------------------------------------------------
# Prediction function
# --------------------------------------------------
def predict_complaint(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    return label_encoder.inverse_transform([pred.argmax()])[0]
