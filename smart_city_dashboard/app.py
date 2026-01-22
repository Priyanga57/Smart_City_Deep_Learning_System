import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from utils.traffic import detect_traffic
from utils.complaints import predict_complaint

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart City Deep Learning System",
    layout="wide"
)

st.title("🏙️ Smart City Deep Learning Dashboard")
st.write(
    "An integrated AI system for **Traffic Monitoring**, "
    "**Complaint Classification**, and **Energy Forecasting**"
)

# --------------------------------------------------
# Resolve base directory safely
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

# --------------------------------------------------
# Sidebar – System Check (Prevents Silent Crash)
# --------------------------------------------------
st.sidebar.header("📂 System Check")

try:
    st.sidebar.write("Models:", os.listdir(MODELS_DIR))
    st.sidebar.write("Sample Data:", os.listdir(DATA_DIR))
except Exception as e:
    st.sidebar.error("Required files missing")
    st.sidebar.exception(e)
    st.stop()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs([
    "🚦 Traffic Monitoring",
    "📝 Complaint Classification",
    "⚡ Energy Forecasting"
])

# ==================================================
# 🚦 MODULE 1: TRAFFIC OBJECT DETECTION (YOLO)
# ==================================================
with tabs[0]:
    st.header("🚦 Traffic Object Detection")

    uploaded_file = st.file_uploader(
        "Upload a traffic image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            image_path = tmp.name

        try:
            counts, total, congestion, annotated_img = detect_traffic(image_path)

            st.image(
                annotated_img,
                caption="YOLO Detection Output",
                use_column_width=True
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cars", counts.get("car", 0))
            c2.metric("Buses", counts.get("bus", 0))
            c3.metric("Trucks", counts.get("truck", 0))
            c4.metric("Bikes", counts.get("motorbike", 0))

            st.metric("Total Vehicles", total)

            if congestion == "High":
                st.error("🚨 HIGH TRAFFIC CONGESTION")
            elif congestion == "Medium":
                st.warning("⚠️ MEDIUM CONGESTION")
            else:
                st.success("✅ TRAFFIC FLOW NORMAL")

        except Exception as e:
            st.error("Traffic detection failed")
            st.exception(e)

# ==================================================
# 📝 MODULE 2: COMPLAINT CLASSIFICATION (BiLSTM)
# ==================================================
with tabs[1]:
    st.header("📝 Complaint Classification")

    complaint_text = st.text_area(
        "Enter a citizen complaint",
        placeholder="Example: Frequent power cuts in my locality"
    )

    if st.button("Classify Complaint"):
        if complaint_text.strip():
            try:
                category = predict_complaint(complaint_text)
                st.success(f"📌 Predicted Category: **{category}**")
            except Exception as e:
                st.error("Complaint classification failed")
                st.exception(e)
        else:
            st.warning("Please enter complaint text")

# ==================================================
# ⚡ MODULE 3: ENERGY FORECASTING (LSTM)
# ==================================================
with tabs[2]:
    st.header("⚡ Energy Consumption Forecasting")

    energy_csv = os.path.join(DATA_DIR, "energy_sample.csv")

    try:
        df = pd.read_csv(energy_csv)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["Actual Energy"], label="Actual")
        ax.plot(df["Predicted Energy"], label="Predicted")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy Consumption")
        ax.legend()

        st.pyplot(fig)

    except Exception as e:
        st.error("Energy forecasting data not available")
        st.exception(e)
