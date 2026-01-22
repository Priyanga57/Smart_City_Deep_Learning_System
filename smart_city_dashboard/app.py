import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from utils.traffic import detect_traffic
from utils.complaints import predict_complaint

st.set_page_config(
    page_title="Smart City Deep Learning System",
    layout="wide"
)

st.title("🏙️ Smart City Deep Learning Dashboard")
st.write("Integrated AI system for Traffic, Complaints & Energy Forecasting")

# =======================
# DEBUG CHECK (SAFE)
# =======================
st.sidebar.header("📂 System Check")
st.sidebar.write("Models:", os.listdir("models"))
st.sidebar.write("Sample Data:", os.listdir("sample_data"))

tabs = st.tabs([
    "🚦 Traffic Monitoring",
    "📝 Complaint Classification",
    "⚡ Energy Forecasting"
])

# ==================================================
# 🚦 TRAFFIC MONITORING
# ==================================================
with tabs[0]:
    st.header("🚦 Traffic Object Detection (YOLO)")

    uploaded_file = st.file_uploader(
        "Upload traffic image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            img_path = tmp.name

        counts, total, congestion, annotated_img = detect_traffic(img_path)

        st.image(annotated_img, use_column_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cars", counts.get("car", 0))
        c2.metric("Buses", counts.get("bus", 0))
        c3.metric("Trucks", counts.get("truck", 0))
        c4.metric("Bikes", counts.get("motorbike", 0))

        st.metric("Total Vehicles", total)

        if congestion == "High":
            st.error("🚨 HIGH CONGESTION")
        elif congestion == "Medium":
            st.warning("⚠️ MEDIUM CONGESTION")
        else:
            st.success("✅ TRAFFIC NORMAL")

# ==================================================
# 📝 COMPLAINT CLASSIFICATION
# ==================================================
with tabs[1]:
    st.header("📝 Complaint Classification (BiLSTM)")

    complaint = st.text_area(
        "Enter citizen complaint",
        placeholder="Example: Frequent power cuts in my area"
    )

    if st.button("Classify Complaint"):
        if complaint.strip():
            category = predict_complaint(complaint)
            st.success(f"Predicted Category: **{category}**")
        else:
            st.warning("Please enter complaint text")

# ==================================================
# ⚡ ENERGY FORECASTING
# ==================================================
with tabs[2]:
    st.header("⚡ Energy Consumption Forecasting")

    df = pd.read_csv("sample_data/energy_sample.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Actual Energy"], label="Actual")
    ax.plot(df["Predicted Energy"], label="Predicted")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Consumption")

    st.pyplot(fig)
