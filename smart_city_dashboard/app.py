import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from utils.traffic import detect_traffic
from utils.complaints import predict_complaint

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Smart City Deep Learning System",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# Resolve Base Paths (Cloud Safe)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

# ==================================================
# Header Section
# ==================================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            color: #2E86C1;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
        }
        .metric-box {
            background-color: #F4F6F6;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🏙️ Smart City Deep Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered Traffic Monitoring, Complaint Classification & Energy Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# Sidebar – System Health Check
# ==================================================
st.sidebar.header("📂 System Status")

try:
    st.sidebar.success("Models Loaded")
    st.sidebar.write(os.listdir(MODELS_DIR))
    st.sidebar.success("Data Available")
    st.sidebar.write(os.listdir(DATA_DIR))
except Exception as e:
    st.sidebar.error("System files missing")
    st.sidebar.exception(e)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Smart City Modules**
    - 🚦 Traffic Monitoring (YOLO)
    - 📝 Complaint Classification (BiLSTM)
    - ⚡ Energy Forecasting (LSTM)
    """
)

# ==================================================
# Tabs
# ==================================================
tabs = st.tabs([
    "🚦 Traffic Monitoring",
    "📝 Complaint Classification",
    "⚡ Energy Forecasting"
])

# ==================================================
# 🚦 MODULE 1: TRAFFIC MONITORING
# ==================================================
with tabs[0]:
    st.subheader("🚦 Real-Time Traffic Object Detection")

    uploaded_image = st.file_uploader(
        "Upload a traffic image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_image.read())
            image_path = tmp.name

        try:
            counts, total, congestion, annotated_img = detect_traffic(image_path)

            st.image(
                annotated_img,
                caption="YOLO Detection Output",
                use_column_width=True
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("🚗 Cars", counts.get("car", 0))
            c2.metric("🚌 Buses", counts.get("bus", 0))
            c3.metric("🚚 Trucks", counts.get("truck", 0))
            c4.metric("🏍️ Bikes", counts.get("motorbike", 0))

            st.metric("🚦 Total Vehicles", total)

            if congestion == "High":
                st.error("🚨 HIGH CONGESTION DETECTED")
            elif congestion == "Medium":
                st.warning("⚠️ MODERATE CONGESTION")
            else:
                st.success("✅ TRAFFIC FLOW NORMAL")

        except Exception as e:
            st.error("Traffic detection failed")
            st.exception(e)

# ==================================================
# 📝 MODULE 2: COMPLAINT CLASSIFICATION
# ==================================================
with tabs[1]:
    st.subheader("📝 Citizen Complaint Classification")

    st.markdown(
        "Automatically categorize citizen complaints using **BiLSTM-based NLP model**."
    )

    complaint_text = st.text_area(
        "Enter complaint text",
        placeholder="Example: There are frequent power cuts in my locality at night."
    )

    if st.button("🔍 Classify Complaint"):
        if complaint_text.strip():
            try:
                category = predict_complaint(complaint_text)
                st.success(f"📌 Predicted Complaint Category: **{category}**")
            except Exception as e:
                st.error("Complaint classification failed")
                st.exception(e)
        else:
            st.warning("Please enter a complaint before submitting")

# ==================================================
# ⚡ MODULE 3: ENERGY FORECASTING
# ==================================================
with tabs[2]:
    st.subheader("⚡ Energy Consumption Forecasting")

    st.markdown(
        "Visualization of **Actual vs Predicted** energy consumption using LSTM."
    )

    energy_csv = os.path.join(DATA_DIR, "energy_sample.csv")

    try:
        df = pd.read_csv(energy_csv)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Actual Energy"], label="Actual Energy", linewidth=2)
        ax.plot(df["Predicted Energy"], label="Predicted Energy", linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy Consumption")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    except Exception as e:
        st.error("Energy forecasting data not available")
        st.exception(e)

# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.markdown(
    "<center>© 2026 | Smart City Deep Learning System | Built with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)
