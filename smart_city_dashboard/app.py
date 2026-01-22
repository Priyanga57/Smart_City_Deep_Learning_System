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
# Resolve Base Paths (Streamlit Cloud Safe)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "sample_data")

# ==================================================
# Custom Styling
# ==================================================
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            font-weight: 800;
            color: #1F618D;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# Header
# ==================================================
st.markdown('<div class="title">🏙️ Smart City Deep Learning Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered Traffic Monitoring, Complaint Classification & Energy Forecasting</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==================================================
# Sidebar – System Status
# ==================================================
st.sidebar.header("📂 System Status")

try:
    st.sidebar.success("Models Loaded")
    st.sidebar.write(os.listdir(MODELS_DIR))
    st.sidebar.success("Data Available")
    st.sidebar.write(os.listdir(DATA_DIR))
except Exception as e:
    st.sidebar.error("Required files missing")
    st.sidebar.exception(e)
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Smart City Modules**
    - 🚦 Traffic Object Detection (YOLO)
    - 📝 Complaint Classification (BiLSTM)
    - ⚡ Energy Forecasting (LSTM)
    """
)

# ==================================================
# Tabs
# ==================================================
tab1, tab2, tab3 = st.tabs([
    "🚦 Traffic Monitoring",
    "📝 Complaint Classification",
    "⚡ Energy Forecasting"
])

# ==================================================
# 🚦 MODULE 1: TRAFFIC MONITORING
# ==================================================
with tab1:
    st.subheader("🚦 Real-Time Traffic Object Detection")

    uploaded_image = st.file_uploader(
        "Upload a traffic image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(uploaded_image.read())
                image_path = tmp.name

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
                st.error("🚨 HIGH TRAFFIC CONGESTION")
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
with tab2:
    st.subheader("📝 Citizen Complaint Classification")

    st.markdown(
        "Classify citizen complaints using a **BiLSTM-based NLP model**."
    )

    complaint_text = st.text_area(
        "Enter complaint text",
        placeholder="Example: Frequent power cuts in my area during night hours."
    )

    if st.button("🔍 Classify Complaint"):
        if complaint_text.strip():
            try:
                category = predict_complaint(complaint_text)
                st.success(f"📌 Predicted Category: **{category}**")
            except Exception as e:
                st.error("Complaint classification failed")
                st.exception(e)
        else:
            st.warning("Please enter a complaint")

# ==================================================
# ⚡ MODULE 3: ENERGY FORECASTING
# ==================================================
with tab3:
    st.subheader("⚡ Energy Consumption Forecasting")

    energy_csv = os.path.join(DATA_DIR, "energy_sample.csv")

    try:
        df = pd.read_csv(energy_csv)

        col1, col2, col3 = st.columns(3)
        mae = abs(df["Actual Energy"] - df["Predicted Energy"]).mean()

        col1.metric("📉 Mean Absolute Error", f"{mae:.4f}")
        col2.metric("⚡ Peak Consumption", f"{df['Actual Energy'].max():.2f}")
        col3.metric("📊 Average Consumption", f"{df['Actual Energy'].mean():.2f}")

        st.markdown("### 📈 Actual vs Predicted Energy")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Actual Energy"], label="Actual", linewidth=2)
        ax.plot(df["Predicted Energy"], label="Predicted", linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.markdown("### 📉 Prediction Error Over Time")

        fig2, ax2 = plt.subplots(figsize=(12, 3))
        ax2.plot(abs(df["Actual Energy"] - df["Predicted Energy"]), color="red")
        ax2.set_ylabel("Absolute Error")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

        if mae < 0.01:
            st.success("✅ Energy forecast is highly accurate")
        else:
            st.warning("⚠️ Forecast deviation detected")

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
