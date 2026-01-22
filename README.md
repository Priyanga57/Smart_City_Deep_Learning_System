# **🏙️ Smart City Deep Learning System** #

  An end-to-end AI-powered Smart City Analytics Platform that integrates Computer Vision, Natural Language Processing, and Time-Series Forecasting into a unified interactive dashboard for intelligent urban decision-making.
  
## **Project Overview** ##

 --> Modern smart cities generate massive amounts of data from traffic systems, citizen services, and energy infrastructure.
  
 --> This project builds a multi-module deep learning system that:

 --> Detects and analyzes traffic congestion

 --> Classifies citizen complaints automatically

 --> Forecasts future energy consumption

 --> Integrates all insights into a single dashboard

##  **Project Modules** ##

**Module 1:**  Traffic Object Detection using YOLO

**Module 2:**  Complaint Classification using LSTM / BiLSTM

**Module 3:**  Energy Consumption Forecasting using LSTM

**Integration & Deployment: Streamlit Dashboard**



## **Module 1: Traffic Object Detection (YOLO)** ##

# Objective #

To detect and classify vehicles in traffic images and estimate congestion levels in real time.
# Dataset #

Vehicle detection dataset (YOLO format)

# Classes: # 
   
  --> car 
  --> bus
  --> truck
  --> motorbike
  --> pickup-van
  --> microbus

# Steps Performed #

---> Dataset inspection and folder structuring (train, valid, labels)
---> Data validation using data.yaml
---> YOLOv8 model selection (yolov8n.pt)
---> Training using Ultralytics YOLO with GPU (Google Colab)
---> Evaluation using mAP, Precision, Recall
---> Saving best-performing model as best.pt
---> Renaming and organizing final model as yolov8_best.pt

# Output #

---> Trained YOLO model
---> Bounding box predictions
---> Congestion level estimation (Low / Medium / High)

## **Module 2: Complaint Classification (BiLSTM)** ##

# Objective #

  To automatically classify citizen complaints into predefined categories using NLP.

# Dataset #

Consumer Complaints Dataset
Text-based complaints with labeled product categories

# Steps Performed #

--> Data cleaning (null removal, duplicates, text normalization)
--> Label encoding of complaint categories
--> Tokenization and padding of text sequences
--> Train–test split
--> Model building using Bidirectional LSTM
--> Training on CPU
--> Evaluation using accuracy and classification report
--> Saving trained model and preprocessing objects

# Output #

Complaint category prediction from free-text input

# Saved files: #

complaint_bilstm_model.h5
tokenizer.pkl
label_encoder.pkl


## **Module 3: Energy Consumption Forecasting (LSTM)** ##

# Objective #

To forecast future energy consumption trends based on historical power usage data.

# Dataset #

Household Electric Power Consumption Dataset

# Steps Performed #

--> Data preprocessing and resampling (hourly)
--> Feature scaling using MinMaxScaler
--> Sequence creation for time-series modeling
--> LSTM model training
--> Model evaluation using loss metrics
--> Inverse scaling of predictions
--> Exporting forecast results for visualization

# Output #

--> Trained energy forecasting model
--> CSV file containing:
--> Actual energy consumption
--> Predicted energy consumption
--> File saved as energy_sample.csv


## **Integration of All Modules** ##

--> All three modules are integrated into a single Streamlit dashboard:
--> Traffic Module → Live YOLO inference on uploaded images
--> Complaint Module → Real-time text classification
--> Energy Module → Visualization of actual vs predicted energy usage
--> Each model is isolated from UI logic using utility scripts for maintainability.

## **Dashboard Deployment (Streamlit)** ##

## **Project Structure** ##


Smart_City_Deep_Learning_System/

│
├── app.py
|
├── requirements.txt
|
│
├── models/
|
│   ├── yolov8_best.pt
|
│   ├── complaint_bilstm_model.h5
|
│   ├── tokenizer.pkl
|
│   ├── label_encoder.pkl
|
│   └── energy_lstm_model.h5
|
│
├── utils/
|
│   ├── traffic.py
|
│   └── complaints.py
|
│
├── sample_data/
|
│   └── energy_sample.csv
|
│
└── README.md
|



## **Deployment via GitHub & Streamlit** ##

# Steps #

--> Push project to GitHub repository
--> Add requirements.txt with all dependencies
--> Connect GitHub repo to Streamlit Cloud
--> Select app.py as entry point
--> Deploy dashboard online

# Result #

A fully functional web-based Smart City AI dashboard accessible through a browser.

# Technologies Used #

--> Python
--> YOLOv8 (Ultralytics)
--> TensorFlow / Keras
--> LSTM & BiLSTM
--> Pandas, NumPy, Matplotlib
--> Streamlit
--> Google Colab
--> GitHub


# Key Learning Outcomes #

  --> End-to-end deep learning pipeline development
  --> Multi-model system integration
  --> Real-world dataset handling
  --> Model deployment and visualization
  --> Industry-style project structuring

 
 # Conclusion #

  This project demonstrates how AI-driven analytics can support smart city planning by combining vision, language, and time-series intelligence into a single scalable system.
