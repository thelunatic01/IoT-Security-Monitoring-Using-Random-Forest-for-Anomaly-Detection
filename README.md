# IoT Security Monitoring: Anomaly Detection with Enhanced Machine Learning Techniques

![Python](https://img.shields.io/badge/python-v3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Completed-brightgreen)

## Overview
This project implements an **IoT Security Monitoring System** that detects anomalies in IoT network traffic using a **Random Forest** machine learning model. The solution includes:
- Real-time anomaly detection.
- Detailed visualizations of anomaly types.
- Actionable recommendations for mitigating potential threats.

The project provides a comprehensive monitoring solution for IoT network traffic data through a user-friendly Streamlit dashboard.

---

## Features
- **Real-Time Anomaly Detection**: Upload IoT traffic data and instantly identify anomalies.
- **Detailed Visualizations**:
  - Bar charts for anomaly counts.
  - Pie charts for anomaly type distribution.
- **Actionable Recommendations**: Mitigation strategies for detected anomalies.
- **User-Friendly Interface**: Intuitive Streamlit dashboard for easy monitoring.

---

## System Architecture
The project consists of the following components:
1. **Data Preprocessing**:
   - Handles missing values and categorical encoding.
   - Ensures feature consistency with the trained model.
2. **Model Training**:
   - Pre-trained Random Forest classifier for anomaly detection.
3. **Streamlit Dashboard**:
   - Displays real-time predictions and visual insights.

---

## Requirements
- **Python**: Version 3.8 or higher
- **RT IoT2022 Dataset**
- **Dependencies**: Install using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
