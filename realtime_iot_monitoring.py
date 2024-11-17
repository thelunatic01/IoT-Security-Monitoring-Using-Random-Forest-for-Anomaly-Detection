# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit App for Real-Time Monitoring
st.title("IoT Security Monitoring - Enhanced Report with Precautions")
st.write("This app detects anomalies in IoT network traffic using a pre-trained Random Forest model.")

# Load the trained Random Forest model
try:
    rf_model = joblib.load('random_forest_model.pkl')  # Ensure the model file exists in the directory
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'random_forest_model.pkl' not found! Please train and save the model.")
    st.stop()

# Predefined precautions for each anomaly type
precautions = {
    1: "Check for unauthorized access to the network.",
    2: "Monitor device configurations and logs for unusual activity.",
    3: "Inspect and secure MQTT configurations.",
    4: "Ensure encryption protocols are up-to-date.",
    5: "Update firmware on all IoT devices.",
    6: "Implement stronger authentication mechanisms.",
    7: "Analyze traffic patterns for potential Distributed Denial of Service (DDoS) attacks.",
    8: "Restrict access to sensitive devices or systems.",
    9: "Perform a full network scan for vulnerabilities.",
    10: "Isolate the affected devices from the network temporarily."
}

# File upload for IoT dataset
uploaded_file = st.file_uploader("Upload IoT Traffic Data (CSV)", type="csv")
if uploaded_file is not None:
    # Load and preprocess the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset loaded successfully. Preprocessing the data...")

    try:
        # Drop unnecessary columns
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')

        # Perform one-hot encoding for categorical variables
        categorical_columns = ['proto', 'service']  # Adjust based on your dataset
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        # Handle missing columns (new or unseen categories)
        model_features = rf_model.feature_names_in_  # Features used during training
        missing_features = set(model_features) - set(df.columns)
        for col in missing_features:
            df[col] = 0  # Add missing columns with default value

        # Ensure column order matches training data
        df = df[model_features]

        st.write("Data preprocessing completed successfully!")

        # Make predictions on all rows at once
        predictions = rf_model.predict(df)

        # Count anomalies and normal instances
        anomaly_indices = np.where(predictions != 0)[0]
        anomaly_count = len(anomaly_indices)
        total_instances = len(predictions)
        normal_count = total_instances - anomaly_count

        # Display summary results
        st.subheader("Summary of Predictions")
        st.write(f"Total Instances Monitored: {total_instances}")
        st.write(f"Total Anomalies Detected: {anomaly_count}")
        st.write(f"Total Normal Instances: {normal_count}")

        # Enhanced output for anomaly types with precautions
        if anomaly_count > 0:
            st.subheader("Types of Anomalies Detected and Recommended Precautions")
            anomaly_types = np.unique(predictions[anomaly_indices], return_counts=True)
            anomaly_labels = [f"Anomaly Type {atype}" for atype in anomaly_types[0]]
            anomaly_data = pd.DataFrame({
                'Anomaly Type': anomaly_labels,
                'Count': anomaly_types[1],
                'Precaution': [precautions.get(atype, "No specific precaution defined.") for atype in anomaly_types[0]]
            })
            st.write(anomaly_data)

            # Bar chart for anomaly types
            st.bar_chart(anomaly_data.set_index('Anomaly Type')['Count'])

            # Pie chart for anomaly distribution
            fig, ax = plt.subplots()
            ax.pie(anomaly_types[1], labels=anomaly_labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
            st.pyplot(fig)

            # Save anomalies to CSV
            st.write("Saving anomalies to 'detected_anomalies.csv'...")
            anomaly_df = pd.DataFrame({
                'Index': anomaly_indices,
                'Prediction': predictions[anomaly_indices]
            })
            anomaly_df.to_csv('detected_anomalies.csv', index=False)
            st.success("Detected anomalies saved successfully!")

    except Exception as e:
        st.error(f"Error during preprocessing or prediction: {e}")
        st.stop()

# Footer
st.write("Developed for IoT Anomaly Detection using Random Forest.")
