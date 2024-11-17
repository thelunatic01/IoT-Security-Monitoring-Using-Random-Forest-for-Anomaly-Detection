# src/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv("\data\RT_IOT2022.csv")
    
    # Drop missing values
    data = data.dropna()
    # Select relevant features
    features = data[['feature1', 'feature2', 'feature3']]  # Update with actual features
    
    # Normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled

if __name__ == "__main__":
    data_path = '../data/RT_IOT2022.csv'
    processed_data = load_and_preprocess_data(data_path)
    print("Data preprocessing complete.")
