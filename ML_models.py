import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import re


def extract_features(folder_path):
    # Initialize lists to hold the features and labels
    features = []
    labels = []

    # Regular expression to extract resolution from filename
    resolution_pattern = re.compile(r'(360p|480p|720p|1080p)')

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract the resolution from the filename
            resolution_match = resolution_pattern.search(filename)
            if resolution_match:
                resolution = resolution_match.group(0)
            else:
                continue  # Skip files without a resolution in the name

            # Read the CSV file
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Assuming the features are columns named 'bitrate', 'total_bytes', etc.
            feature_set = df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
            # ,,,,bitrate
            
            # Append the features and corresponding resolution label
            features.append(feature_set)
            labels.append(resolution)

    # Convert lists to DataFrame and Series
    x = pd.DataFrame(features)
    y = pd.Series(labels)
    return x, y


training_data = "/home/best/Desktop/EEE4022S/Data/training_data/"
x,y = extract_features(training_data)
#print(training_data)
print(x)
print(y)
