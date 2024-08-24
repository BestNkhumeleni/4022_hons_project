import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import re


def encode_and_train(X,y, testing_data):
    # Encode the resolution labels (360p, 480p, 720p, 1080p)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    unseen_df = pd.read_csv(testing_data)
    unseen_features = unseen_df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
    unseen_features_scaled = scaler.transform([unseen_features])

    # Predict the resolution
    predicted_resolution = model.predict(unseen_features_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_resolution)

    print("Predicted Resolution:", predicted_label[0])


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
testing_data = "/home/best/Desktop/EEE4022S/Data/testing_data/test_480p.csv"
x,y = extract_features(training_data)
encode_and_train(x,y,testing_data)
# unseen_data(testing_data)
#print(training_data)