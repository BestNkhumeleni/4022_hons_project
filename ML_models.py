import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import re


import cv2
import subprocess
import json

video_path = "/home/best/Desktop/EEE4022S/Data/Raw_Videos/"

def get_video_info(video_path):
    # Get video resolution and framerate using OpenCV
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    
    # Resolution
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Frame rate
    fps = video.get(cv2.CAP_PROP_FPS)
    
    video.release()
    
    return width, height, fps

def get_ffprobe_metadata(video_path):
    # Use ffprobe to get more detailed metadata if needed
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)



def random_forest_model_resolution(X,y, testing_data):
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
    print("Resolution accuracy:", accuracy_score(y_test, y_pred)*100 ,end = "% \n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
   
    
    unseen_df = pd.read_csv(testing_data)
    unseen_features = unseen_df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
    unseen_features_scaled = scaler.transform([unseen_features])

    # Predict the resolution
    predicted_resolution = model.predict(unseen_features_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_resolution)

    print("Predicted Resolution:", predicted_label[0])
    print()

def extract_features_resolution(folder_path):
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
                videos_path = video_path + filename[:-4] +".mp4"
                width, resolution, fps = get_video_info(videos_path)

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


def random_forest_model_fps(folder_path, testing_data):
    # Folder containing the CSV files
    

    # Initialize lists to hold the features and labels
    features = []
    labels = []

    # Regular expression to extract resolution from filename
    resolution_pattern = re.compile(r'(360p|480p|720p|1080p)')
    frame_rate_30_pattern = re.compile(r'30')
    frame_rate_60_pattern = re.compile(r'60')

    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # Extract the resolution from the filename
            resolution_match = resolution_pattern.search(filename)
            if resolution_match:
                resolution = resolution_match.group(0)
            else:
                videos_path = video_path + filename[:-4] +".mp4"
                width, resolution, frame_rate = get_video_info(videos_path)
            # Determine the ground truth frame rate based on the filename
            if resolution in ['1080p', '720p']:
                if resolution == '720p' and frame_rate_30_pattern.search(filename):
                    frame_rate = 30
                    # print("were in")
                else:
                    frame_rate = 60
            else:
                frame_rate = 60 if frame_rate_60_pattern.search(filename) else 30
                
            
            
            frame_rate = round(frame_rate)

            # Read the CSV file
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Assuming the features are columns named 'bitrate', 'total_bytes', etc.
            feature_set = df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
            
            # Append the features and corresponding frame rate label
            features.append(feature_set)
            labels.append(frame_rate)

    # Convert lists to DataFrame and Series
    X = pd.DataFrame(features)
    y = pd.Series(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    print("Accuracy:", accuracy_score(y_test, y_pred)*100 ,end = "% \n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    

    # Predict on a new CSV file
    unseen_df = pd.read_csv(testing_data)
    unseen_features = unseen_df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
    unseen_features_scaled = scaler.transform([unseen_features])

    predicted_frame_rate = model.predict(unseen_features_scaled)
    print("Predicted Frame Rate:", predicted_frame_rate[0])
    print()

training_data = "/home/best/Desktop/EEE4022S/Data/training_data/"
testing_data = "/home/best/Desktop/EEE4022S/Data/testing_data/testdata_4.csv"


x,y = extract_features_resolution(training_data)
random_forest_model_resolution(x,y,testing_data)
random_forest_model_fps(training_data, testing_data)