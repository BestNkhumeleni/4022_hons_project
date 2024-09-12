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
import csv
from imblearn.over_sampling import SMOTE


video_path = "/home/best/Desktop/EEE4022S/Data/Raw_Videos/"
smote = SMOTE()

def read_bitrate_from_csv(csv_file_path):
    """Read the bitrate from the CSV file."""
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        bitrates = [float(row['bitrate']) for row in reader]
        bitrate = bitrates[0] * 0.000125 
    return bitrate

def generate_json(resolution, fps, bitrate, duration, start):
    data = {
        "I11": {
            "segments": [
                {
                    "bitrate": 0,
                    "codec": "aaclc",
                    "duration": 0,
                    "start": 0
                }
            ],
            "streamId": 42
        },
        "I13": {
            "segments": [
                {
                    "bitrate": bitrate,
                    "codec": "h264",
                    "duration": duration,
                    "fps": fps,
                    "resolution": resolution,
                    "start": start
                }
            ],
            "streamId": 42
        },
        "I23": {
            "stalling": [],
            "streamId": 42
        },
        "IGen": {
            "device": "pc",
            "displaySize": "1920x1080",
            "viewingDistance": "150cm"
        }
    }
    
    with open('output.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


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
    #
    height = str(height) + "p"
    #print(height)
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
    # print(x,y)
    X_resampled, y_resampled = smote.fit_resample(x, y) #generates synthetic data using smote.
    return X_resampled, y_resampled


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
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
   
    
    unseen_df = pd.read_csv(testing_data)
    unseen_features = unseen_df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
    unseen_features_scaled = scaler.transform([unseen_features])

    # Predict the resolution
    predicted_resolution = model.predict(unseen_features_scaled)
    predicted_label = label_encoder.inverse_transform(predicted_resolution)

    print("Predicted Resolution:", predicted_label[0])
    print()
    
    return predicted_label[0]

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
    X_resampled = pd.DataFrame(features)
    y_resampled = pd.Series(labels)
    X, y = smote.fit_resample(X_resampled, y_resampled) #generates synthetic data using smote
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
    print("FPS accuracy:", accuracy_score(y_test, y_pred)*100 ,end = "% \n")
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    

    # Predict on a new CSV file
    unseen_df = pd.read_csv(testing_data)
    unseen_features = unseen_df[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)
    unseen_features_scaled = scaler.transform([unseen_features])

    predicted_frame_rate = model.predict(unseen_features_scaled)
    print("Predicted Frame Rate:", predicted_frame_rate[0])
    print()
    return predicted_frame_rate[0]

training_data = "/home/best/Desktop/EEE4022S/Data/training_data/"
testing_data = "test_480p.csv"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

x,y = extract_features_resolution(training_data)
res = random_forest_model_resolution(x,y,testing_data)
fps = random_forest_model_fps(training_data, testing_data)
bitrate = read_bitrate_from_csv(testing_data)

#print(type(fps))
if res == "720p":
    res = "1280x720"
elif res == "1080p":
    res = "1920x1080"
elif res == "360p":
    res = "640x360"
elif res == "480p":
    res = "854x480"

generate_json(res,int(fps),bitrate,30,0)
os.system("python3 -m itu_p1203 output.json")