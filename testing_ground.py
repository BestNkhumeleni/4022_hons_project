import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_model(pkl_file):
    """Load a trained model from a .pkl file."""
    with open(pkl_file, 'rb') as file:
        model = pickle.load(file)
    return model

def load_label_encoder(pkl_file):
    """Load the label encoder from a .pkl file."""
    with open(pkl_file, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

def predict_from_csv(input_csv):
    # Load the pre-trained models
    resolution_model = load_model('/home/best/Desktop/EEE4022S/scripts/resolution_model.pkl')  # Replace with actual file path
    fps_model = load_model('/home/best/Desktop/EEE4022S/scripts/fps_model.pkl')  # Replace with actual file path

    # Load the label encoder for resolution
    resolution_label_encoder = load_label_encoder('/home/best/Desktop/EEE4022S/scripts/label_encoder_resolution.pkl')  # Replace with actual file path

    # Load and preprocess the new input data for prediction
    input_data = pd.read_csv(input_csv)

    # Drop the video name (or any unnecessary) column for prediction
    input_features = input_data[['bitrate', 'num_bytes', 'num_packets', 'interval', 'packet_size']].mean(axis=0)

    # Scale the input data using a scaler (assumed to have been saved during training)
    fps_scaler = load_model("/home/best/Desktop/EEE4022S/scripts/fps_scaler")
    res_scaler = load_model("/home/best/Desktop/EEE4022S/scripts/res_scaler")
    # input_features_scaled = scaler.fit_transform(input_features)  # Adjust this if the scaler was saved separately
    
    unseen_features_scaled_fps = fps_scaler.transform([input_features])
    unseen_features_scaled_res = res_scaler.transform([input_features])
    # # Predict resolution using the pre-trained model
    resolution_prediction_encoded = resolution_model.predict(unseen_features_scaled_res)[0]
    # # Inverse transform the encoded resolution prediction to get the original label
    resolution_prediction = resolution_label_encoder.inverse_transform([resolution_prediction_encoded])[0]
    
    print(f"\nPredicted Resolution: {resolution_prediction}")

    # Predict fps using the pre-trained model
    fps_prediction = fps_model.predict(unseen_features_scaled_fps)[0]
    print(f"Predicted FPS: {fps_prediction}")
    # resolution_prediction = ""
    # Return the best predictions
    return resolution_prediction, fps_prediction

if __name__ == "__main__":
    # Example usage, replace 'input_file.csv' with your actual CSV file
    for filename in os.listdir("/home/best/Desktop/EEE4022S/Data/testing_data"):
        
        predict_from_csv("/home/best/Desktop/EEE4022S/Data/testing_data/"+filename)

    # predict_from_single_input("/home/best/Desktop/EEE4022S/Data/training_data/bold_colours_720p_30_3.csv")

    # predict_from_csv()
