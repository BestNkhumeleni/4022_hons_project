from mininet.net import Mininet
from mininet.node import OVSController
from mininet.log import info
from mininet.util import dumpNodeConnections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import cv2
import shutil
import csv
import pyshark
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import json
import re
import subprocess
import pickle
# encript the transmission,
# introducing some unrealiability
# specify the drop rate and error 
# simulate delays
# 

resolutions = []
fps_values = []
time_values = []
qoe_values = []


def write_to_txt(time_values, qoe_values, resolutions, fps_values, filename='graphs.txt'):
    #rint(time_values, qoe_values, resolutions, fps_values)
    if not (len(time_values) == len(qoe_values) == len(resolutions) == len(fps_values)):
        raise ValueError("All arrays must have the same length.")

    with open(filename, 'w') as file:
        file.write("Time\tQoE\tResolution\tFPS\n")  # Writing headers
        for i in range(len(time_values)):
            file.write(f"{time_values[i]}\t{qoe_values[i]}\t{resolutions[i]}\t{fps_values[i]}\n")

    print(f"Data successfully written to {filename}")


def find_number_in_qoe(filename='qoe.txt'):
    try:
        with open(filename, 'r') as file:
            for line in file:
                if '"O46": ' in line:
                    qoe_match = re.search(r'"O46": ([\d.]+)', line)
                    return float(qoe_match.group(1))
        print("O46: not found in the file.")
    except FileNotFoundError:
        print(f"The file {filename} does not exist.")


def play_video():
    video_path = "received_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while cap.isOpened():
        # Read each frame from the video
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Exit when 'q' key is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()


def read_output_file(output_txt_file):
    # Initialize variables to store the values
    packet_count = None
    average_interval = None

    # Read the output file
    with open(output_txt_file, 'r') as f:
        for line in f:
            # Check if the line contains packet count
            if "Packet Count" in line:
                packet_count = int(line.split(":")[1].strip())
            # Check if the line contains average interval
            elif "Average Interval" in line:
                average_interval = float(line.split(":")[1].strip().split()[0])

    return packet_count, average_interval


def get_packet_count_pcapng(capture_file):
    cap = pyshark.FileCapture(capture_file, use_json=True, include_raw=False)
    #capture = pyshark.FileCapture(pcap_file, use_json=True, include_raw=False)
    count = 0
    for _ in cap:
        count += 1
        #print(count)
    cap.close()
    return count


def check_file_size(file_path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)
    
    # Convert size to megabytes (1 MB = 1024 * 1024 bytes)
    size_in_mb = file_size / (1024 * 1024)
    
    # Check if the file size is less than 1 MB
    return size_in_mb


def delete_directories(directory):
    # Loop through all items in the specified directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        # Check if the item is a directory and ends with "30" or "p"
        if os.path.isdir(item_path) and ( not item.endswith("git") ):
            try:
                # Delete the directory and all its contents
                shutil.rmtree(item_path)
                print(f"Deleted directory: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")


def get_video_duration_opencv(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the duration in seconds
    duration = frame_count / fps

    # Release the video file
    video.release()

    return duration


def append_to_csv(file_name, data):
    """
    Appends the provided data to a CSV file.
    
    :param file_name: str, the name of the CSV file.
    :param data: dict, dictionary where keys are the column names and values are the data to append.
    """
    # Check if the file already exists
    file_exists = False
    try:
        with open(file_name, 'r', newline='') as file:
            file_exists = True
    except FileNotFoundError:
        pass
    
    # Open the file in append mode
    with open(file_name, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        
        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        # Write the data to the file
        writer.writerow(data)


def generate_json(resolution, fps, bitrate, duration, start):
    data = {
        "I11": {
            "segments": [
                {
                    "bitrate": 192,
                    "codec": "aaclc",
                    "duration": duration,
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
            "viewingDistance": 0
        }
    }
    
    with open('output.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)


# def predict_from_csv(input_csv):
#     # Initialize SMOTE
#     smote = SMOTE()

#     # Load the training data for resolution and fps prediction
#     X_resolution = pd.read_csv('features_resolution.csv')  # Input features for resolution
#     y_resolution = pd.read_csv('labels_resolution.csv')    # Labels for resolution
#     X_fps = pd.read_csv('features_fps.csv')  # Input features for fps
#     y_fps = pd.read_csv('labels_fps.csv')    # Labels for fps

#     # Drop unnecessary columns from the datasets
#     y_fps = y_fps.drop(columns=['emp'])
#     X_resolution = X_resolution.drop(columns=['index'])
#     y_resolution = y_resolution.drop(columns=['index', 'emp'])
#     X_fps = X_fps.drop(columns=['index'])
#     y_fps = y_fps.drop(columns=['index'])

#     # Apply SMOTE for resolution data
#     X_resolution, y_resolution = smote.fit_resample(X_resolution, y_resolution)
#     # X_fps, y_fps = smote.fit_resample(X_fps, y_fps)  # Uncomment for fps if needed

#     # Apply StandardScaler for feature scaling
#     scaler = StandardScaler()
#     X_resolution_scaled = scaler.fit_transform(X_resolution)
#     X_fps_scaled = scaler.fit_transform(X_fps)

#     # Train-test split for both resolution and fps prediction
#     X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_resolution_scaled, y_resolution, test_size=0.3, random_state=42)
#     X_train_fps, X_test_fps, y_train_fps, y_test_fps = train_test_split(X_fps_scaled, y_fps, test_size=0.3, random_state=42)

#     # Define models for both resolution and fps prediction
#     models_resolution = {
#         "Random Forest": RandomForestClassifier(),
#         "SVM": SVC(),
#         "Logistic Regression": LogisticRegression(max_iter=5000)
#     }
#     models_fps = {
#         "Random Forest": RandomForestClassifier(),
#         "SVM": SVC(),
#         "Logistic Regression": LogisticRegression(max_iter=5000)
#     }

#     # Train and evaluate models function
#     def train_and_evaluate(X_train, X_test, y_train, y_test, models):
#         trained_models = {}
#         model_accuracies = {}
#         for name, model in models.items():
#             model.fit(X_train, y_train.values.ravel())  # Flatten y for training
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             trained_models[name] = model
#             model_accuracies[name] = accuracy
#             print(f"{name} Accuracy: {accuracy:.4f}")
#         # Find the model with the highest accuracy
#         best_model_name = max(model_accuracies, key=model_accuracies.get)
#         best_model = trained_models[best_model_name]
#         return best_model, trained_models

#     # Train models for resolution prediction and get the best one
#     print("Training Resolution Models:")
#     best_resolution_model, trained_resolution_models = train_and_evaluate(X_train_res, X_test_res, y_train_res, y_test_res, models_resolution)

#     # Train models for fps prediction and get the best one
#     print("\nTraining FPS Models:")
#     best_fps_model, trained_fps_models = train_and_evaluate(X_train_fps, X_test_fps, y_train_fps, y_test_fps, models_fps)

#     # Load the new input data for prediction
#     input_data = pd.read_csv(input_csv)

#     # Drop the video name (or any unnecessary) column for prediction
#     input_features = input_data.drop(columns=['name'])

#     # Scale the input data for prediction
#     input_features_scaled = scaler.transform(input_features)

#     # Predict resolution using the best model
#     resolution_prediction = best_resolution_model.predict(input_features_scaled)[0]
#     print(f"\nBest Model predicts Resolution: {resolution_prediction}")

#     # Predict fps using the best model
#     fps_prediction = best_fps_model.predict(input_features_scaled)[0]
#     print(f"Best Model predicts FPS: {fps_prediction}")

#     # Return the best predictions
#     return resolution_prediction, fps_prediction

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


def update_and_show_plot():
    """
    Updates and shows the FPS, Resolution, and QoE plots in real-time.

    :param resolutions: List of resolution values over time
    :param fps_values: List of FPS values over time
    :param qoe_values: List of QoE values over time
    :param time_values: List of time points corresponding to resolutions, FPS, and QoE values
    """
    plt.clf()  # Clear the previous plots to update them

    # Create subplots for Resolution, FPS, and QoE
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

    # Plot resolution over time
    ax1.plot(time_values, resolutions, label='Resolution', color='blue')
    ax1.set_title('Resolution Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Resolution (px)')
    ax1.legend()

    # Plot FPS over time
    ax2.plot(time_values, fps_values, label='FPS', color='orange')
    ax2.set_title('FPS Over Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('FPS')
    ax2.legend()

    # Plot QoE over time
    ax3.plot(time_values, qoe_values, label='QoE', color='green')
    ax3.set_title('QoE Over Time')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('QoE')
    ax3.legend()

    plt.tight_layout()
    plt.draw()  # Redraw the current figure

def convert_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes} minute(s) and {remaining_seconds} second(s)"

def extract_features(pcap_file, video_path, interval, num_packets, average_interval):
    capture = pyshark.FileCapture(pcap_file, use_json=True, include_raw=False)
    video_name = os.path.basename(pcap_file)
    filename = video_name[:-5]
    
    print(f"\nAnalyzing {filename}\n")

    total_bytes = 0
    start_time = None
    end_time = None
    total_packets = 0
    packet_sizes = []
    packet_times = []
    current_time = interval
    
    #idea use the proportion of the capture to that of the video
    
    # t.tic()
    video_length = get_video_duration_opencv(video_path)
    # Initialize lists for plotting

    
    # t.toc()
    print("starting:")
    while(current_time<=video_length):
        #print("here?")
        total_bytes = 0
        total_packets = 0
        stop_packets = round(num_packets*(current_time/video_length))
        inv = video_length/current_time
        # t.tic()
        #print(current_time)
        for packet in capture:
            #print("yes")
            try:
                start = time.time()
                total_packets += 1
                packet_time = float(packet.sniff_timestamp)
                if hasattr(packet, 'frame_info'):
                    packet_len = int(packet.length)
                    total_bytes += packet_len
                    packet_sizes.append(packet_len)
                    packet_times.append(packet_time)
                    
                    if start_time is None:
                        start_time = packet_time
                    end_time = packet_time
                if total_packets >= stop_packets:
                    end = time.time()
                    # Calculate how long the iteration took
                    elapsed_time = end - start
                    # Calculate the remaining time to sleep if the iteration finished early
                    time_to_sleep = interval - elapsed_time -0.2
                    # if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                    break

                # Exit early after 30 seconds
                #print(packet_time - start_time)
            
            except AttributeError:
                continue  # Skip packets that don't have the required fields
        capture.close()
        # t.toc()
        mean_packet_size = sum(packet_sizes) / len(packet_sizes)
            
        # total_packets *= inv
        # total_bytes *= inv
        if start_time and end_time:
            duration = average_interval*stop_packets
            bit = (total_bytes * 8) / duration  # bits per second
            csvstor = {
            'name': filename,
            'bitrate': bit,
            'num_bytes': total_bytes,
            'num_packets': num_packets,
            'interval': average_interval,
            'packet_size': mean_packet_size,
            }
            print(f"Analysis complete, feature extracted for {filename} are:")
            print(csvstor)
            print()
            out_csv = filename +".csv"
            os.system(f"rm {out_csv}")
            append_to_csv(out_csv, csvstor)
            resolution,fps = predict_from_csv(out_csv)
            duration = get_video_duration_opencv(video_path)
            bitrate = (total_bytes * 8) / duration
            fps = int(fps)
            res = resolution
            
            if res == "720p":
                res = "1280x720"
            elif res == "1080p":
                res = "1920x1080"
            elif res == "360p":
                res = "640x360"
            elif res == "480p":
                res = "854x480"
            generate_json(res,fps,bitrate,current_time,0)
            print()
            
            res_value = int(resolution.replace('p', '')) if 'p' in resolution else 0
            resolutions.append(res_value)
            fps_values.append(fps)
            time_values.append(current_time)
            os.system("/home/best/miniconda3/bin/python -m itu_p1203 output.json > qoe.txt")
            qoe_values.append(find_number_in_qoe())
            print(convert_seconds(current_time))
            print(f"{current_time/duration * 100}% through the video")
            print(qoe_values[-1])
            
            write_to_txt(time_values,qoe_values,resolutions,fps_values)
            #subprocess.run(["/home/best/miniconda3/bin/python", "graphs.py"])
         
            
        current_time+=interval
        
    # duration = get_video_duration_opencv(video_path)
    # bitrate = (total_bytes * 8) / duration
    


#os.system("/home/best/miniconda3/bin/python -m itu_p1203 output.json")


def setup_mininet_and_transmit(video_file):
    # Create a network
    net = Mininet(controller=OVSController)
    
    # Add a controller
    net.addController('c0')
    
    # Add hosts
    h1 = net.addHost('h1', ip='10.0.0.1')
    h2 = net.addHost('h2', ip='10.0.0.2')
    
    # Add a switch
    s1 = net.addSwitch('s1')
    
    # Create links between hosts and switch
    link1 = net.addLink(h1, s1)
    link2 = net.addLink(h2, s1)
    
    # Start the network
    net.start()
    
    # Dump host connections
    dumpNodeConnections(net.hosts)
    
    # Path to video file on h1 (update this path as necessary)
    
    
    # Ensure the video file exists
    if not os.path.isfile(video_file):
        info("Video file not found!\n")
        net.stop()
        return
    
    # Extract the last five characters of the video file name (excluding the extension)
    video_name = os.path.basename(video_file)
    folder_name = video_name[:-4]  # This will extract the last five characters excluding the extension
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Start capturing packets on the link between h1 and s1
    capture_file = folder_name+'.pcap'
    info(f"*** Capturing packets on link1 (h1 <-> s1) to {capture_file} ***\n")
    h1.cmd(f'tcpdump -i {link1.intf1} -w {capture_file} &')
    
    # Investigate if the protocol have an impact on the qoe
    # Add encryption/ if neccesary
    # Designing a questionaire
    # UDP, stp
    
    # Start a simple video transmission using netcat
    info("*** Starting video transmission from h1 to h2 ***\n")
    
    # Start a server on h2 to receive the video
    h2.cmd('nc -l -p 12345 > received_video.mp4 &')
    
    # Send the video from h1 to h2
    h1.cmd(f'cat {video_file} | nc 10.0.0.2 12345 &')
    
    # Wait for 30 seconds
    time.sleep(5)
    
    # Stop the tcpdump capture and video transmission
    info("*** Stopping video transmission and capture after 30 seconds ***\n")
    h1.cmd('pkill -f tcpdump')
    #h1.cmd('pkill -f nc')
    #h2.cmd('pkill -f nc')
    
    # Stop the network
    net.stop()
    info("*** Mininet stopped ***\n")
    
    while check_file_size(capture_file)<1: #if it failed, keep trying until it doesnt
        print()
        print("There was a problem during capture for "+capture_file)
        # video_name = filename[:-5]
        # video_file = directory +"/" + video_name + ".mp4"
        print("Recapturing")
        os.system(f"rm {capture_file}")
        setup_mininet_and_transmit(video_file)
        print()
    return capture_file


if __name__ == "__main__":
    
    video = "/home/best/Desktop/EEE4022S/Data/Raw_Videos/bold_colours_720p_30.mp4"
    interval = 1 # How often do you want to sample the stream in seconds
    # pcap_file = setup_mininet_and_transmit(video)
    pcap_file = "/home/best/Desktop/EEE4022S/scripts/bold_colours_720p_30.pcap"
    delete_directories("/home/best/Desktop/EEE4022S/scripts")
    # subprocess.run(["/home/best/miniconda3/bin/python", "packet_counter.py", pcap_file])
    numberofpackets, average_interval = read_output_file('output_packet_count.txt')
    
    thread1 = threading.Thread(target=extract_features, args=(pcap_file,video,interval, numberofpackets, average_interval))
    thread2 = threading.Thread(target=play_video)
    # Start threads
    thread1.start()
    thread2.start()
    # Wait for both threads to complete
    thread1.join()
    thread2.join()
    #os.system("python3 -m itu_p1203 output.json")
    
    # update_and_show_plot()
    # plt.show()