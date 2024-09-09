import pyshark
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import csv
import shutil
import concurrent.futures
from multiprocessing import Pool

#get_fps(pcap_file) #if you know

#get_resolution(pcap_file) #how many packets make up a frame in each resolution

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




def extract_features(pcap_file):
    capture = pyshark.FileCapture(pcap_file)
    
    video_name = os.path.basename(pcap_file)
    filename = video_name[:-5]
    print()
    print("Analysing "+filename)
    print()
    total_bytes = 0
    start_time = None
    end_time = None
    total_packets = 0
    packet_sizes = []
    packet_times = []
    
    for packet in capture:
        total_packets+=1
        packet_time = float(packet.sniff_timestamp)
        
        if hasattr(packet, 'frame_info'):
            total_bytes += int(packet.length)
            
            packet_sizes.append(int(packet.length))
            packet_times.append(float(packet.sniff_timestamp))
            
            if start_time is None:
                start_time = packet_time
            end_time = packet_time
        tim = packet_time-start_time
        print(f"{video_name} : {tim}")
        print()
        if math.trunc(packet_time-start_time) == 30:
            break
        #print(packet.timestamp())
    # print(f"the total number of bytes is: {total_bytes}")
    # print(f"the total number of packets is: {total_packets}")
    
    
    capture.close()
    
    time_intervals = [t2 - t1 for t1, t2 in zip(packet_times[:-1], packet_times[1:])]
    mean_interval = sum(time_intervals) / len(time_intervals)
    mean_packet_size = sum(packet_sizes) / len(packet_sizes)
    
    if mean_interval is not None:
        mean_interval = mean_interval * 1000
    #     print(f"Mean interval between packets: {mean_interval:.6f} ms")
    # else:
    #     print("Unable to calculate mean interval between packets")

    # if mean_packet_size is not None:
    #     print(f"Mean packet size: {mean_packet_size:.2f} bytes")
    # else:
    #     print("Unable to calculate mean packet size")
    
    
    
    
    if start_time and end_time:
        duration = end_time - start_time
        bitrate = (total_bytes * 8) / duration  # bits per second
        csvstor = {
        'name': filename,
        'num_bytes': total_bytes,
        'num_packets': total_packets,
        'interval': mean_interval,
        'packet_size': mean_packet_size,
        'bitrate': bitrate
        }
        
        print(f"Analysis complete, feature extracted for {filename} are:")
        print(csvstor)
        
        print()

        out_csv = filename +".csv"
        append_to_csv(out_csv, csvstor)
        
        
        destination_path = "/home/best/Desktop/EEE4022S/Data/training_data/"
        shutil.move(out_csv, destination_path)
        
        print(f"Moving {out_csv} to {destination_path}")
        print()
        return bitrate
    else:
        return None

directory_path = '/home/best/Desktop/EEE4022S/Data/pcap_files'
destination_path = "/home/best/Desktop/EEE4022S/Data/training_data/"
files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
files.sort()

#print(files)
csvs = [f for f in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, f))]
csvs.sort()

# os.path.basename
undone_files = []
for file in files:
    fil = os.path.basename(file)[:-5] + ".csv" 
    if fil not in csvs:
        undone_files.append(file)

with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the get_bitrate function to the files in parallel
        if len(undone_files) != 0:
            list(executor.map(extract_features, undone_files))
        else:
            print("All files have been done")
        

print()
print("All files feature extraction complete")
print()
os.system("/home/best/miniconda3/bin/python /home/best/Desktop/EEE4022S/scripts/ML_models.py")





