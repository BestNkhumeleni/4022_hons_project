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




def get_bitrate(pcap_file):
    capture = pyshark.FileCapture(pcap_file)
    
    video_name = os.path.basename(pcap_file)
    filename = video_name[:-5]
    print("We are analysing "+filename)
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
        
        out_csv = filename +".csv"
        append_to_csv(out_csv, csvstor)
        
        destination_path = "/home/best/Desktop/EEE4022S/Data/training_data/"
        shutil.move(out_csv, destination_path)
        return bitrate
    else:
        return None

directory_path = '/home/best/Desktop/EEE4022S/Data'
files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
files.sort()

index = len(files)

with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the get_bitrate function to the files in parallel
        list(executor.map(get_bitrate, files))



#os.system("sudo python3 mininet_setup.py")
    # Loop through all the files in the directory
# for filename in files:
#     # Check if it's a file (not a directory)
#     if os.path.isfile(os.path.join(directory, filename)):
#         print()
#         print("We are analysing "+filename)
#         pcap_file = directory +"/" + filename
#         bitrate = get_bitrate(pcap_file, filename)
#         #fps = get_fps(pcap_file)
#         if bitrate:
#             print(f"Bitrate: {bitrate} bits per second")
#             print(f"which is {bitrate/8} bytes per second")
#         else:
#             print("Unable to calculate bitrate")
    







