import pyshark
import math
import os
import csv
import shutil
import concurrent.futures
from pytictoc import TicToc
import cv2
import time

t = TicToc()

#get_fps(pcap_file) #if you know

#get_resolution(pcap_file) #how many packets make up a frame in each resolution
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

def get_packet_count_pcapng(capture_file):
    cap = capture_file
    count = 0
    for _ in cap:
        count += 1
        #print(count)
    return count


def extract_features(pcap_file, video_path, interval):
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
    
    num_packets = get_packet_count_pcapng(capture)
    # t.toc()
    print("starting:")
    while(current_time<=video_length):
        stop_packets = round(num_packets*(current_time/video_length))
        t.tic()
        for packet in capture:
            try:
                start_time = time.time()
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
                    end_time = time.time()
                    # Calculate how long the iteration took
                    elapsed_time = end_time - start_time
                    # Calculate the remaining time to sleep if the iteration finished early
                    time_to_sleep = interval - elapsed_time -0.2
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    break

                # Exit early after 30 seconds
                #print(packet_time - start_time)
            
            except AttributeError:
                continue  # Skip packets that don't have the required fields
        capture.close()
        t.toc()
        time_intervals = [t2 - t1 for t1, t2 in zip(packet_times[:-1], packet_times[1:])]
        mean_interval = sum(time_intervals) / len(time_intervals)
        mean_packet_size = sum(packet_sizes) / len(packet_sizes)
        
        if mean_interval is not None:
            mean_interval = mean_interval * 1000

        if start_time and end_time:
            duration = end_time - start_time
            # print(duration)
            bit = (total_bytes * 8) / duration  # bits per second
            csvstor = {
            'name': filename,
            'bitrate': bit,
            'num_bytes': total_bytes,
            'num_packets': total_packets,
            'interval': mean_interval,
            'packet_size': mean_packet_size,
            }
            
            print(f"Analysis complete, feature extracted for {filename} are:")
            print(csvstor)
            print()
            out_csv = filename +".csv"
            append_to_csv(out_csv, csvstor)
            
        current_time+=interval
        
    duration = get_video_duration_opencv(video_path)
    bitrate = (total_bytes * 8) / duration
    return bitrate



#t.tic()
pcap = '/home/best/Desktop/EEE4022S/scripts/test_480p.pcap'
video = "/home/best/Desktop/EEE4022S/Data/Raw_Videos/test_480p.mp4"
interval = 5

extract_features(pcap,video,interval)
# print(get_video_duration_opencv("/home/best/Desktop/EEE4022S/Data/Raw_Videos/test_480p.mp4"))
#t.toc()

print()
print("All files feature extraction complete")
print()
#os.system("/home/best/miniconda3/bin/python /home/best/Desktop/EEE4022S/scripts/ML_models.py")





