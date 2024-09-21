import pyshark
import os
import time
import csv
import cv2

# Function to get video duration using OpenCV
def get_video_duration_opencv(video_dir):
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file not found or could not be opened: {video_dir}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    return duration

# Helper function to append results to a CSV file
def append_to_csv(filename, data):
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

def extract_feature_mk2(pcap_file, video_path, interval, num_packets):
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
    
    # Get the total video duration
    video_length = get_video_duration_opencv(video_path)

    print("Starting analysis:")
    while current_time <= video_length:
        # Calculate the number of packets to process in this interval
        stop_packets = round(num_packets * (current_time / video_length))
        inv = video_length / current_time
        
        for packet in capture:
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
                    elapsed_time = end - start
                    time_to_sleep = interval - elapsed_time - 0.2
                    if time_to_sleep > 0:
                        time.sleep(time_to_sleep)
                    break
            
            except AttributeError:
                continue  # Skip packets that don't have the required fields
        capture.close()

        # Calculate statistics
        time_intervals = [t2 - t1 for t1, t2 in zip(packet_times[:-1], packet_times[1:])]
        mean_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0
        mean_packet_size = sum(packet_sizes) / len(packet_sizes) if packet_sizes else 0
        
        if mean_interval is not None:
            mean_interval *= 1000  # Convert to milliseconds
            
        total_packets *= inv
        total_bytes *= inv
        
        if start_time and end_time:
            duration = end_time - start_time
            if duration == 0:
                duration = 0.1
            
            bit = (total_bytes * 8) / duration  # bits per second
            
            csvstor = {
                'name': filename,
                'bitrate': bit,
                'num_bytes': total_bytes,
                'num_packets': total_packets,
                'interval_ms': mean_interval,
                'mean_packet_size': mean_packet_size
            }
            
            print(f"Analysis complete, feature extracted for {filename} are:")
            print(csvstor)
            print()
            
            out_csv = filename + ".csv"
            os.system(f"rm {out_csv}")
            append_to_csv(out_csv, csvstor)

            # Further analysis could be done here (e.g., predict resolution and fps)
            # For simplicity, I'll skip the JSON generation and other steps.

        # Move to the next interval
        current_time += interval

# Example usage:
pcap_file = '/home/best/Desktop/EEE4022S/scripts/test_480p.pcap'
video_path = '/home/best/Desktop/EEE4022S/Data/Raw_Videos/test_480p.mp4'
interval = 10  # 10 seconds interval in real-time
num_packets = 5000  # Total number of packets in the PCAP

extract_feature_mk2(pcap_file, video_path, interval, num_packets)
