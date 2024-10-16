from mininet.net import Mininet
from mininet.node import OVSController
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import os
import time
import shutil
import re

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

def check_file_size(file_path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)
    
    # Convert size to megabytes (1 MB = 1024 * 1024 bytes)
    size_in_mb = file_size / (1024 * 1024)
    
    # Check if the file size is less than 1 MB
    return size_in_mb
        
def setup_mininet_and_transmit(video_file, stream_number):
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
    
    # Start capturing packets on the link between h1 and s1
    capture_file = f"{folder_name}_{stream_number}.pcap"
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
    
    
    # Move the capture file to the folder
    destination_path = "/home/best/Desktop/EEE4022S/Data/pcap_files"
    file_path = os.path.join(destination_path, capture_file)
    if os.path.isfile(file_path):
        if check_file_size(file_path) < 1:
            print(f"File {file_path} is less than 1MB, deleting and recapturing...")
            os.remove(file_path)
    else:
        shutil.move(capture_file, destination_path)
        info(f"*** Moved capture file to {destination_path} ***\n")
    
    # Stop the network
    net.stop()
    info("*** Mininet stopped ***\n")


def check_and_recapture(destination_path, video_file, stream_number):
    capture_file = f"{os.path.basename(video_file)[:-4]}_{stream_number}.pcap"
    file_path = os.path.join(destination_path, capture_file)

    if os.path.isfile(file_path):
        while check_file_size(file_path) < 1:
            print(f"File {file_path} is less than 1MB, deleting and recapturing...")
            os.remove(file_path)
            setup_mininet_and_transmit(video_file, stream_number)
        else:
            print(f"File {file_path} already exists and is over 1MB, skipping capture.")
    else:
        setup_mininet_and_transmit(video_file, stream_number)


if __name__ == '__main__':
    delete_directories("/home/best/Desktop/EEE4022S/scripts")
    # Loop through all files in the directory
    directory = "/home/best/Desktop/EEE4022S/Data/Raw_Videos"

    for filename in os.listdir(directory):
        # Check if the file is a video file (e.g., with common extensions)
        if filename.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            # Replace spaces with underscores
            new_filename = filename.replace(' ', '_')
            
            # Remove any brackets (both round () and square [])
            new_filename = re.sub(r'[\(\)\[\]]', '', new_filename)
            
            # Get the full file path for renaming
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            
            # Rename the file
            os.rename(old_file, new_file)
            #print(f'Renamed: {old_file} -> {new_file}')

    print("Renaming completed!")
    print()

    setLogLevel('info')
    # Specify the directory path
    directory = '/home/best/Desktop/EEE4022S/Data/Raw_Videos'
    destination_path = "/home/best/Desktop/EEE4022S/Data/pcap_files"
    #files = [f for f in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, f))]
    #print(files)
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        video_file = os.path.join(directory, filename)
        for stream_number in range(1, 5):
            check_and_recapture(destination_path, video_file, stream_number)

    for filename in os.listdir(destination_path):
        file = os.path.join(destination_path, filename)
        if os.path.isfile(file) and check_file_size(file) < 1:
            video_name = filename[:-5]
            video_file = os.path.join(directory, video_name[:-2] + ".mp4")
            stream_number = int(video_name[-1:])
            print("Recapturing due to failure...")
            os.remove(file)
            setup_mininet_and_transmit(video_file,stream_number)
            check_and_recapture(destination_path, video_file, stream_number)
    
    

delete_directories("/home/best/Desktop/EEE4022S/scripts")
                
os.system("/home/best/miniconda3/bin/python /home/best/Desktop/EEE4022S/scripts/Feature_extractor.py")

    
    
