from mininet.net import Mininet
from mininet.node import OVSController
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import os
import time
import shutil

def check_file_size(file_path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)
    
    # Convert size to megabytes (1 MB = 1024 * 1024 bytes)
    size_in_mb = file_size / (1024 * 1024)
    
    # Check if the file size is less than 1 MB
    return size_in_mb
        
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
    
    # Start a simple video transmission using netcat
    info("*** Starting video transmission from h1 to h2 ***\n")
    
    # Start a server on h2 to receive the video
    h2.cmd('nc -l -p 12345 > received_video.mp4 &')
    
    # Send the video from h1 to h2
    h1.cmd(f'cat {video_file} | nc 10.0.0.2 12345 &')
    
    # Wait for 30 seconds
    time.sleep(10)
    
    # Stop the tcpdump capture and video transmission
    info("*** Stopping video transmission and capture after 30 seconds ***\n")
    h1.cmd('pkill -f tcpdump')
    #h1.cmd('pkill -f nc')
    #h2.cmd('pkill -f nc')
    
    
    # Move the capture file to the folder
    destination_path = "/home/best/Desktop/EEE4022S/Data/pcap_files"
    shutil.move(capture_file, destination_path)
    info(f"*** Moved capture file to {destination_path} ***\n")
    
    # Stop the network
    net.stop()
    info("*** Mininet stopped ***\n")

if __name__ == '__main__':
    setLogLevel('info')
    
    # Specify the directory path
    directory = '/home/best/Desktop/EEE4022S/Data/Raw_Videos'
    destination_path = "/home/best/Desktop/EEE4022S/Data/pcap_files"
    files = [f for f in os.listdir(destination_path) if os.path.isfile(os.path.join(destination_path, f))]
    #print(files)
    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        video_file = directory +"/" + filename
        # Check if it's a file (not a directory)
        temp = filename[:-4] + ".pcap"
        
        if temp in files: #check if video has already been streamed succesfully
            file = os.path.join(destination_path, temp)
            # print(f"{file}")
            if check_file_size(os.path.join(destination_path, temp)) > 1:
                continue
            else:
                os.system(f"rm {file}")
                
        elif os.path.isfile(os.path.join(directory, filename)):
            setup_mininet_and_transmit(video_file)
            #os.system("sudo mn -c")
    
    
    
    for filename in os.listdir(destination_path):
        # Check if it's a file (not a directory)
        file = os.path.join(destination_path, filename)
        if os.path.isfile(file):
            while check_file_size(file)<1: #if it failed, keep trying until it doesnt
                print()
                print("There was a problem during capture for "+filename)
                video_name = filename[:-5]
                video_file = directory +"/" + video_name + ".mp4"
                print("Recapturing")
                os.system(f"rm {file}")
                setup_mininet_and_transmit(video_file)
                
#os.system("/home/best/miniconda3/bin/python /home/best/Desktop/EEE4022S/scripts/Feature_extractor.py")
os.system("rmdir *p")
os.system("rmdir *30")
    
    
