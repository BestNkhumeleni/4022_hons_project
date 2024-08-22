from mininet.net import Mininet
from mininet.node import OVSController
from mininet.log import setLogLevel, info
from mininet.util import dumpNodeConnections
import os
import time
import shutil

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
    time.sleep(30)
    
    # Stop the tcpdump capture and video transmission
    info("*** Stopping video transmission and capture after 30 seconds ***\n")
    h1.cmd('pkill -f tcpdump')
    #h1.cmd('pkill -f nc')
    #h2.cmd('pkill -f nc')
    
    # Move the capture file to the folder
    destination_path = "/home/best/Desktop/EEE4022S/Data/"+folder_name+"_packets"
    shutil.move(capture_file, destination_path)
    info(f"*** Moved capture file to {destination_path} ***\n")
    
    # Stop the network
    net.stop()
    info("*** Mininet stopped ***\n")

if __name__ == '__main__':
    setLogLevel('info')
    
    import os

    # Specify the directory path
    directory = '/home/best/Desktop/EEE4022S/Data/Raw_Videos'

    # Loop through all the files in the directory
    for filename in os.listdir(directory):
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(directory, filename)):
            video_file = directory +"/" + filename
            setup_mininet_and_transmit(video_file)
    os.system("rmdir *p")

    
    
    
