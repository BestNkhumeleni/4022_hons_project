import pyshark
import sys

def count_packets_and_calculate_interval(pcap_file, output_txt_file):
    # Load the pcap file
    cap = pyshark.FileCapture(pcap_file)
    
    # Initialize variables
    packet_count = 0
    total_interval = 0
    previous_timestamp = None
    packet_times = []

    print("Counting packets and calculating intervals")
    
    # Iterate through packets in the pcap file
    for packet in cap:
        packet_count += 1
        
        # Get the packet's timestamp
        current_timestamp = float(packet.sniff_time.timestamp())
        packet_times.append(current_timestamp)

        # If there's a previous timestamp, calculate the interval
        if previous_timestamp is not None:
            interval = current_timestamp - previous_timestamp
            total_interval += interval

        # Update the previous timestamp
        previous_timestamp = current_timestamp
        
        # Progress logging for large pcap files
        if packet_count % 1000 == 0:
            print(f"Processed {packet_count} packets")
            # print(packet)

    # Close the capture file
    cap.close()
    time_intervals = [t2 - t1 for t1, t2 in zip(packet_times[:-1], packet_times[1:])]
    mean_interval = sum(time_intervals) / len(time_intervals)
    # Calculate average interval if there were at least two packets
    # average_interval = total_interval / (packet_count - 1) if packet_count > 1 else 0
    average_interval = mean_interval
    average_interval *= 1000
    # Write the results to the text file
    
    
    
    with open(output_txt_file, 'w') as f:
        f.write(f"Packet Count: {packet_count}\n")
        f.write(f"Average Interval: {average_interval} seconds\n")

    print(f"Packet count ({packet_count}) and average interval ({average_interval} seconds) written to {output_txt_file}")

if __name__ == "__main__":
    # Ensure a pcap file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python program1.py <pcap_file>")
        sys.exit(1)

    # Read the pcap file path from the command-line argument
    pcap_file = sys.argv[1]
    output_txt_file = 'output_packet_count.txt'
    
    # Count packets, calculate interval, and write to text file
    count_packets_and_calculate_interval(pcap_file, output_txt_file)
