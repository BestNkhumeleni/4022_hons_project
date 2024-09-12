import pyshark
import sys

def count_packets_in_pcap(pcap_file, output_txt_file):
    # Load the pcap file
    cap = pyshark.FileCapture(pcap_file)
    
    # Count the number of packets
    packet_count = 0
    print("Counting packets")
    for packet in cap:
        packet_count += 1
        if packet_count%1000 == 0:
            print(packet_count)

    # Close the capture file
    cap.close()

    # Overwrite the text file with just the number of packets
    with open(output_txt_file, 'w') as f:
        f.write(f"{packet_count}")

    print(f"Packet count ({packet_count}) written to {output_txt_file}")

if __name__ == "__main__":
    # Ensure a pcap file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python program1.py <pcap_file>")
        sys.exit(1)

    # Read the pcap file path from the command-line argument
    pcap_file = sys.argv[1]
    output_txt_file = 'output_packet_count.txt'
    
    # Count packets and write to text file
    count_packets_in_pcap(pcap_file, output_txt_file)
