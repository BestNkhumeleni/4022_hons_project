import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import json
from collections import deque
import threading

# Initialize data storage
qoe_values = deque(maxlen=100)  # Store last 100 QoE values
resolutions = deque(maxlen=100)  # Store last 100 resolutions
fps_values = deque(maxlen=100)   # Store last 100 FPS values

def update_data(line):
    # Extract QoE, resolution, and FPS values from the output line
    qoe_match = re.search(r'"O46": ([\d.]+)', line)
    resolution_match = re.search(r'Best Model predicts Resolution: p(\w+)', line)
    fps_match = re.search(r'Best Model predicts FPS: (\d+)', line)

    if qoe_match:
        qoe_values.append(float(qoe_match.group(1)))
    
    if resolution_match:
        resolutions.append(resolution_match.group(1))  # Remove 'p' from the prediction
    
    if fps_match:
        fps_values.append(int(fps_match.group(1)))

def run_script():
    # Run the overall_script.py and capture the output
    process = subprocess.Popen(['python3', 'overall_script.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line.strip())  # Print to console
        update_data(line)

    process.stdout.close()
    process.wait()

# Set up real-time plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

def animate(i):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    # Plot QoE
    ax1.plot(qoe_values, label='QoE (O46)', color='blue')
    ax1.set_title('Quality of Experience (QoE)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('QoE Value')
    ax1.legend()
    
    # Plot predicted resolution
    ax2.plot(resolutions, label='Predicted Resolution', color='orange')
    ax2.set_title('Predicted Resolution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Resolution')
    ax2.legend()
    
    # Plot predicted FPS
    ax3.plot(fps_values, label='Predicted FPS', color='green')
    ax3.set_title('Predicted Frame Rate (FPS)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('FPS Value')
    ax3.legend()

# Run the script and animate
if __name__ == "__main__":
    # Start the script in a separate thread
    script_thread = threading.Thread(target=run_script)
    script_thread.start()

    # Set up animation
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.tight_layout()
    plt.show()
