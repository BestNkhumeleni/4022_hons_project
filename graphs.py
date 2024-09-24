import matplotlib.pyplot as plt

# Function to convert resolution strings to integers (e.g., '1080p' -> 1080)
def resolution_to_int(resolution):
    return int(resolution)  # Remove the 'p' and convert to integer

# Function to read data from the text file
def read_data(filename):
    time_values = []
    qoe_values = []
    resolution_values = []
    fps_values = []
    
    with open(filename, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            data = line.strip().split()
            time_values.append(float(data[0]))
            qoe_values.append(float(data[1]))
            resolution_values.append(resolution_to_int(data[2]))  # Treat resolution as int
            fps_values.append(float(data[3]))
    
    return time_values, qoe_values, resolution_values, fps_values

# Function to compute accuracy rate over time (in percentage)
def compute_accuracy(predicted_values, correct_value):
    accuracy_rate = []
    correct_count = 0

    for i in range(len(predicted_values)):
        if predicted_values[i] == correct_value:
            correct_count += 1
        accuracy_rate.append((correct_count / (i + 1)) * 100)  # Cumulative accuracy rate in percentage

    return accuracy_rate, accuracy_rate[-1], predicted_values[-1] == correct_value

# Function to create subplots
def create_graphs(time_values, qoe_values, resolution_values, fps_values, correct_resolution, correct_fps):
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))  # Create 3x2 grid for subplots

    # Plot time vs QoE
    axs[0, 0].plot(time_values, qoe_values, 'b-')
    axs[0, 0].set_title('Time vs QoE')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('QoE')

    # Plot time vs Resolution (as integer)
    axs[0, 1].plot(time_values, resolution_values, 'g-')
    axs[0, 1].set_title('Time vs Resolution')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Resolution (p)')
    axs[0, 1].set_yticks([360, 480, 720, 1080])  # Set the tick locations
    axs[0, 1].set_yticklabels(['360', '480', '720', '1080'], fontweight='bold')  # Bold the labels

    # Plot time vs FPS
    axs[1, 0].plot(time_values, fps_values, 'r-')
    axs[1, 0].set_title('Time vs FPS')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('FPS')

    # Compute accuracy for both Resolution and FPS
    resolution_accuracy, final_res_accuracy, final_res_match = compute_accuracy(resolution_values, correct_resolution)
    fps_accuracy, final_fps_accuracy, final_fps_match = compute_accuracy(fps_values, correct_fps)

    # Plot accuracy rates for Resolution and FPS
    axs[1, 1].plot(time_values, resolution_accuracy, 'g-', label='Resolution Accuracy (%)')
    axs[1, 1].plot(time_values, fps_accuracy, 'r-', label='FPS Accuracy (%)')
    axs[1, 1].set_title('Accuracy Rate for Resolution and FPS')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Accuracy Rate (%)')
    axs[1, 1].legend()

    # Hide the unused subplot (bottom-right corner)
    axs[2, 1].axis('off')
    axs[2, 0].axis('off')

    # Adjust layout for better spacing
    

    # Print the final accuracy values and whether the last prediction matches the correct value
    print(f"Final Resolution Accuracy: {final_res_accuracy:.2f}%")
    print(f"Final FPS Accuracy: {final_fps_accuracy:.2f}%")
    print(f"Final Resolution Matches Correct Value: {final_res_match}")
    print(f"Final FPS Matches Correct Value: {final_fps_match}")
    print()
    
    print()
    plt.tight_layout()
    plt.show()

# Main function to run the program
if __name__ == "__main__":
    filename = 'graphs.txt'  # Assuming the file is named graphs.txt
    time_values, qoe_values, resolution_values, fps_values = read_data(filename)

    # Replace with the correct single resolution and fps values
    correct_resolution = 720  # Example: correct resolution
    correct_fps = 30  # Example: correct FPS

    create_graphs(time_values, qoe_values, resolution_values, fps_values, correct_resolution, correct_fps)
