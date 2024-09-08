import os
import re

# Specify the directory containing the video files
directory = '/home/best/Desktop/EEE4022S/Data/Raw_Videos'

# Loop through all files in the directory
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
        print(f'Renamed: {old_file} -> {new_file}')

print("Renaming completed!")
