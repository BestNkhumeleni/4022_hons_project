import cv2
import subprocess
import json
import os

def get_video_info(video_path):
    # Get video resolution and framerate using OpenCV
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise ValueError(f"Error opening video file {video_path}")
    
    # Resolution
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Frame rate
    fps = video.get(cv2.CAP_PROP_FPS)
    
    video.release()
    
    return width, height, fps

def get_ffprobe_metadata(video_path):
    # Use ffprobe to get more detailed metadata, suppressing errors
    command = [
        'ffprobe',
        '-v', 'error', # Suppresses all errors
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'json',
        video_path
    ]
    
    # Redirect stderr to /dev/null to suppress any error output
    with open(os.devnull, 'w') as devnull:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=devnull)
    
    return json.loads(result.stdout)

if __name__ == "__main__":
    #video_path = "/home/best/Desktop/EEE4022S/Data/Raw_Videos/bold_colours_360p.mp4"
    
    # Get video info using OpenCV
    # width, height, fps = get_video_info(video_path)
    
    # print(f"Resolution: {width}x{height}")
    # print(f"Frame rate: {fps} fps")
    
    # # Optionally get detailed metadata using ffprobe
    # metadata = get_ffprobe_metadata(video_path)
    # print("FFprobe Metadata:", json.dumps(metadata, indent=2))


    video_path = "/home/best/Desktop/EEE4022S/Data/Raw_Videos"
    for filename in os.listdir(video_path):
    # Get video info using OpenCV
        video = video_path + "/" + filename
        width, height, fps = get_video_info(video)
        print(video)
        print(f"Resolution: {height}p")
        print(f"Frame rate: {fps} fps")



    # # Optionally get detailed metadata using ffprobe
    # metadata = get_ffprobe_metadata(video_path)
    # print("FFprobe Metadata:", json.dumps(metadata, indent=2))
