import cv2
import os
from glob import glob

# Path to frames
frame_folder = "dataset"

# Get all images and sort them
images = sorted(glob(os.path.join(frame_folder, "*.jpg")))

# Read first image to get dimensions
first_frame = cv2.imread(images[0])
height, width, _ = first_frame.shape

# Video writer
fps = 30
video = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Add frames
for image_path in images:
    frame = cv2.imread(image_path)
    video.write(frame)

video.release()

print("Video saved as output.mp4")