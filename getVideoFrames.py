import cv2
import os

video_path = '/Users/rajdipshah/ALL PROJECTS/Valentine/breakdown video/Animated_Dome_Crumbling_Heart_Rises.mp4'
output_folder = '/Users/rajdipshah/ALL PROJECTS/Valentine/frames'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", frame)
    count += 1

cap.release()
print(f"Done! Extracted {count} frames.")