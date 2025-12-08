import os
import cv2

# Set input video and output folder
video_path = "../data/raw/test.mp4"
output_dir = "../data/frames/test"

os.makedirs(output_dir, exist_ok=True)

# Extract 5 frames per second
fps = 5
vidcap = cv2.VideoCapture(video_path)
frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
interval = int(frame_rate / fps)

count = 0
success = True
frame_id = 0

while success:
    success, frame = vidcap.read()
    if not success:
        break
    if int(vidcap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
        out_name = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(out_name, frame)
        frame_id += 1
        print(f"Saved {out_name}")
vidcap.release()

print(f"\nDone! Extracted {frame_id} frames.")
