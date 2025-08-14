# import statements
import cv2
import os
from tqdm import tqdm

# Main frame extraction function
def extract_frames(video_path, output_dir, frame_skip=1):
    """
    Extract frames from a video.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        frame_skip (int): Save every Nth frame (default=1 = every frame).
    Returns:
        int: Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Gets total frame count for progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0
    idx = 0

    # Use tqdm for progress bar
    with tqdm(total=frame_count, desc="Extracting frames") as pbar:
        while True:
            # Will read frame by frame and if it doesn not exist already then save it
            ret, frame = cap.read()
            if not ret:
                # Breaks when the video ends
                break

            if idx % frame_skip == 0:
                frame_filename = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                if not os.path.exists(frame_filename):
                    cv2.imwrite(frame_filename, frame)
                    extracted += 1
            idx += 1
            pbar.update(1)

    cap.release()
    return extracted
