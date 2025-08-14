# import statements
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

# Main CPU detection function
def detect_objects_in_frames(frames_dir, output_dir, model_name="yolov8n.pt", device="cpu"):
    """
    Run YOLO object detection on all frames in a directory.

    Args:
        frames_dir (str): Directory containing extracted frames.
        output_dir (str): Directory to save annotated frames and metadata.
        model_name (str): YOLOv8 model to use (default 'yolov8n.pt').
        device (str): 'cpu' or 'cuda'.
    Returns:
        str: Path to metadata JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = YOLO(model_name)
    model.to(device)

    metadata = {}

    # Sort frames for sequential processing
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png")))

    for frame_file in tqdm(frame_files, desc="Detecting objects"):
        frame_path = os.path.join(frames_dir, frame_file)       
        results = model(frame_path, device=device, verbose=False)[0]

        # Save annotated image
        annotated_path = os.path.join(output_dir, frame_file)
        results.save(filename=annotated_path)

        # Extract detections for class/confidence pair
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({"class": cls_name, "confidence": conf})

        # Store in metadata
        metadata[frame_file] = detections

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "detections.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

