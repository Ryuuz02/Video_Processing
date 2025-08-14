# Import statements
import os
import json
import torch
from ultralytics import YOLO
from tqdm import tqdm

# Primary detection function
def detect_objects_gpu(frames_dir, output_dir, model_name="yolov8n.pt", batch_size=16):
    """
    Run YOLOv8 detection on frames using GPU in batches.

    Args:
        frames_dir (str): Directory with frames.
        output_dir (str): Directory to save outputs.
        model_name (str): YOLOv8 model name.
        batch_size (int): Number of frames to process at once.

    Returns:
        str: Path to JSON metadata file.
    """
    # Initialize model
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda"
    model = YOLO(model_name)

    # Finds all the frame files
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))
    )
    
    metadata = {}

    # Process the frames in batches based on batch_size
    for i in tqdm(range(0, len(frame_files), batch_size), desc="GPU detection batches"):
        batch_files = frame_files[i : i + batch_size]
        batch_paths = [os.path.join(frames_dir, f) for f in batch_files]

        # Start GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # Batch inference
        results = model(batch_paths, device=device, verbose=False)

        end_event.record()
        # Use sychronize to make sure it doesn't break
        torch.cuda.synchronize() 

        # Parse detections per frame
        for frame_file, res in zip(batch_files, results):
            detections = []
            for box in res.boxes:
                # Puts the detection class and confidence in detections list, which will then go to metadata
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                detections.append({"class": cls_name, "confidence": conf})
            metadata[frame_file] = detections

    # Save metadata to JSON
    metadata_path = os.path.join(output_dir, "detections_gpu.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path

