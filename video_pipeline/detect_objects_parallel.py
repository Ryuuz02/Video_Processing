# import statements
import os
import json
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
from tqdm import tqdm


# Frame processing for each individidual worker/frame
def _process_frame(args):
    """Worker function for a single frame."""
    # Model loaded inside each worker to avoid issues with multiprocessing
    frame_path, model_name = args
    model = YOLO(model_name)
    model.to("cpu")

    results = model(frame_path, device="cpu", verbose=False)[0]

    # Go through each detection and put it in a list to put in metadat
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"class": cls_name, "confidence": conf})

    return os.path.basename(frame_path), detections

# Main detection function
def detect_objects_parallel(frames_dir, output_dir, model_name="yolov8n.pt", workers=None):
    """
    Run YOLO object detection in parallel on CPU.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Determine number of workers
    reserved_cpus = int(input("Enter number of reserved CPUs (default 1): ")) or 1

    if workers is None:
        workers = max(1, cpu_count() - reserved_cpus)

    # Imports the frame files
    frame_files = sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    )

    # Splits the work across multiple processes
    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap(_process_frame, [(fp, model_name) for fp in frame_files]),
                total=len(frame_files),
                desc="Detecting objects (parallel CPU)",
            )
        )

    # Merge results into metadata
    metadata = {frame_file: dets for frame_file, dets in results}

    # Writes metadata to JSON
    metadata_path = os.path.join(output_dir, "detections_parallel.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path
