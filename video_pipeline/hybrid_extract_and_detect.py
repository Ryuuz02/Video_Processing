# import statements
import os
import cv2
import json
import threading
import queue
import torch
from ultralytics import YOLO
from tqdm import tqdm
import copy

# Extraction function to read video and push frames to queue
def hybrid_extract(video_path, frame_queue, frame_skip=1):
    """Extract frames and push to queue."""
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Breaks when the video ends
            break
        if frame_idx % frame_skip == 0:
            frame_queue.put((count, frame))
            count += 1
        frame_idx += 1

    cap.release()
    frame_queue.put(None) 
    print(f"[Hybrid] Extraction done. {count} frames queued.")


# Detection function to read frames from queue, run detection, and save results
def hybrid_detect(frame_queue, output_dir, model_name="yolov8n.pt", batch_size=16, save=False, conf_threshold=0.5):
    """Consume frames from queue, run detection, and save annotated frames and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model and variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_name)

    metadata = {}
    batch_frames = []
    batch_ids = []

    pbar = tqdm(desc="Hybrid GPU detection", unit="frame")

    while True:
        # Gets next frame in queue
        item = frame_queue.get()
        if item is None:  
            # Process last partial batch and break
            if batch_frames:
                save_batch_annotations(model, batch_frames, batch_ids, output_dir, metadata, device, conf_threshold)
            break

        fid, frame = item
        batch_frames.append(frame)
        batch_ids.append(fid)

        if len(batch_frames) >= batch_size:
            # Process and save the current batch
            save_batch_annotations(model, batch_frames, batch_ids, output_dir, metadata, device, conf_threshold)
            batch_frames.clear()
            batch_ids.clear()

        pbar.update(1)

    pbar.close()

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "detections_hybrid.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Hybrid] Detection complete. Metadata saved to {metadata_path}")

# Function to filter results by confidence threshold
def filter_results(res, conf_thresh):
    """Filter YOLO results by confidence threshold."""
    filtered_res = copy.deepcopy(res)
    mask = res.boxes.conf >= conf_thresh
    filtered_res.boxes = res.boxes[mask]
    return filtered_res

# Function to save batch annotations and update metadata
def save_batch_annotations(model, frames, ids, output_dir, metadata, device, conf_threshold):
    results = model(frames, device=device, verbose=False)

    for fid, res in zip(ids, results):
        # filters each result by confidence threshold
        filtered_res = filter_results(res, conf_threshold)

        # Save metadata from filtered boxes to detections and then metadata
        detections = []
        for box in filtered_res.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": [float(x) for x in box.xyxy[0]]
            })
        metadata[f"frame_{fid:06d}.jpg"] = detections

        # redraws with ultralytics plot function to keep class coloration
        annotated_img = filtered_res.plot()
        out_path = os.path.join(output_dir, f"frame_{fid:06d}.jpg")
        cv2.imwrite(out_path, annotated_img)


# Function to parse detections into serializable format
def parse_detections(model, result):
    """Convert YOLO result into serializable dict."""
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        detections.append({"class": cls_name, "confidence": conf})
    return detections


# Main function to run the hybrid pipeline
def run_hybrid(video_path, output_dir, model, frame_skip=1, batch_size=16, conf_threshold=0.5):
    """Run hybrid CPU–GPU pipeline."""
    # Adjust max_size as needed to avoid memory problems
    frame_queue = queue.Queue(maxsize=64)  

    # Create producer and consumer threads
    producer = threading.Thread(target=hybrid_extract, args=(video_path, frame_queue, frame_skip))
    consumer = threading.Thread(target=hybrid_detect, args=(frame_queue, output_dir, model, batch_size, conf_threshold))

    # Start both threads
    producer.start()
    consumer.start()

    # Wait for both threads to finish
    producer.join()
    consumer.join()

# Alternate function to run hybrid detection without drawing
def run_hybrid_detect_only(video_path, model, frame_skip=1, batch_size=16):
    """Run hybrid CPU–GPU pipeline, store raw detection results without drawing."""
    results_store = {}
    frame_queue = queue.Queue(maxsize=64)

    def consumer_detect_only(frame_queue, model, batch_size, results_store):
        batch_frames = []
        batch_ids = []
        while True:
            item = frame_queue.get()
            if item is None:
                # If no more frames, break
                break
            fid, frame = item
            batch_frames.append(frame)
            batch_ids.append(fid)
            if len(batch_frames) >= batch_size:
                # Load the model and run detection on the batch
                batch_results = model(batch_frames, device="cuda", verbose=False)
                for id_, res in zip(batch_ids, batch_results):
                    results_store[id_] = res 
                batch_frames.clear()
                batch_ids.clear()
        # Flush last partial batch
        if batch_frames:
            batch_results = model(batch_frames, device="cuda", verbose=False)
            for id_, res in zip(batch_ids, batch_results):
                results_store[id_] = res

    # Create, start, and synchronize producer and consumer threads
    producer = threading.Thread(target=hybrid_extract, args=(video_path, frame_queue, frame_skip))
    consumer = threading.Thread(target=consumer_detect_only, args=(frame_queue, model, batch_size, results_store))

    producer.start()
    consumer.start()

    producer.join()
    consumer.join()

    return results_store

# Function to redraw frames with a confidence threshold
def redraw_with_threshold(results_store, output_dir, model, conf_threshold):
    import copy
    os.makedirs(output_dir, exist_ok=True)
    for fid, res in results_store.items():
        # Create a deepcopy, use it to make a mask, then plot using the mask filter
        filtered_res = copy.deepcopy(res)
        mask = res.boxes.conf >= conf_threshold
        filtered_res.boxes = res.boxes[mask]
        annotated = filtered_res.plot()
        cv2.imwrite(os.path.join(output_dir, f"frame_{fid:06d}.jpg"), annotated)
