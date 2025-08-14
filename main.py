# Import statements
import argparse
from video_pipeline.extract_frames import extract_frames
from video_pipeline.detect_objects import detect_objects_in_frames
from video_pipeline.detect_objects_parallel import detect_objects_parallel
from video_pipeline.detect_objects_gpu import detect_objects_gpu
from video_pipeline.profiling import benchmark
from video_pipeline.hybrid_extract_and_detect import run_hybrid

# Define each function with benchmarking
@benchmark
def run_extraction(video_path, frames_dir):
    return extract_frames(video_path, frames_dir, frame_skip=1)

@benchmark
def run_detection_cpu(frames_dir, detections_dir):
    return detect_objects_in_frames(frames_dir, detections_dir, device="cpu")

@benchmark
def run_detection_parallel(frames_dir, detections_dir):
    return detect_objects_parallel(frames_dir, detections_dir)

@benchmark
def run_detection_gpu(frames_dir, detections_dir):
    return detect_objects_gpu(frames_dir, detections_dir, batch_size=64)

@benchmark
def run_hybrid_pipeline(video_path, output_dir):
    return run_hybrid(video_path, output_dir, frame_skip=1, batch_size=64, save=True)

if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cpu", "parallel", "gpu", "hybrid"], required=True)
    args = parser.parse_args()

    video_name = "money game"
    # !!! Uncomment the following line to allow user input for video name !!!
    # video_name = input("Enter the video name (without extension): ").lower()
    video_name = video_name.replace(" ", "_") 
    video_path = "videos/" + video_name + ".mp4"
    frame_dir = "output/frames/" + video_name
    detect_dir = f"output/detections_{args.mode}/" + video_name

    # use the corresponding function based on the mode
    if args.mode == "cpu":
        count, _ = run_extraction(video_path, frame_dir)
        run_detection_cpu(frame_dir, detect_dir)

    elif args.mode == "parallel":
        count, _ = run_extraction(video_path, frame_dir)
        run_detection_parallel(frame_dir, detect_dir)

    elif args.mode == "gpu":
        count, _ = run_extraction(video_path, frame_dir)
        run_detection_gpu(frame_dir, detect_dir)
    
    elif args.mode == "hybrid":
        run_hybrid_pipeline(video_path, detect_dir)
