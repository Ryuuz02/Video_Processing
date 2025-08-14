# import statements
import streamlit as st
import tempfile
import os
import shutil
import time
from pathlib import Path
from ultralytics import YOLO
import copy
import cv2
from video_pipeline.hybrid_extract_and_detect import run_hybrid_detect_only

# Use streamlit run app_copy.py

st.set_page_config(page_title="Hybrid GPU Object Detection", layout="centered")

st.title("🎥 Hybrid GPU Object Detection (YOLO12-Small)")
st.write("Upload a video and see GPU-powered object detection with hybrid CPU–GPU parallelism.")

# Confidence threshold slider
conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Only show detections with confidence above this value"
)

# Model selection
model_selection = st.selectbox(
    "Select detection Model",
    options=["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
    help="Choose the YOLO model to use for detection",
    index=1
)

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.video(input_path)

    # Load model
    if "model" not in st.session_state or st.session_state.get("model_name") != model_selection:
        st.session_state.model = YOLO(model_selection)
        st.session_state.model_name = model_selection

    # Run detection ONCE per uploaded video
    if "results_store" not in st.session_state or st.session_state.get("video_name") != uploaded_file.name:
        st.write("Running detection... please wait ⏳")
        start_time = time.time()
        st.session_state.results_store = run_hybrid_detect_only(
            video_path=input_path,
            model=st.session_state.model
        )
        st.session_state.video_name = uploaded_file.name
        total_time = time.time() - start_time
        st.success(f"✅ Detection complete in {total_time:.2f} seconds")

    # Redraw frames with the chosen confidence threshold
    output_dir = os.path.join(tmp_dir, "detections_hybrid")
    os.makedirs(output_dir, exist_ok=True)

    for fid, res in st.session_state.results_store.items():
        filtered_res = copy.deepcopy(res)
        mask = res.boxes.conf >= conf_threshold
        filtered_res.boxes = res.boxes[mask]
        annotated = filtered_res.plot()
        cv2.imwrite(os.path.join(output_dir, f"frame_{fid:06d}.jpg"), annotated)

    # Convert frames to video
    output_video_path = os.path.join(tmp_dir, "annotated.mp4")
    os.system(
        f"ffmpeg -y -framerate 30 -i {output_dir}/frame_%06d.jpg "
        f"-c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )

    st.write("### Annotated Output")
    st.video(output_video_path)

    # Download option
    with open(output_video_path, "rb") as vid_file:
        st.download_button(
            "⬇️ Download Annotated Video",
            vid_file,
            file_name="annotated.mp4",
            mime="video/mp4"
        )
