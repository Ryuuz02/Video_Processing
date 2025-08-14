# import statments
import streamlit as st
import tempfile
import os
import shutil
import time
from pathlib import Path
from video_pipeline.hybrid_extract_and_detect import run_hybrid

# use streamlit run app.py

# Confidence threshold slider
conf_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Only show detections with confidence above this value"
)
model_selection = st.selectbox(
    "Select detection Model",
    options=["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
    help="Choose the YOLO model to use for detection",
    index=1
)

st.set_page_config(page_title="Hybrid GPU Object Detection", layout="centered")

st.title("🎥 Hybrid GPU Object Detection (YOLO12-Small)")
st.write("Upload a video and see GPU-powered object detection with hybrid CPU–GPU parallelism.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    # Save uploaded file to a temp directory
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    st.video(input_path)

    # Run detection
    st.write("Running detection... please wait ⏳")
    start_time = time.time()

    output_dir = os.path.join(tmp_dir, "detections_hybrid")
    os.makedirs(output_dir, exist_ok=True)

    run_hybrid(video_path=input_path, output_dir=output_dir, model=model_selection, frame_skip=1, batch_size=64, conf_threshold=conf_threshold)

    total_time = time.time() - start_time
    st.success(f"✅ Processing complete in {total_time:.2f} seconds")

    # Put the annotated frames into a video
    output_video_path = os.path.join(tmp_dir, "annotated.mp4")
    os.system(
        f"ffmpeg -y -framerate 30 -i {output_dir}/frame_%06d.jpg "
        f"-c:v libx264 -pix_fmt yuv420p {output_video_path}"
    )

    st.write("### Annotated Output")
    st.video(output_video_path)

    # Download button for the annotated video
    with open(output_video_path, "rb") as vid_file:
        st.download_button(
            "⬇️ Download Annotated Video",
            vid_file,
            file_name="annotated.mp4",
            mime="video/mp4"
        )
