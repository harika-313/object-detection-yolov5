import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from collections import defaultdict

st.set_page_config(page_title="YOLOv5 Object Detection Demo", layout="wide")
st.title("YOLOv5 Object Detection Demo")

# Load YOLOv5 Nano model
@st.cache_resource
def load_model():
    return YOLO('yolov5nu.pt')

model = load_model()

# ------------------ IMAGE DETECTION ------------------
st.header("Image Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = model(image, imgsz=320)
    annotated_img = results[0].plot()
    st.image(annotated_img[:, :, ::-1], caption='Detected Image', use_container_width=True)

    # Show detected object summary for image
    class_counts = defaultdict(int)
    for detection in results[0].boxes.data.cpu().numpy():
        class_id = int(detection[5])
        class_name = results[0].names[class_id]
        class_counts[class_name] += 1
    if class_counts:
        st.write("Detected objects:", ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in class_counts.items()]))
    else:
        st.write("No objects detected.")

# ------------------ VIDEO DETECTION ------------------
st.header("Video Detection")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    unique_objects = defaultdict(set)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320, conf=0.5)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", caption=f"Frame {frame_count}", use_container_width=True)

        # Track unique objects
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for tid, cid in zip(track_ids, class_ids):
                class_name = results[0].names[cid]
                unique_objects[class_name].add(tid)
    cap.release()

    # Show unique object summary after video processing
    st.subheader("Unique objects detected in the video:")
    if unique_objects:
        st.write(", ".join([f"{len(ids)} {cls}{'s' if len(ids) > 1 else ''}" for cls, ids in unique_objects.items()]))
    else:
        st.write("No objects detected in the video.")

# ------------------ WEBCAM PHOTO DETECTION ------------------
st.header("Webcam Photo Detection")
camera_image = st.camera_input("Take a picture")
if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    results = model(image, imgsz=320)
    annotated_img = results[0].plot()
    st.image(annotated_img[:, :, ::-1], caption='Detected Webcam Image', use_container_width=True)

    # Show detected object summary for webcam image
    class_counts = defaultdict(int)
    for detection in results[0].boxes.data.cpu().numpy():
        class_id = int(detection[5])
        class_name = results[0].names[class_id]
        class_counts[class_name] += 1
    if class_counts:
        st.write("Detected objects:", ", ".join([f"{v} {k}{'s' if v > 1 else ''}" for k, v in class_counts.items()]))
    else:
        st.write("No objects detected.")
