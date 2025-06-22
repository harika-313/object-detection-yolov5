# YOLOv5 Object Detection – Jupyter Notebook

## Overview

This repository contains a Jupyter Notebook that demonstrates object detection and tracking using the YOLOv5 Nano model. The notebook provides a step-by-step workflow for detecting and summarizing unique objects in images, videos, and webcam photos, with clear explanations in markdown cells[#].

- **Model:** YOLOv5 Nano (Ultralytics)
- **Libraries:** OpenCV, NumPy, Matplotlib, tqdm
- **Features:**
  - Image, video, and webcam photo detection
  - Unique object counting in videos (with tracking)
  - Annotated visual outputs and text summaries
  - Well-documented workflow with markdown explanations

## Contents

- `object_detection_notebook.ipynb` — The main Jupyter Notebook with code and explanations
- `app.py` — (Optional) Streamlit web app for user-friendly interface (run separately)
- `requirements.txt` — List of required Python packages
- Example images/videos for testing

## How to Use

1. **Clone the repository:**
git clone https://github.com/harika-313/object-detection-yolov5.git
cd object-detection-yolov5

2. **Install dependencies:**
pip install -r requirements.txt

3. **Open the notebook:**
jupyter notebook object_detection_notebook.ipynb

- Follow the notebook step by step.
- Each code cell is explained with a markdown cell above it.
- You can test on your own images/videos or use the provided samples.

4. **(Optional) Run the Streamlit app:**
- For an interactive web interface, run:
  ```
  streamlit run app.py
  ```
- This is a standalone app and not part of the notebook.

## Project Structure

├── object_detection_notebook.ipynb
├── app.py
├── requirements.txt
├── sample_images/
├── sample_videos/
└── README.md

## Key Features

- **Step-by-step workflow:** Each stage of the detection process is explained and visualized.
- **Unique object tracking:** Video detection uses tracking to avoid overcounting the same object.
- **Clear outputs:** Results include annotated images/videos and concise text summaries.
- **Separation of concerns:** The notebook focuses on code and documentation, while the Streamlit app provides a user interface.

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)[#]
- [OpenCV Documentation](https://docs.opencv.org/)[#]
- [Streamlit Documentation](https://docs.streamlit.io/)[#]

## License

This project is for educational purposes.

# YOLOv5 Object Detection Streamlit App

A user-friendly web application for object detection and unique object summarization in images, videos, and webcam photos, powered by YOLOv5 and Streamlit[#].

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Best Practices](#best-practices)
- [References](#references)

## Overview

This Streamlit app allows users to:
- Upload images or videos for object detection
- Capture webcam photos for detection
- View annotated results and a summary of unique objects detected (for videos, the summary is across the whole video, not per frame)

The app leverages the YOLOv5 Nano model for fast, accurate detection and uses ByteTrack for robust tracking in videos[#].

## Features

- **Image Detection:** Upload an image and see detected objects with bounding boxes and labels.
- **Video Detection:** Upload a video, see annotated frames, and get a summary of unique objects detected throughout the video.
- **Webcam Photo Detection:** Take a photo using your webcam and get instant detection results.
- **Intuitive UI:** Simple, clean interface built with Streamlit.
- **Efficient Processing:** Optimized for performance and usability.

## Demo

*Sample output: Detected objects in an image and summary for a video.*

## Installation

1. **Clone this repository:**
git clone https://github.com/harika-313/object-detection-yolov5.git
cd object-detection-yolov5

2. **Install dependencies:**
pip install -r requirements.txt

## Usage

1. **Run the Streamlit app:**
streamlit run app.py

2. **Open the app in your browser:**  
Streamlit will provide a local URL (e.g., http://localhost:8501).

3. **Use the interface to:**
- Upload images or videos
- Capture a webcam photo
- View annotated results and unique object summaries

## Project Structure

.
├── app.py # Main Streamlit app
├── requirements.txt # Python dependencies
├── assets/ # (Optional) Images/screenshots for README
├── README.md # This file
└── (other files as needed)

## Best Practices

- The app code is modular and focused on UI and workflow.
- Heavy logic (e.g., detection, tracking) is kept clean and separated for maintainability.
- Use `st.set_page_config(layout="wide")` for better layout.
- Use `st.cache_resource` for model loading to speed up app startup.
- For multi-page apps, consider moving pages to a `/pages` directory.
- Keep assets (images, screenshots) in an `/assets` folder for clarity.

## References

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)[#]
- [Streamlit Documentation](https://docs.streamlit.io/)[#]
- [OpenCV Documentation](https://docs.opencv.org/)[#]
- [Best Practices for Streamlit Apps](https://blog.streamlit.io/best-practices-for-building-genai-apps-with-streamlit/)[#]

## License

This app is for educational purposes.
