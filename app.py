import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import json

# --- CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Annotation Tool",
    page_icon="🎯",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8-seg model, caches for performance."""
    try:
        model = YOLO('yolov8n-seg.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model()
if model:
    CLASS_NAMES = list(model.names.values())
else:
    CLASS_NAMES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'] # Fallback

# --- SESSION STATE INITIALIZATION ---
# This is Streamlit's way of keeping track of variables across interactions
if 'annotations' not in st.session_state:
    st.session_state.annotations = {} # Will store annotations per image
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_0" # Unique key for the canvas

# --- HELPER FUNCTIONS ---
def run_yolo_on_image(image_as_pil, confidence):
    """Performs YOLO prediction and returns results in a Streamlit-friendly format."""
    if not model:
        st.warning("YOLO model not loaded. Cannot perform auto-detection.")
        return []

    # Convert PIL Image to NumPy array for YOLO
    img_array = np.array(image_as_pil)
    results = model.predict(source=img_array, conf=confidence, verbose=False)

    # Convert YOLO results to the format expected by streamlit-drawable-canvas
    canvas_objects = []
    for result in results:
        # Process Boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            canvas_objects.append({
                "type": "rect",
                "left": x1, "top": y1,
                "width": x2 - x1, "height": y2 - y1,
                "fill": "#FF6B6B33", "stroke": "#FF6B6B", "strokeWidth": 2,
                "label": f"{model.names[int(box.cls[0])]} ({box.conf[0]:.2f})"
            })
        # Process Polygons
        if result.masks:
            for i, mask in enumerate(result.masks):
                polygon_points = mask.xy[0].tolist()
                label = f"{model.names[int(result.boxes[i].cls[0])]} ({result.boxes[i].conf[0]:.2f})"
                canvas_objects.append({
                    "type": "path",
                    "path": [["M"] + polygon_points[0]] + [["L"] + p for p in polygon_points[1:]] + [["Z"]],
                    "fill": "#4ECDC433", "stroke": "#4ECDC4", "strokeWidth": 2,
                    "label": label
                })
    return canvas_objects

# --- UI RENDERING ---
st.title("🎯 YOLOv8 Data Annotation Tool")
st.write("Upload images, run auto-detection, and manually adjust annotations.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        # Initialize annotation state for new files
        for f in uploaded_files:
            if f.name not in st.session_state.annotations:
                st.session_state.annotations[f.name] = {"objects": []}

    if not st.session_state.uploaded_files:
        st.info("Please upload one or more images to begin.")
        st.stop()

    st.header("2. Navigation")
    # Dropdown to select image
    filenames = [f.name for f in st.session_state.uploaded_files]
    selected_filename = st.selectbox(
        "Select an image",
        filenames,
        index=st.session_state.current_image_index,
        key="image_selector"
    )
    st.session_state.current_image_index = filenames.index(selected_filename)

    # Navigation buttons
    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.current_image_index == 0)):
        st.session_state.current_image_index -= 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}" # Force remount
        st.experimental_rerun()

    if col2.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_image_index >= len(st.session_state.uploaded_files) - 1)):
        st.session_state.current_image_index += 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}" # Force remount
        st.experimental_rerun()


    st.header("3. Auto-Detection")
    confidence_slider = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    if st.button("🎯 Run Auto-Detect", use_container_width=True):
        current_file = st.session_state.uploaded_files[st.session_state.current_image_index]
        image = Image.open(io.BytesIO(current_file.getvalue()))
        detected_objects = run_yolo_on_image(image, confidence_slider)
        # We replace existing annotations with the new detections
        st.session_state.annotations[selected_filename]["objects"] = detected_objects
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}_{confidence_slider}" # Force remount
        st.experimental_rerun()

    st.header("4. Annotation Tools")
    drawing_mode = st.selectbox(
        "Annotation Mode", ("rect", "path", "freedraw", "transform", "point", "line")
    )
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#eee")

    st.header("5. Export")
    if st.button("💾 Download All Annotations", use_container_width=True):
        # Format the annotations from all images into the desired JSON structure
        export_data = {
            "classes": CLASS_NAMES,
            "images": [
                {
                    "filename": fname,
                    "annotations": data["objects"]
                }
                for fname, data in st.session_state.annotations.items()
            ]
        }
        st.download_button(
            label="Click to Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name="annotations.json",
            mime="application/json"
        )


# --- MAIN CANVAS AREA ---
current_file_obj = st.session_state.uploaded_files[st.session_state.current_image_index]
current_image = Image.open(io.BytesIO(current_file_obj.getvalue()))

# Retrieve existing annotations for the current image
initial_drawing = {"objects": st.session_state.annotations[selected_filename].get("objects", [])}

st.subheader(f"Annotating: `{selected_filename}`")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=current_image,
    update_streamlit=True,
    height=600,
    drawing_mode=drawing_mode,
    initial_drawing={"version": "5.3.0", "objects": initial_drawing["objects"]},
    key=st.session_state.canvas_key, # Use a dynamic key to force re-render on image change
    width=current_image.width
)

# Update the annotations in session state whenever the canvas is updated
if canvas_result.json_data is not None:
    st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]

# Display the annotations data for debugging
st.subheader("Current Annotations (JSON)")
st.json(st.session_state.annotations.get(selected_filename, {}))
