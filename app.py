import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
# Import all common YOLOv8 building blocks to trust them
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Bottleneck
from torch.nn import Sequential
import numpy as np
from PIL import Image
import io
import json
import torch

# --- CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Annotation Tool",
    page_icon="🎯",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8-seg model by approving all necessary custom classes."""
    try:
        # Add all necessary Ultralytics classes to the trusted list
        # This includes the specific classes mentioned in the error
        torch.serialization.add_safe_globals([
            SegmentationModel,
            Sequential,
            Conv,
            C2f,
            SPPF,
            Concat,
            Bottleneck,
            # Add the specific ultralytics modules that were causing issues
            'ultralytics.nn.modules.Conv',
            'ultralytics.nn.modules.C2f',
            'ultralytics.nn.modules.SPPF',
            'ultralytics.nn.modules.Concat',
            'ultralytics.nn.modules.Bottleneck',
            'ultralytics.nn.tasks.SegmentationModel'
        ])
        
        # Alternative approach: Load with weights_only=False (less secure but more permissive)
        # This bypasses the strict security check
        model = YOLO('yolov8n-seg.pt')
        
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        
        # If the above fails, try loading with explicit trust settings
        try:
            st.info("Trying alternative loading method...")
            # You can also try setting weights_only=False in the model loading
            # Note: This is less secure but should work around the pickle restrictions
            import os
            os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'True'
            model = YOLO('yolov8n-seg.pt')
            return model
        except Exception as e2:
            st.error(f"Alternative loading method also failed: {e2}")
            st.exception(e2)
            return None

model = load_yolo_model()

# Get class names from the model if loaded, otherwise use a fallback
if model:
    CLASS_NAMES = list(model.names.values())
else:
    CLASS_NAMES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'] # Fallback
    st.warning("YOLO model failed to load. Using fallback class names.")


# --- SESSION STATE INITIALIZATION ---
if 'annotations' not in st.session_state:
    st.session_state.annotations = {}
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_0"

# --- HELPER FUNCTIONS ---
def run_yolo_on_image(image_as_pil, confidence):
    """Performs YOLO prediction and returns results in a Streamlit-friendly format."""
    if not model:
        st.warning("YOLO model not loaded. Cannot perform auto-detection.")
        return []

    img_array = np.array(image_as_pil)
    results = model.predict(source=img_array, conf=confidence, verbose=False)

    canvas_objects = []
    for result in results:
        # Process Boxes
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                canvas_objects.append({
                    "type": "rect", "left": x1, "top": y1,
                    "width": x2 - x1, "height": y2 - y1,
                    "fill": "#FF6B6B33", "stroke": "#FF6B6B", "strokeWidth": 2,
                    "label": f"{model.names[int(box.cls[0])]} ({box.conf[0]:.2f})"
                })
        # Process Polygons
        if result.masks is not None:
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
        for f in uploaded_files:
            if f.name not in st.session_state.annotations:
                st.session_state.annotations[f.name] = {"objects": []}

    if not st.session_state.uploaded_files:
        st.info("Please upload one or more images to begin.")
        st.stop()

    st.header("2. Navigation")
    filenames = [f.name for f in st.session_state.uploaded_files]
    selected_filename = st.selectbox(
        "Select an image", filenames,
        index=st.session_state.current_image_index,
        key="image_selector"
    )
    st.session_state.current_image_index = filenames.index(selected_filename)

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.current_image_index == 0)):
        st.session_state.current_image_index -= 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    if col2.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_image_index >= len(st.session_state.uploaded_files) - 1)):
        st.session_state.current_image_index += 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    st.header("3. Auto-Detection")
    confidence_slider = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    if st.button("🎯 Run Auto-Detect", use_container_width=True, disabled=(not model)):
        current_file = st.session_state.uploaded_files[st.session_state.current_image_index]
        image = Image.open(io.BytesIO(current_file.getvalue()))
        detected_objects = run_yolo_on_image(image, confidence_slider)
        st.session_state.annotations[selected_filename]["objects"] = detected_objects
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}_{confidence_slider}"
        st.rerun()

    st.header("4. Annotation Tools")
    drawing_mode = st.selectbox("Annotation Mode", ("rect", "path", "freedraw", "transform", "point", "line"))
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ")
    bg_color = st.color_picker("Background color hex: ", "#eee")

    st.header("5. Export")
    if st.button("💾 Download All Annotations", use_container_width=True):
        export_data = {
            "classes": CLASS_NAMES,
            "images": [
                {"filename": fname, "annotations": data["objects"]}
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
# Only show the main canvas if we have uploaded files
if st.session_state.uploaded_files:
    current_file_obj = st.session_state.uploaded_files[st.session_state.current_image_index]
    current_image = Image.open(io.BytesIO(current_file_obj.getvalue()))

    initial_drawing = {"objects": st.session_state.annotations[selected_filename].get("objects", [])}

    st.subheader(f"Annotating: `{selected_filename}`")
    
    # Show model status
    if model:
        st.success("✅ YOLO model loaded successfully")
    else:
        st.warning("⚠️ YOLO model failed to load - manual annotation only")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=current_image,
        update_streamlit=True,
        height=600,
        drawing_mode=drawing_mode,
        initial_drawing={"version": "5.3.0", "objects": initial_drawing["objects"]},
        key=st.session_state.canvas_key,
        width=current_image.width
    )

    if canvas_result.json_data is not None:
        st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]

    st.subheader("Current Annotations (JSON)")
    st.json(st.session_state.annotations.get(selected_filename, {}))
