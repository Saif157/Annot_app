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
    """Loads the YOLOv8-seg model with multiple fallback approaches."""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Method 1: Try loading with comprehensive safe globals
    try:
        # Get all ultralytics classes dynamically
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks
        
        # Collect all classes from the modules
        safe_classes = []
        for name in dir(modules):
            attr = getattr(modules, name)
            if isinstance(attr, type):
                safe_classes.append(attr)
        
        for name in dir(tasks):
            attr = getattr(tasks, name)
            if isinstance(attr, type):
                safe_classes.append(attr)
        
        # Add torch classes
        safe_classes.extend([Sequential])
        
        torch.serialization.add_safe_globals(safe_classes)
        model = YOLO('yolov8n-seg.pt')
        st.success("✅ Model loaded with safe globals method")
        return model
        
    except Exception as e1:
        st.warning(f"Safe globals method failed: {e1}")
    
    # Method 2: Monkey patch the torch.load function
    try:
        st.info("Trying monkey patch method...")
        import torch
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        model = YOLO('yolov8n-seg.pt')
        torch.load = original_load  # Restore original
        st.success("✅ Model loaded with monkey patch method")
        return model
        
    except Exception as e2:
        st.warning(f"Monkey patch method failed: {e2}")
        torch.load = original_load  # Restore original even if failed
    
    # Method 3: Use older PyTorch serialization
    try:
        st.info("Trying compatibility mode...")
        import pickle
        import os
        
        # Temporarily disable pickle restrictions
        os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = 'False'
        
        model = YOLO('yolov8n-seg.pt')
        st.success("✅ Model loaded with compatibility mode")
        return model
        
    except Exception as e3:
        st.error(f"All loading methods failed. Error: {e3}")
        st.info("🔧 **Workaround suggestions:**")
        st.info("1. Try running: `pip install torch==1.13.1 ultralytics==8.0.20`")
        st.info("2. Or set environment variable: `export TORCH_SERIALIZATION_WEIGHTS_ONLY=False`")
        st.info("3. The app will work in manual annotation mode")
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
    
    # Show image info and instructions
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info(f"📐 Image size: {current_image.size[0]}×{current_image.size[1]} pixels")
    with col2:
        if st.button("🎯 Quick Auto-Detect", help="Run YOLO detection with current confidence"):
            detected_objects = run_yolo_on_image(current_image, confidence_slider)
            st.session_state.annotations[selected_filename]["objects"] = detected_objects
            st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}_{confidence_slider}"
            st.rerun()
    with col3:
        if st.button("🗑️ Clear All", help="Clear all annotations"):
            st.session_state.annotations[selected_filename]["objects"] = []
            st.session_state.canvas_key = f"canvas_clear_{st.session_state.current_image_index}"
            st.rerun()

    # Calculate canvas dimensions to fit the image properly
    max_width = 800
    max_height = 600
    
    # Scale image to fit in canvas while maintaining aspect ratio
    img_width, img_height = current_image.size
    scale_factor = min(max_width / img_width, max_height / img_height, 1.0)
    
    canvas_width = int(img_width * scale_factor)
    canvas_height = int(img_height * scale_factor)
    
    # Resize image for canvas if needed
    if scale_factor < 1.0:
        display_image = current_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    else:
        display_image = current_image

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=display_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        initial_drawing={"version": "5.3.0", "objects": initial_drawing["objects"]},
        key=st.session_state.canvas_key
    )

    if canvas_result.json_data is not None:
        st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]

    st.subheader("Current Annotations (JSON)")
    st.json(st.session_state.annotations.get(selected_filename, {}))
