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
import zipfile
from datetime import datetime
import base64

# --- CONFIGURATION ---
st.set_page_config(
    page_title="YOLOv8 Annotation Tool Pro",
    page_icon="🎯",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8-seg model with multiple fallback approaches."""
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        import ultralytics.nn.modules as modules
        import ultralytics.nn.tasks as tasks
        safe_classes = [getattr(modules, cls) for cls in dir(modules) if isinstance(getattr(modules, cls), type)]
        safe_classes += [getattr(tasks, cls) for cls in dir(tasks) if isinstance(getattr(tasks, cls), type)]
        safe_classes.extend([Sequential])
        torch.serialization.add_safe_globals(safe_classes)
        model = YOLO('yolov8n-seg.pt')
        st.success("✅ Model loaded with safe globals method")
        return model
    except Exception as e1:
        st.warning(f"Safe globals method failed: {e1}")
    
    try:
        st.info("Trying monkey patch method...")
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        model = YOLO('yolov8n-seg.pt')
        torch.load = original_load
        st.success("✅ Model loaded with monkey patch method")
        return model
    except Exception as e2:
        st.warning(f"Monkey patch method failed: {e2}")
        if 'original_load' in locals(): torch.load = original_load
    
    try:
        st.info("Trying compatibility mode...")
        import os
        os.environ['TORCH_SERIALIZATION_WEIGHTS_ONLY'] = 'False'
        model = YOLO('yolov8n-seg.pt')
        st.success("✅ Model loaded with compatibility mode")
        return model
    except Exception as e3:
        st.error(f"All loading methods failed. Error: {e3}")
        return None

model = load_yolo_model()

if model:
    CLASS_NAMES = list(model.names.values())
else:
    CLASS_NAMES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
    st.warning("YOLO model failed to load. Using fallback class names.")

# --- SESSION STATE INITIALIZATION ---
if 'annotations' not in st.session_state: st.session_state.annotations = {}
if 'current_image_index' not in st.session_state: st.session_state.current_image_index = 0
if 'uploaded_files' not in st.session_state: st.session_state.uploaded_files = []
if 'canvas_key' not in st.session_state: st.session_state.canvas_key = "canvas_0"
if 'class_colors' not in st.session_state:
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    st.session_state.class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(CLASS_NAMES)}
if 'annotation_stats' not in st.session_state: st.session_state.annotation_stats = {}

# --- HELPER FUNCTIONS ---
def run_yolo_on_image(image_as_pil, confidence, selected_classes=None):
    if not model:
        st.warning("YOLO model not loaded. Cannot perform auto-detection.")
        return []
    img_array = np.array(image_as_pil)
    results = model.predict(source=img_array, conf=confidence, verbose=False)
    canvas_objects = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if selected_classes and class_name not in selected_classes: continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                color = st.session_state.class_colors.get(class_name, "#FF6B6B")
                canvas_objects.append({
                    "type": "rect", "left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1,
                    "fill": "rgba(0,0,0,0)", "stroke": color, "strokeWidth": 3,
                    "label": f"{class_name} ({box.conf[0]:.2f})", "class": class_name,
                    "confidence": float(box.conf[0])
                })
    return canvas_objects

def update_annotation_stats():
    stats = {"total_images": len(st.session_state.uploaded_files), "annotated_images": 0, "total_objects": 0, "class_distribution": {}, "confidence_stats": []}
    for filename, data in st.session_state.annotations.items():
        objects = data.get("objects", [])
        if objects:
            stats["annotated_images"] += 1
            stats["total_objects"] += len(objects)
            for obj in objects:
                obj_class = obj.get("class", "unknown")
                stats["class_distribution"][obj_class] = stats["class_distribution"].get(obj_class, 0) + 1
                if "confidence" in obj: stats["confidence_stats"].append(obj["confidence"])
    st.session_state.annotation_stats = stats

# --- UI RENDERING ---
st.title("🎯 YOLOv8 Data Annotation Tool Pro")
st.markdown("*Advanced annotation tool with auto-detection, manual editing, and comprehensive export options*")

update_annotation_stats()

if st.session_state.uploaded_files:
    cols = st.columns(4)
    cols[0].metric("Total Images", st.session_state.annotation_stats["total_images"])
    cols[1].metric("Annotated", st.session_state.annotation_stats["annotated_images"])
    cols[2].metric("Total Objects", st.session_state.annotation_stats["total_objects"])
    completion = (st.session_state.annotation_stats["annotated_images"] / st.session_state.annotation_stats["total_images"] * 100) if st.session_state.annotation_stats["total_images"] > 0 else 0
    cols[3].metric("Completion", f"{completion:.1f}%")

with st.sidebar:
    st.header("🗂️ Project Management")
    st.subheader("1. Upload Data")
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        for f in uploaded_files:
            if f.name not in st.session_state.annotations:
                st.session_state.annotations[f.name] = {"objects": []}
    if not st.session_state.uploaded_files:
        st.info("Please upload one or more images to begin.")
        st.stop()

    st.subheader("2. Navigation")
    filenames = [f.name for f in st.session_state.uploaded_files]
    selected_filename = st.selectbox("Select an image", filenames, index=st.session_state.current_image_index, key="image_selector")
    if filenames.index(selected_filename) != st.session_state.current_image_index:
        st.session_state.current_image_index = filenames.index(selected_filename)
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    nav_cols = st.columns(2)
    if nav_cols[0].button("⬅️ Prev", use_container_width=True, disabled=(st.session_state.current_image_index == 0)):
        st.session_state.current_image_index -= 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()
    if nav_cols[1].button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_image_index >= len(filenames) - 1)):
        st.session_state.current_image_index += 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()
    st.progress((st.session_state.current_image_index + 1) / len(filenames), text=f"Image {st.session_state.current_image_index + 1} of {len(filenames)}")

    st.subheader("3. 🤖 Auto-Detection")
    confidence_slider = st.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05)
    selected_classes = st.multiselect("Filter Classes (empty = all)", CLASS_NAMES, default=[], help="Select specific classes to detect")
    if st.button("🎯 Run Auto-Detect", use_container_width=True, disabled=(not model)):
        current_file = st.session_state.uploaded_files[st.session_state.current_image_index]
        image = Image.open(io.BytesIO(current_file.getvalue())).convert("RGB")
        detected_objects = run_yolo_on_image(image, confidence_slider, selected_classes or None)
        st.session_state.annotations[selected_filename]["objects"] = detected_objects
        st.session_state.canvas_key = f"canvas_detect_{st.session_state.current_image_index}"
        st.rerun()

    st.subheader("4. 🎨 Annotation Tools")
    drawing_mode = st.selectbox("Annotation Mode", ("transform", "rect", "path", "freedraw", "point", "line"))
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ", "#FF6B6B")
    # This color picker is now for drawing without a background image, it won't affect the main canvas.
    bg_color_picker = st.color_picker("Background color hex: ", "#eee")

if st.session_state.uploaded_files:
    current_file_obj = st.session_state.uploaded_files[st.session_state.current_image_index]
    current_image = Image.open(io.BytesIO(current_file_obj.getvalue())).convert("RGB")
    initial_drawing = {"objects": st.session_state.annotations[selected_filename].get("objects", [])}

    header_cols = st.columns([3, 1, 1, 1])
    header_cols[0].subheader(f"📷 {selected_filename}")
    header_cols[0].caption(f"Size: {current_image.size[0]}×{current_image.size[1]}px | Objects: {len(initial_drawing['objects'])}")
    if header_cols[1].button("🎯 Quick Detect", help="Run YOLO with current settings"):
        detected_objects = run_yolo_on_image(current_image, confidence_slider, selected_classes or None)
        st.session_state.annotations[selected_filename]["objects"] = detected_objects
        st.session_state.canvas_key = f"canvas_quick_{st.session_state.current_image_index}"
        st.rerun()
    if header_cols[2].button("➕ Add to Current", help="Add detections to existing annotations"):
        detected_objects = run_yolo_on_image(current_image, confidence_slider, selected_classes or None)
        current_objects = st.session_state.annotations[selected_filename]["objects"]
        st.session_state.annotations[selected_filename]["objects"] = current_objects + detected_objects
        st.session_state.canvas_key = f"canvas_add_{st.session_state.current_image_index}"
        st.rerun()
    if header_cols[3].button("🗑️ Clear All", help="Remove all annotations"):
        st.session_state.annotations[selected_filename]["objects"] = []
        st.session_state.canvas_key = f"canvas_clear_{st.session_state.current_image_index}"
        st.rerun()

    max_width, max_height = 900, 600
    img_width, img_height = current_image.size
    scale_factor = min(max_width / img_width, max_height / img_height, 1.0)
    canvas_width = int(img_width * scale_factor)
    canvas_height = int(img_height * scale_factor)
    display_image = current_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS) if scale_factor < 1.0 else current_image

    # --- IMAGE NOT SHOWING FIX ---
    # The background_color is set to be transparent to ensure the background_image is visible.
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="rgba(0, 0, 0, 0)",
        background_image=display_image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        initial_drawing={"version": "5.3.0", "objects": initial_drawing["objects"]},
        key=st.session_state.canvas_key
    )

    if canvas_result.json_data is not None and canvas_result.json_data["objects"] != initial_drawing["objects"]:
        st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]
        st.rerun()

    with st.expander("🔍 Object Details"):
        objects = st.session_state.annotations.get(selected_filename, {}).get("objects", [])
        if objects:
            for i, obj in enumerate(objects):
                detail_cols = st.columns([1, 2, 1])
                detail_cols[0].write(f"**Object {i+1}**")
                obj_class = obj.get("class", "manual")
                conf_text = f" ({obj.get('confidence', 0):.3f})" if 'confidence' in obj else ""
                detail_cols[1].write(f"{obj.get('type', 'unknown')} - {obj_class}{conf_text}")
                if detail_cols[2].button("🗑️", key=f"delete_{i}_{selected_filename}", help="Delete this object"):
                    objects.pop(i)
                    st.session_state.annotations[selected_filename]["objects"] = objects
                    st.rerun()
        else:
            st.info("No annotations yet. Use auto-detect or draw manually.")

    with st.expander("📋 Raw JSON Data"):
        st.json(st.session_state.annotations.get(selected_filename, {}))
