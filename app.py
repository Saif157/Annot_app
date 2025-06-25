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
if 'class_colors' not in st.session_state:
    # Predefined colors for different classes
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
    st.session_state.class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(CLASS_NAMES)}
if 'annotation_stats' not in st.session_state:
    st.session_state.annotation_stats = {}

# --- HELPER FUNCTIONS ---
def run_yolo_on_image(image_as_pil, confidence, selected_classes=None):
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
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Filter by selected classes if specified
                if selected_classes and class_name not in selected_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                color = st.session_state.class_colors.get(class_name, "#FF6B6B")

                canvas_objects.append({
                    "type": "rect", "left": x1, "top": y1,
                    "width": x2 - x1, "height": y2 - y1,
                    "fill": f"{color}33", "stroke": color, "strokeWidth": 2,
                    "label": f"{class_name} ({box.conf[0]:.2f})",
                    "class": class_name,
                    "confidence": float(box.conf[0])
                })

        # Process Polygons
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                class_id = int(result.boxes[i].cls[0])
                class_name = model.names[class_id]

                # Filter by selected classes if specified
                if selected_classes and class_name not in selected_classes:
                    continue

                polygon_points = mask.xy[0].tolist()
                color = st.session_state.class_colors.get(class_name, "#4ECDC4")

                canvas_objects.append({
                    "type": "path",
                    "path": [["M"] + polygon_points[0]] + [["L"] + p for p in polygon_points[1:]] + [["Z"]],
                    "fill": f"{color}33", "stroke": color, "strokeWidth": 2,
                    "label": f"{class_name} ({result.boxes[i].conf[0]:.2f})",
                    "class": class_name,
                    "confidence": float(result.boxes[i].conf[0])
                })
    return canvas_objects

def update_annotation_stats():
    """Update annotation statistics for the dashboard."""
    stats = {
        "total_images": len(st.session_state.uploaded_files),
        "annotated_images": 0,
        "total_objects": 0,
        "class_distribution": {},
        "confidence_stats": []
    }

    for filename, data in st.session_state.annotations.items():
        objects = data.get("objects", [])
        if objects:
            stats["annotated_images"] += 1
            stats["total_objects"] += len(objects)

            for obj in objects:
                obj_class = obj.get("class", "unknown")
                stats["class_distribution"][obj_class] = stats["class_distribution"].get(obj_class, 0) + 1

                if "confidence" in obj:
                    stats["confidence_stats"].append(obj["confidence"])

    st.session_state.annotation_stats = stats

def export_to_yolo_format():
    """Export annotations in YOLO format (txt files)."""
    yolo_data = {}

    for filename, data in st.session_state.annotations.items():
        # Get image dimensions (you might need to store these separately)
        current_file = next(f for f in st.session_state.uploaded_files if f.name == filename)
        image = Image.open(io.BytesIO(current_file.getvalue()))
        img_width, img_height = image.size

        yolo_lines = []
        for obj in data.get("objects", []):
            if obj["type"] == "rect":
                # Convert to YOLO format: class_id center_x center_y width height (normalized)
                x = obj["left"]
                y = obj["top"]
                w = obj["width"]
                h = obj["height"]

                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                class_name = obj.get("class", "unknown")
                class_id = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else 0

                yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

        yolo_data[filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')] = '\n'.join(yolo_lines)

    return yolo_data

def create_download_package():
    """Create a ZIP file containing all annotations in multiple formats."""
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add JSON annotations
        json_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_images": len(st.session_state.uploaded_files),
                "classes": CLASS_NAMES,
                "statistics": st.session_state.annotation_stats
            },
            "classes": CLASS_NAMES,
            "images": [
                {"filename": fname, "annotations": data["objects"]}
                for fname, data in st.session_state.annotations.items()
            ]
        }
        zip_file.writestr("annotations.json", json.dumps(json_data, indent=2))

        # Add YOLO format files
        yolo_data = export_to_yolo_format()
        for filename, content in yolo_data.items():
            zip_file.writestr(f"yolo_labels/{filename}", content)

        # Add classes.txt for YOLO
        zip_file.writestr("yolo_labels/classes.txt", '\n'.join(CLASS_NAMES))

        # Add CSV summary
        csv_data = []
        for filename, data in st.session_state.annotations.items():
            for obj in data.get("objects", []):
                csv_data.append({
                    "filename": filename,
                    "object_type": obj["type"],
                    "class": obj.get("class", "unknown"),
                    "confidence": obj.get("confidence", ""),
                    "x": obj.get("left", ""),
                    "y": obj.get("top", ""),
                    "width": obj.get("width", ""),
                    "height": obj.get("height", "")
                })

        if csv_data:
            df = pd.DataFrame(csv_data)
            zip_file.writestr("summary.csv", df.to_csv(index=False))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# --- UI RENDERING ---
st.title("🎯 YOLOv8 Data Annotation Tool Pro")
st.markdown("*Advanced annotation tool with auto-detection, manual editing, and comprehensive export options*")

# Update stats
update_annotation_stats()

# --- MAIN LAYOUT ---
# Top metrics row
if st.session_state.uploaded_files:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", st.session_state.annotation_stats["total_images"])
    with col2:
        st.metric("Annotated", st.session_state.annotation_stats["annotated_images"])
    with col3:
        st.metric("Total Objects", st.session_state.annotation_stats["total_objects"])
    with col4:
        completion_rate = (st.session_state.annotation_stats["annotated_images"] /
                          st.session_state.annotation_stats["total_images"] * 100) if st.session_state.annotation_stats["total_images"] > 0 else 0
        st.metric("Completion", f"{completion_rate:.1f}%")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Project Management")

    # Upload section
    st.subheader("1. Upload Data")
    uploaded_files = st.file_uploader(
        "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        # Check if new files have been uploaded
        new_filenames = {f.name for f in uploaded_files}
        current_filenames = {f.name for f in st.session_state.get('uploaded_files', [])}
        
        # If the set of files is different, re-initialize state
        if new_filenames != current_filenames:
            st.session_state.uploaded_files = uploaded_files
            st.session_state.annotations = {f.name: {"objects": []} for f in uploaded_files}
            st.session_state.current_image_index = 0
            st.session_state.canvas_key = "canvas_0"
            st.rerun() # Rerun to update the UI with new files

    if not st.session_state.uploaded_files:
        st.info("Please upload one or more images to begin.")
        st.stop()

    # Navigation section
    st.subheader("2. Navigation")
    filenames = [f.name for f in st.session_state.uploaded_files]
    selected_filename = st.selectbox(
        "Select an image", filenames,
        index=st.session_state.current_image_index,
        key="image_selector"
    )
    # Update index if selectbox changes it
    if st.session_state.current_image_index != filenames.index(selected_filename):
        st.session_state.current_image_index = filenames.index(selected_filename)
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Prev", use_container_width=True, disabled=(st.session_state.current_image_index == 0)):
        st.session_state.current_image_index -= 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    if col2.button("Next ➡️", use_container_width=True, disabled=(st.session_state.current_image_index >= len(st.session_state.uploaded_files) - 1)):
        st.session_state.current_image_index += 1
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}"
        st.rerun()

    # Progress indicator
    progress_value = (st.session_state.current_image_index + 1) / len(st.session_state.uploaded_files)
    st.progress(progress_value, text=f"Image {st.session_state.current_image_index + 1} of {len(st.session_state.uploaded_files)}")


    # Auto-detection section
    st.subheader("3. 🤖 Auto-Detection")
    confidence_slider = st.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05)

    # Class filter
    selected_classes = st.multiselect(
        "Filter Classes (empty = all)",
        CLASS_NAMES,
        default=[],
        help="Select specific classes to detect"
    )

    if st.button("🎯 Run Auto-Detect", use_container_width=True, disabled=(not model)):
        current_file = st.session_state.uploaded_files[st.session_state.current_image_index]
        image = Image.open(io.BytesIO(current_file.getvalue()))
        detected_objects = run_yolo_on_image(image, confidence_slider, selected_classes or None)
        st.session_state.annotations[selected_filename]["objects"] = detected_objects
        st.session_state.canvas_key = f"canvas_{st.session_state.current_image_index}_{confidence_slider}"
        st.rerun()

    # Annotation tools section
    st.subheader("4. 🎨 Annotation Tools")
    drawing_mode = st.selectbox("Annotation Mode", ("transform", "rect", "path", "freedraw", "point", "line"))
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.color_picker("Stroke color hex: ", "#FF6B6B")
    bg_color = st.color_picker("Background color hex: ", "#eee")

    # Export section
    st.subheader("5. 📤 Export Options")

    export_format = st.selectbox(
        "Export Format",
        ["Complete Package (ZIP)", "JSON Only", "YOLO Format", "CSV Summary"]
    )

    if st.button("💾 Download Annotations", use_container_width=True):
        if export_format == "Complete Package (ZIP)":
            zip_data = create_download_package()
            st.download_button(
                label="📦 Download ZIP Package",
                data=zip_data,
                file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
        elif export_format == "JSON Only":
            export_data = {
                "classes": CLASS_NAMES,
                "images": [
                    {"filename": fname, "annotations": data["objects"]}
                    for fname, data in st.session_state.annotations.items()
                ]
            }
            st.download_button(
                label="📋 Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name="annotations.json",
                mime="application/json"
            )

    # Statistics section
    st.subheader("📊 Project Statistics")
    if st.session_state.annotation_stats["class_distribution"]:
        st.write("**Class Distribution:**")
        for class_name, count in st.session_state.annotation_stats["class_distribution"].items():
            st.write(f"• {class_name}: {count}")

    if st.session_state.annotation_stats["confidence_stats"]:
        avg_conf = np.mean(st.session_state.annotation_stats["confidence_stats"])
        st.metric("Avg Confidence", f"{avg_conf:.3f}")

# --- MAIN CANVAS AREA ---
if st.session_state.uploaded_files:
    # Ensure current_image_index is valid
    if st.session_state.current_image_index >= len(st.session_state.uploaded_files):
        st.session_state.current_image_index = 0

    current_file_obj = st.session_state.uploaded_files[st.session_state.current_image_index]
    selected_filename = current_file_obj.name
    current_image = Image.open(io.BytesIO(current_file_obj.getvalue()))

    # Ensure annotations entry exists
    if selected_filename not in st.session_state.annotations:
        st.session_state.annotations[selected_filename] = {"objects": []}

    initial_drawing = {"objects": st.session_state.annotations[selected_filename].get("objects", [])}

    # Header with image info and quick actions
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        st.subheader(f"📷 {selected_filename}")
        st.caption(f"Size: {current_image.size[0]}×{current_image.size[1]}px | Objects: {len(initial_drawing['objects'])}")

    with col2:
        if st.button("🎯 Quick Detect", help="Run YOLO with current settings"):
            detected_objects = run_yolo_on_image(current_image, confidence_slider, selected_classes or None)
            st.session_state.annotations[selected_filename]["objects"] = detected_objects
            st.session_state.canvas_key = f"canvas_quick_{st.session_state.current_image_index}"
            st.rerun()

    with col3:
        if st.button("➕ Add to Current", help="Add detections to existing annotations"):
            detected_objects = run_yolo_on_image(current_image, confidence_slider, selected_classes or None)
            current_objects = st.session_state.annotations[selected_filename].get("objects", [])
            st.session_state.annotations[selected_filename]["objects"] = current_objects + detected_objects
            st.session_state.canvas_key = f"canvas_add_{st.session_state.current_image_index}"
            st.rerun()

    with col4:
        if st.button("🗑️ Clear All", help="Remove all annotations"):
            st.session_state.annotations[selected_filename]["objects"] = []
            st.session_state.canvas_key = f"canvas_clear_{st.session_state.current_image_index}"
            st.rerun()

    # Calculate canvas dimensions
    max_width = 900
    max_height = 600

    img_width, img_height = current_image.size
    scale_factor = min(max_width / img_width, max_height / img_height, 1.0)

    canvas_width = int(img_width * scale_factor)
    canvas_height = int(img_height * scale_factor)

    if scale_factor < 1.0:
        display_image = current_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
    else:
        display_image = current_image

    # Canvas
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

    # --- FIX ---
    # The block below was causing the issue. After a button press (e.g., Quick Detect),
    # the script reruns. `st.session_state` is correctly updated with detections, but
    # `canvas_result` still holds the state from *before* the button was pressed (i.e., empty).
    # This code would then incorrectly overwrite the new detections with the old, empty state.
    # By removing it, we make the buttons the single source of truth for their actions.
    # The trade-off is that manual drawings on the canvas will not be saved automatically.
    # A more complex callback system would be needed to support both programmatic and manual updates simultaneously.

    # if canvas_result.json_data is not None:
    #     st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]

    # ---
    
    # You can re-enable saving manual drawings with this more careful logic:
    if canvas_result.json_data is not None and canvas_result.json_data["objects"] != initial_drawing["objects"]:
        st.session_state.annotations[selected_filename]["objects"] = canvas_result.json_data["objects"]
        st.rerun()


    # Show object details in expandable section
    with st.expander("🔍 Object Details", expanded=False):
        objects = st.session_state.annotations.get(selected_filename, {}).get("objects", [])
        if objects:
            for i, obj in enumerate(objects):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.write(f"**Object {i+1}**")
                with col2:
                    obj_type = obj.get("type", "unknown")
                    obj_class = obj.get("class", "manual")
                    confidence = obj.get("confidence", "")
                    conf_text = f" ({confidence:.3f})" if isinstance(confidence, float) else ""
                    st.write(f"{obj_type} - {obj_class}{conf_text}")
                with col3:
                    if st.button("🗑️", key=f"delete_{i}_{selected_filename}", help="Delete this object"):
                        objects.pop(i)
                        st.session_state.annotations[selected_filename]["objects"] = objects
                        st.rerun()
        else:
            st.info("No annotations yet. Use auto-detect or draw manually.")

    # JSON view (collapsible)
    with st.expander("📋 Raw JSON Data", expanded=False):
        st.json(st.session_state.annotations.get(selected_filename, {}))
