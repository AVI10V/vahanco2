import streamlit as st
import cv2
import tempfile
import pandas as pd
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(
    page_title="VAHAN C0-2",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Dark & Realistic UI ---
st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Section headers */
    h1, h2, h3 {
        color: #58a6ff; /* A bright blue for headers */
    }
    /* Metric labels and values */
    .stMetric .st-ax {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }
    .stMetric > label {
        color: #8b949e; /* Lighter gray for metric labels */
    }
    .stMetric > div > div > span {
        color: #c9d1d9; /* Main text color for metric values */
    }
    /* Style for containers */
    .st-emotion-cache-1r4qj8v {
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        background-color: #161b22;
    }
    /* Button and file uploader styling */
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
    }
    .st-emotion-cache-1erivf3 {
        background-color: #21262d;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Title ---
st.title("VAHAN CO2")
st.markdown("Upload a video to detect vehicles and estimate potential CO₂ emissions in real-time.")
st.divider()


# --- Model & Class Configuration ---
# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the vehicle classes we want to detect (based on COCO dataset names)
VEHICLE_CLASSES = ['car', 'bus', 'bicycle', 'motorcycle', 'truck']

# Approximate CO₂ emissions in grams per kilometer (g/km)
# This is a simplified model for demonstration purposes.
POLLUTION_FACTORS = {
    'car': 120,
    'bus': 1000,
    'bicycle': 0,
    'motorcycle': 90,
    'truck': 1500,
}


# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded_file is not None:
    # Use a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # --- Streamlit UI Placeholders ---
    # Create two columns for video and dashboard
    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.header("🎬 Live Video Feed")
        video_placeholder = st.empty()

    with col2:
        st.header("📊 Real-time Dashboard")
        # Placeholders for KPIs, charts, and metrics
        kpi_placeholder = st.empty()
        frame_metrics_placeholder = st.empty()
        st.divider()
        history_placeholder = st.empty()


    # --- Processing Loop ---
    # Initialize data storage
    vehicle_counts = {cls: 0 for cls in VEHICLE_CLASSES}
    total_pollution = 0
    pollution_history = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("✅ Video processing finished.")
            break

        frame_count += 1
        
        # Perform inference on the frame
        results = model(frame, verbose=False) # verbose=False to suppress console output
        
        # Reset counts for the current frame
        frame_counts = {cls: 0 for cls in VEHICLE_CLASSES}
        
        # Process detections
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if class_name in VEHICLE_CLASSES:
                    # Increment count for detected vehicle type in this frame
                    frame_counts[class_name] += 1
                    
                    # Draw bounding box and label on the frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 222, 100), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (28, 222, 100), 2)

        # Update total running counts
        for v_type, count in frame_counts.items():
            vehicle_counts[v_type] += count

        # --- Dashboard Updates ---
        # Calculate pollution for the current frame
        current_pollution = sum(frame_counts[v_type] * POLLUTION_FACTORS[v_type] for v_type in frame_counts)
        total_pollution += current_pollution
        pollution_history.append({'Frame': frame_count, 'CO2_Index': current_pollution})
        
        total_vehicles = sum(vehicle_counts.values())

        # Update the main KPIs
        with kpi_placeholder.container():
            st.subheader("📈 Overall Summary")
            kpi_cols = st.columns(2)
            kpi_cols[0].metric(label="Total Frames  Captured so far", value=f"{total_vehicles:,}")
            kpi_cols[1].metric(label="Net CO₂ Index (g/km)", value=f"{total_pollution / 1000:,.2f} gm")
            st.caption("The CO₂ Index is a proxy value based on vehicle counts and average emission factors.")


        # Update the metrics for the current frame
        with frame_metrics_placeholder.container():
            st.subheader("👁️ Current Frame Analysis")
            m_cols = st.columns(len(VEHICLE_CLASSES))
            for i, (v_type, count) in enumerate(frame_counts.items()):
                m_cols[i].metric(label=v_type.capitalize(), value=count)
        
        # Update historical charts
        with history_placeholder.container():
            st.subheader("🗂️ Historical Data")
            tab1, tab2 = st.tabs(["Vehicle Distribution", "Pollution Over Time"])
            
            with tab1:
                chart_data = pd.DataFrame(list(vehicle_counts.items()), columns=['Vehicle Type', 'Count'])
                st.bar_chart(chart_data.set_index('Vehicle Type'), color="#3498db")

            with tab2:
                line_chart_data = pd.DataFrame(pollution_history).set_index('Frame')
                st.line_chart(line_chart_data, color="#e74c3c")


        # Display the processed frame
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

    # Release video capture object
    cap.release()

else:
    st.info("👋 Welcome! Please upload a video file to begin analysis.")