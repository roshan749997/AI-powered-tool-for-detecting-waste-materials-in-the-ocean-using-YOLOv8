import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Load the YOLO model
model_path = r"C:\Users\Roshan\OneDrive\Desktop\Ocean Plastics Waste Detection\best.pt"
model = YOLO(model_path)

# Streamlit App Configuration
st.set_page_config(
    page_title="Ocean Plastics Waste Detection",
    page_icon="🌊",
    layout="wide",
)

# --- Sidebar Customization ---
st.sidebar.image(r"C:\Users\Roshan\OneDrive\Desktop\Ocean Plastics Waste Detection\avcoe_logo.jpeg", use_container_width=True)  # Optional: Add logo image to sidebar

# Adding CSS to make sidebar more responsive
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #4B9CD3;
        }
        @media (max-width: 600px) {
            .sidebar .sidebar-content {
                background-color: #2F6E98;
            }
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
st.sidebar.markdown("""
    Explore different sections of the app:
    - **Upload Image**: Upload an image for detection.
    - **About Model**: Learn about the model and dataset.
    - **Instructions**: How to use the app.
""")

# --- Main Page ---
st.title("🌊 Ocean Plastics Waste Detection")
st.write("**AI-powered tool for detecting waste materials in the ocean using YOLOv8.**")
st.markdown("Upload an image to detect plastics, glass, metal, and trash in ocean environments.")

# --- Add Attractive Tabs ---
tabs = st.tabs(
    [
        "🔍 Upload Image",  # Upload Image tab with search icon
        "📊 About Model",   # About Model tab with bar chart icon
        "📖 Instructions"   # Instructions tab with book icon
    ]
)

# --- Tab 1: Upload and Detect ---
with tabs[0]:
    st.subheader("Step 1: Upload an Image")
    uploaded_file = st.file_uploader("Choose an image (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save the uploaded image locally
        image_path = "uploaded_image.jpg"
        image.save(image_path)

        st.subheader("Step 2: Detection in Progress")
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)

        # Perform YOLO prediction
        results = model.predict(source=image_path, conf=0.25)

        st.success("✅ Detection Completed!")
        st.subheader("Detection Results")

        # Create a copy of the original image to draw bounding boxes on
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        result_data = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = box.conf[0]
                # Extract the box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0]
                # Draw the bounding box on the image
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                # Add class name and confidence score text on the image
                draw.text((x1, y1 - 10), f"{model.names[cls]} {confidence:.2f}", fill="red")
                result_data.append(f"**Class:** {model.names[cls]} | **Confidence:** {confidence:.2f}")

        # Display the detection results
        st.write("\n".join(result_data))

        # Save and display the annotated image with bounding boxes
        annotated_image_path = "annotated_image.jpg"
        draw_image.save(annotated_image_path)
        
        # Resize the image to a more appropriate size
        max_width = 800  # Set a max width for the image
        max_height = 600  # Set a max height for the image
        annotated_image = Image.open(annotated_image_path)
        annotated_image = annotated_image.resize((max_width, max_height), Image.Resampling.LANCZOS)

        # Display the resized annotated image
        st.subheader("Annotated Image with Bounding Boxes")
        st.image(annotated_image, caption="Detection Results", use_container_width=True)

# --- Tab 2: About Model ---
with tabs[1]:
    st.subheader("About the Model")
    st.markdown("""
    - **YOLOv8**: The latest version of the YOLO (You Only Look Once) model, optimized for object detection.
    - **Dataset**: Custom-trained using the Ocean Plastics Waste dataset.
    - **Classes Detected**: 
      - Glass
      - Metal
      - Plastic
      - Trash
    - **Confidence Threshold**: 25%
    """)
    st.info("The model was trained using Roboflow for high accuracy in ocean plastics waste detection.")

# --- Tab 3: Instructions ---
with tabs[2]:
    st.subheader("How to Use the App")
    st.markdown("""
    1. Go to the **Upload Image** tab.
    2. Upload an image in JPG/PNG format.
    3. Wait for detection to complete.
    4. View results: detected objects, class names, and confidence.
    5. Download the annotated image for further use.
    """)
    st.warning("Ensure the uploaded image is clear and contains visible waste for better detection results.")
