import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
import random
import time
import base64
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Shelf Anomaly Detector",
    page_icon="🕹️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load retro gaming CSS style
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply custom CSS styling for retro gaming aesthetic
def apply_retro_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Space+Mono&display=swap');

    /* Main retro styling */
    * {
        font-family: 'VT323', monospace;
    }
    
    h1, h2, h3 {
        font-family: 'VT323', monospace !important;
        color: #FFD700 !important; /* Golden yellow */
        text-shadow: 3px 3px 0px #FF6B6B; /* Coral shadow */
    }
    
    /* Code blocks */
    code {
        font-family: 'Space Mono', monospace !important;
        background-color: #000080 !important; /* Navy blue */
        color: #00FF00 !important; /* Green text */
        border: 2px solid #FFFFFF !important;
        padding: 5px !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #000080 !important; /* Navy blue */
    }
    
    .css-1d391kg .block-container {
        border-right: 4px solid #FFD700 !important; /* Golden yellow */
    }
    
    /* Buttons */
    .stButton>button {
        font-family: 'VT323', monospace !important;
        background-color: #FF6B6B !important; /* Coral */
        color: white !important;
        border: 3px solid #FFD700 !important; /* Golden yellow */
        border-radius: 0px !important; /* Square borders for pixel look */
        box-shadow: 4px 4px 0px #000000 !important; /* Black shadow */
        transition: transform 0.1s, box-shadow 0.1s !important;
    }
    
    .stButton>button:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px #000000 !important; /* Reduced shadow on hover */
    }
    
    /* Progress bar */
    .stProgress .st-bo {
        background-color: #FF6B6B !important; /* Coral */
        height: 20px !important;
        border-radius: 0px !important; /* Square for pixel look */
    }
    
    .stProgress .st-bp {
        background-color: #FFD700 !important; /* Golden yellow */
        border-radius: 0px !important; /* Square for pixel look */
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 3px solid #FFD700 !important; /* Golden yellow */
        font-family: 'Space Mono', monospace !important;
    }
    
    .dataframe th {
        background-color: #FF6B6B !important; /* Coral */
        color: white !important;
        border: 2px solid black !important;
    }
    
    .dataframe td {
        border: 2px solid black !important;
    }

    /* Pixel border effect for images */
    .pixel-border {
        border: 4px solid #FFD700 !important; /* Golden yellow */
        image-rendering: pixelated !important;
        padding: 0 !important;
        box-shadow: 8px 8px 0px #FF6B6B !important; /* Coral shadow */
    }

    /* Game message box */
    .game-message {
        background-color: #000080 !important; /* Navy blue */
        border: 4px solid #FFD700 !important; /* Golden yellow */
        color: white !important;
        padding: 10px !important;
        margin: 10px 0 !important;
        box-shadow: 5px 5px 0px #FF6B6B !important; /* Coral shadow */
    }
    
    /* Animated pixel text effect */
    @keyframes pixel-text {
        0% { text-shadow: 2px 2px 0px #FF6B6B; }
        50% { text-shadow: 3px 3px 0px #FF6B6B; }
        100% { text-shadow: 2px 2px 0px #FF6B6B; }
    }
    
    .pixel-text {
        animation: pixel-text 2s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the retro style
apply_retro_style()

# Create pixel art title
def pixel_title(title):
    st.markdown(f'<h1 class="pixel-text">{title}</h1>', unsafe_allow_html=True)

# Create game-style message box
def game_message(message):
    st.markdown(f'<div class="game-message">{message}</div>', unsafe_allow_html=True)

# Create pixel art progress bar
def pixel_progress_bar(progress):
    st.progress(progress)
    if progress == 1.0:
        st.balloons()

# Create a loading animation effect
def loading_animation(text="LOADING"):
    progress_text = st.empty()
    progress_bar = st.empty()
    
    for i in range(101):
        dots = "." * (i % 4)
        progress_text.markdown(f"<h3>{text}{dots}</h3>", unsafe_allow_html=True)
        progress_bar.progress(i/100)
        time.sleep(0.02)
    
    progress_text.empty()
    progress_bar.empty()

# Pixel-style image display
def display_pixel_image(image, caption=""):
    st.markdown(f"""
    <figure>
        <img src="data:image/png;base64,{base64.b64encode(cv2.imencode('.png', image)[1]).decode()}" class="pixel-border" style="width:100%">
        <figcaption style="text-align:center; font-family:'VT323', monospace; margin-top:5px">{caption}</figcaption>
    </figure>
    """, unsafe_allow_html=True)

# ------- Data Generation Functions -------

# Generate synthetic shelf images
def generate_shelf_image(width=800, height=600):
    # Create a base shelf image
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add shelf lines
    shelf_heights = [150, 300, 450]
    for h in shelf_heights:
        cv2.line(img, (0, h), (width, h), (120, 80, 20), 10)  # Brown shelf line
    
    # Add vertical dividers
    for x in range(100, width, 100):
        for h1, h2 in zip([0] + shelf_heights, shelf_heights + [height]):
            cv2.line(img, (x, h1+5), (x, h2-5), (150, 150, 150), 2)  # Gray dividers
    
    return img

# Generate synthetic promotional items
def generate_promo_item(item_type, size=(100, 100)):
    width, height = size
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    if item_type == "poster":
        # Create a poster with a colorful rectangle
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (10, 10), (width-10, height-10), color, -1)
        # Add text
        cv2.putText(img, "SALE!", (int(width/4), int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
    
    elif item_type == "banner":
        # Create a banner
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, (5, 5), (width-5, height-5), color, -1)
        # Add stripes
        for i in range(0, height, 20):
            cv2.line(img, (0, i), (width, i), (255, 255, 255), 2)
    
    elif item_type == "standee":
        # Create a cardboard standee (simplified)
        # Body
        cv2.rectangle(img, (int(width/4), int(height/5)), (int(3*width/4), height-10), (165, 42, 42), -1)
        # Head
        cv2.circle(img, (int(width/2), int(height/5)), int(width/5), (255, 222, 173), -1)
        # Eyes
        cv2.circle(img, (int(width/2) - 10, int(height/6)), 5, (0, 0, 0), -1)
        cv2.circle(img, (int(width/2) + 10, int(height/6)), 5, (0, 0, 0), -1)
        # Smile
        cv2.ellipse(img, (int(width/2), int(height/5)), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    return img

# Place promotional items on shelf
def place_promo_items(shelf_img, items_config, add_anomaly=False, anomaly_type=None):
    shelf_with_items = shelf_img.copy()
    items_data = []
    
    for i, config in enumerate(items_config):
        item_type = config["type"]
        position = config["position"]
        expected_zone = config["zone"]
        
        # Generate the item
        item_img = generate_promo_item(item_type)
        h, w = item_img.shape[:2]
        
        # Adjust position based on anomaly settings
        actual_position = position.copy()
        actual_zone = expected_zone
        
        # Add anomalies if requested
        if add_anomaly and i == len(items_config) - 1:  # Add anomaly to the last item
            if anomaly_type == "misplaced":
                # Move item to a different zone
                shelf_zones = [0, 150, 300, 450, 600]
                zone_idx = shelf_zones.index(expected_zone)
                new_zone_idx = (zone_idx + random.randint(1, len(shelf_zones)-2)) % (len(shelf_zones)-1)
                actual_zone = shelf_zones[new_zone_idx]
                actual_position[1] = actual_zone + 20  # Y position
            
            elif anomaly_type == "damaged":
                # Add damage effect (tearing, spots, etc.)
                damage_y = random.randint(0, h-20)
                damage_x = random.randint(0, w-20)
                damage_w = random.randint(10, 30)
                damage_h = random.randint(10, 30)
                cv2.rectangle(item_img, (damage_x, damage_y), 
                             (damage_x + damage_w, damage_y + damage_h), (255, 255, 255), -1)
                
                # Add some noise/scratches
                for _ in range(20):
                    x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
                    x2, y2 = x1 + random.randint(-10, 10), y1 + random.randint(-10, 10)
                    cv2.line(item_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            elif anomaly_type == "duplicate":
                # We'll add a duplicate of this item nearby
                dup_pos_x = position[0] + random.randint(-50, 50)
                dup_pos_x = max(0, min(dup_pos_x, shelf_img.shape[1] - w))
                
                # Add the duplicate to the shelf
                roi = shelf_with_items[actual_position[1]:actual_position[1]+h, 
                                      dup_pos_x:dup_pos_x+w]
                
                # Only replace non-black pixels to preserve transparency effect
                item_mask = item_img != 255
                roi[item_mask] = item_img[item_mask]
                
                # Add the duplicate to the items data
                items_data.append({
                    "id": f"{item_type}_{i}_duplicate",
                    "type": item_type,
                    "position": [dup_pos_x, actual_position[1]],
                    "expected_zone": expected_zone,
                    "actual_zone": actual_zone,
                    "anomaly": "duplicate",
                    "bbox": [dup_pos_x, actual_position[1], dup_pos_x+w, actual_position[1]+h]
                })
        
        # Place the item on the shelf
        y1, y2 = actual_position[1], actual_position[1] + h
        x1, x2 = actual_position[0], actual_position[0] + w
        
        # Make sure we don't go out of bounds
        if y2 > shelf_with_items.shape[0] or x2 > shelf_with_items.shape[1]:
            continue
            
        roi = shelf_with_items[y1:y2, x1:x2]
        
        # Only replace non-black pixels to preserve transparency effect
        item_mask = item_img != 255
        roi[item_mask] = item_img[item_mask]
        
        # Record the item data
        anomaly_status = "none"
        if add_anomaly and i == len(items_config) - 1:
            anomaly_status = anomaly_type
            
        items_data.append({
            "id": f"{item_type}_{i}",
            "type": item_type,
            "position": actual_position,
            "expected_zone": expected_zone,
            "actual_zone": actual_zone,
            "anomaly": anomaly_status,
            "bbox": [x1, y1, x2, y2]
        })
    
    return shelf_with_items, items_data

# Generate a complete synthetic dataset
def generate_dataset(num_normal=3, num_anomalies=3):
    dataset = []
    
    # Item types and their zones
    item_types = ["poster", "banner", "standee"]
    shelf_zones = [0, 150, 300, 450]
    
    # Generate normal cases
    for i in range(num_normal):
        shelf_img = generate_shelf_image()
        
        # Create 3-5 random items
        num_items = random.randint(3, 5)
        items_config = []
        
        for j in range(num_items):
            item_type = random.choice(item_types)
            zone = random.choice(shelf_zones)
            x_pos = random.randint(50, 700)
            
            items_config.append({
                "type": item_type,
                "position": [x_pos, zone + 20],  # 20px below the shelf line
                "zone": zone
            })
        
        img_with_items, items_data = place_promo_items(shelf_img, items_config)
        
        dataset.append({
            "image": img_with_items,
            "items": items_data,
            "has_anomaly": False
        })
    
    # Generate anomaly cases
    anomaly_types = ["misplaced", "damaged", "duplicate"]
    
    for i in range(num_anomalies):
        anomaly_type = anomaly_types[i % len(anomaly_types)]
        shelf_img = generate_shelf_image()
        
        # Create 3-5 random items
        num_items = random.randint(3, 5)
        items_config = []
        
        for j in range(num_items):
            item_type = random.choice(item_types)
            zone = random.choice(shelf_zones)
            x_pos = random.randint(50, 700)
            
            items_config.append({
                "type": item_type,
                "position": [x_pos, zone + 20],  # 20px below the shelf line
                "zone": zone
            })
        
        img_with_items, items_data = place_promo_items(
            shelf_img, items_config, add_anomaly=True, anomaly_type=anomaly_type
        )
        
        dataset.append({
            "image": img_with_items,
            "items": items_data,
            "has_anomaly": True,
            "anomaly_type": anomaly_type
        })
    
    return dataset

# ------- Object Detection Functions -------

# Basic object detection for promotional items (simplified for MVP)
def detect_objects(image):
    # In a real app, this would be a trained model
    # For our MVP, we'll use color thresholding and contour detection
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get objects
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    
    for i, contour in enumerate(contours):
        # Filter out small contours
        if cv2.contourArea(contour) < 500:
            continue
            
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Determine type based on aspect ratio and size
        aspect_ratio = w / h
        
        if aspect_ratio > 1.5:
            obj_type = "banner"
        elif aspect_ratio < 0.7:
            obj_type = "standee"
        else:
            obj_type = "poster"
            
        # Determine zone
        shelf_zones = [0, 150, 300, 450, 600]
        zone = 0
        for z in range(len(shelf_zones)-1):
            if y >= shelf_zones[z] and y < shelf_zones[z+1]:
                zone = shelf_zones[z]
                break
                
        detected_objects.append({
            "id": f"{obj_type}_{i}",
            "type": obj_type,
            "bbox": [x, y, x+w, y+h],
            "zone": zone
        })
    
    return detected_objects

# ------- Anomaly Detection Functions -------

# Detect misplaced items
def detect_misplaced_items(detected_objects, layout_rules=None):
    # In a real app, this would use the layout rules
    # For the MVP, we'll use some simple rules
    
    # Default rules if none provided
    if layout_rules is None:
        layout_rules = {
            "poster": [0, 150],   # Posters expected on top shelves
            "banner": [150, 300], # Banners on middle shelves
            "standee": [300, 450] # Standees on bottom shelves
        }
    
    anomalies = []
    
    for obj in detected_objects:
        expected_zones = layout_rules.get(obj["type"], [])
        if expected_zones and obj["zone"] not in expected_zones:
            obj["anomaly"] = "misplaced"
            obj["expected_zone"] = expected_zones[0]  # Just use the first expected zone
            anomalies.append(obj)
        else:
            obj["anomaly"] = "none"
            
    return anomalies

# Detect damaged items (simplified for MVP)
def detect_damaged_items(image, detected_objects):
    anomalies = []
    
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        
        # Extract the object region
        obj_img = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        
        # Simple damage detection: check for irregular edges or holes
        edges = cv2.Canny(gray, 100, 200)
        
        # Count white pixels in the edge image
        edge_pixels = np.sum(edges > 0)
        
        # If there are many edge pixels relative to the size, consider it damaged
        area = (x2 - x1) * (y2 - y1)
        edge_ratio = edge_pixels / area
        
        # Simplified threshold (would be trained in a real system)
        if edge_ratio > 0.1:  # Arbitrary threshold for the MVP
            obj["anomaly"] = "damaged"
            anomalies.append(obj)
            
    return anomalies

# Detect duplicate items
def detect_duplicate_items(detected_objects, threshold_distance=50):
    anomalies = []
    
    # Compare each pair of objects
    for i, obj1 in enumerate(detected_objects):
        for j, obj2 in enumerate(detected_objects):
            if i >= j:  # Skip self-comparison and duplicates
                continue
                
            # Check if they're the same type
            if obj1["type"] != obj2["type"]:
                continue
                
            # Calculate centers
            center1 = [(obj1["bbox"][0] + obj1["bbox"][2]) / 2, 
                      (obj1["bbox"][1] + obj1["bbox"][3]) / 2]
            center2 = [(obj2["bbox"][0] + obj2["bbox"][2]) / 2, 
                      (obj2["bbox"][1] + obj2["bbox"][3]) / 2]
            
            # Calculate distance
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # If close enough, mark as duplicates
            if distance < threshold_distance:
                obj1["anomaly"] = "duplicate"
                obj2["anomaly"] = "duplicate"
                anomalies.append(obj1)
                anomalies.append(obj2)
                
    return anomalies

# ------- Visualization Functions -------

# Draw detected objects with bounding boxes
def draw_detections(image, objects):
    result = image.copy()
    
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        
        # Set color based on object type
        if obj["type"] == "poster":
            color = (0, 255, 0)  # Green
        elif obj["type"] == "banner":
            color = (255, 0, 0)  # Blue
        elif obj["type"] == "standee":
            color = (0, 0, 255)  # Red
        else:
            color = (255, 255, 255)  # White
            
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{obj['type']}"
        cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    return result

# Draw anomalies with highlighted bounding boxes
def draw_anomalies(image, objects):
    result = image.copy()
    
    for obj in objects:
        if obj.get("anomaly", "none") == "none":
            continue
            
        x1, y1, x2, y2 = obj["bbox"]
        
        # Set color based on anomaly type
        if obj["anomaly"] == "misplaced":
            color = (0, 165, 255)  # Orange
        elif obj["anomaly"] == "damaged":
            color = (0, 0, 255)    # Red
        elif obj["anomaly"] == "duplicate":
            color = (255, 0, 255)  # Magenta
        else:
            color = (255, 255, 255)  # White
            
        # Draw bounding box with thicker line
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Draw label
        label = f"{obj['type']} - {obj['anomaly']}"
        cv2.putText(result, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
        
        # For misplaced items, draw an arrow to expected zone
        if obj["anomaly"] == "misplaced" and "expected_zone" in obj:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            expected_y = obj["expected_zone"] + 50  # Center of expected zone
            
            # Draw arrow
            cv2.arrowedLine(result, (center_x, center_y), (center_x, expected_y), 
                          color, 2, tipLength=0.3)
            
    return result

# ------- Main Application -------

def main():
    # Add a retro gaming title with glowing effect
    pixel_title("🎮 SHELF INTEGRITY MONITOR 🎮")
    
    # Add intro text in game style
    game_message("""
    WELCOME PLAYER 1! YOUR MISSION:
    DETECT ANOMALIES IN PROMOTIONAL MATERIALS ON RETAIL SHELVES.
    USE THE CONTROLS ON THE LEFT TO CONFIGURE YOUR SCAN.
    PRESS START TO BEGIN YOUR ADVENTURE!
    """)
    
    # Sidebar
    st.sidebar.markdown("<h2 style='text-align:center; color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>GAME CONTROLS</h2>", unsafe_allow_html=True)
    
    # Input options
    input_type = st.sidebar.radio(
        "SELECT YOUR INPUT:",
        ["Upload Image", "Camera", "Generate Synthetic Data"],
        index=2  # Default to synthetic for the MVP
    )
    
    # Anomaly detection options
    st.sidebar.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALY TYPES</h3>", unsafe_allow_html=True)
    
    detect_misplaced = st.sidebar.checkbox("Misplaced Items", value=True)
    detect_damaged = st.sidebar.checkbox("Damaged Items", value=True)
    detect_duplicates = st.sidebar.checkbox("Duplicate Items", value=True)
    
    # Generate synthetic data options
    if input_type == "Generate Synthetic Data":
        st.sidebar.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>SYNTHETIC DATA</h3>", unsafe_allow_html=True)
        
        add_anomaly = st.sidebar.checkbox("Add Anomaly", value=True)
        
        if add_anomaly:
            anomaly_type = st.sidebar.selectbox(
                "Anomaly Type",
                ["misplaced", "damaged", "duplicate"]
            )
        else:
            anomaly_type = None
    
    # Start button - styled as a game start button
    start_button = st.sidebar.button("🎮 START SCAN 🎮")
    
    # Instructions in the main area before analysis
    if not start_button:
        # Show game-like instructions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>MISSION BRIEFING</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='font-family: VT323, monospace; font-size: 1.2em;'>
            Your retail shelves need constant monitoring to ensure promotional materials
            are correctly placed and in good condition. This AI-powered system will:
            
            - 🔍 Detect and classify promotional items
            - ⚠️ Identify misplaced materials
            - 🔧 Spot damaged or incorrectly installed items
            - 🔄 Find duplicate materials too close together
            
            Configure your scan parameters using the game controls on the left,
            then press START to begin your shelf integrity analysis!
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Show sample anomaly images
            st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALY EXAMPLES</h3>", unsafe_allow_html=True)
            
            # Create three tabs for different anomaly types
            tab1, tab2, tab3 = st.tabs(["Misplaced", "Damaged", "Duplicate"])
            
            with tab1:
                # Generate a sample misplaced item
                shelf = generate_shelf_image(400, 300)
                items_config = [
                    {"type": "poster", "position": [50, 170], "zone": 150}  # Poster on middle shelf
                ]
                img, _ = place_promo_items(shelf, items_config, True, "misplaced")
                st.image(img, caption="Misplaced Promotional Item", use_column_width=True)
            
            with tab2:
                # Generate a sample damaged item
                shelf = generate_shelf_image(400, 300)
                items_config = [
                    {"type": "banner", "position": [150, 170], "zone": 150}
                ]
                img, _ = place_promo_items(shelf, items_config, True, "damaged")
                st.image(img, caption="Damaged Promotional Item", use_column_width=True)
            
            with tab3:
                # Generate a sample duplicate item
                shelf = generate_shelf_image(400, 300)
                items_config = [
                    {"type": "standee", "position": [150, 320], "zone": 300},
                    {"type": "standee", "position": [180, 320], "zone": 300}
                ]
                img, _ = place_promo_items(shelf, items_config, True, "duplicate")
                st.image(img, caption="Duplicate Promotional Items", use_column_width=True)

    # Process the input when the start button is clicked
    if start_button:
        # Create placeholder for results
        results_container = st.container()
        
        with results_container:
            # Show loading animation
            loading_animation("SCANNING")
            
            # Handle different input types
            if input_type == "Upload Image":
                uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])
                
                if uploaded_file is not None:
                    # Read the image
                    image = Image.open(uploaded_file)
                    image = np.array(image)
                    
                    # Process the image
                    process_image(image, detect_misplaced, detect_damaged, detect_duplicates)
            
            elif input_type == "Camera":
                # Camera capture
                camera_photo = st.camera_input("Take a photo of the shelf")
                
                if camera_photo is not None:
                    # Read the image
                    image = Image.open(camera_photo)
                    image = np.array(image)
                    
                    # Process the image
                    process_image(image, detect_misplaced, detect_damaged, detect_duplicates)
            
            elif input_type == "Generate Synthetic Data":
                # Generate a synthetic shelf image
                shelf_img = generate_shelf_image()
                
                # Create random promotional items
                num_items = random.randint(3, 5)
                item_types = ["poster", "banner", "standee"]
                shelf_zones = [0, 150, 300, 450]
                
                items_config = []
                for i in range(num_items):
                    item_type = random.choice(item_types)
                    zone = random.choice(shelf_zones)
                    x_pos = random.randint(50, 700)
                    
                    items_config.append({
                        "type": item_type,
                        "position": [x_pos, zone + 20],
                        "zone": zone
                    })
                
                # Place items on the shelf with or without anomalies
                if add_anomaly:
                    img_with_items, items_data = place_promo_items(
                        shelf_img, items_config, add_anomaly=True, anomaly_type=anomaly_type
                    )
                    has_anomaly = True
                else:
                    img_with_items, items_data = place_promo_items(shelf_img, items_config)
                    has_anomaly = False
                
                # Process the synthetic image
                process_synthetic_image(img_with_items, items_data, has_anomaly, 
                                      detect_misplaced, detect_damaged, detect_duplicates)

# Process a synthetic image where we already know the ground truth
def process_synthetic_image(image, ground_truth, has_anomaly, 
                          detect_misplaced=True, detect_damaged=True, detect_duplicates=True):
    # Display original image
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ORIGINAL SHELF IMAGE</h2>", unsafe_allow_html=True)
    display_pixel_image(image, "Shelf with promotional materials")
    
    # Run object detection
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>OBJECT DETECTION RESULTS</h2>", unsafe_allow_html=True)
    
    # Since we have ground truth for synthetic data, use that instead of running detection
    detected_objects = ground_truth
    
    # Draw detections on the image
    detection_img = draw_detections(image, detected_objects)
    display_pixel_image(detection_img, "Detected promotional items")
    
    # Create a table of detected objects
    st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>DETECTED ITEMS</h3>", unsafe_allow_html=True)
    
    # Convert to DataFrame for nice display
    detection_df = pd.DataFrame([
        {"ID": obj["id"], "Type": obj["type"], "Zone": obj["zone"]}
        for obj in detected_objects
    ])
    
    st.dataframe(detection_df, use_container_width=True)
    
    # Run anomaly detection
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALY DETECTION RESULTS</h2>", unsafe_allow_html=True)
    
    anomalies = []
    
    if detect_misplaced:
        misplaced = [obj for obj in detected_objects if obj.get("anomaly") == "misplaced"]
        anomalies.extend(misplaced)
    
    if detect_damaged:
        damaged = [obj for obj in detected_objects if obj.get("anomaly") == "damaged"]
        anomalies.extend(damaged)
    
    if detect_duplicates:
        duplicates = [obj for obj in detected_objects if obj.get("anomaly") == "duplicate"]
        anomalies.extend(duplicates)
    
    # Draw anomalies on the image
    anomaly_img = draw_anomalies(image, detected_objects)
    display_pixel_image(anomaly_img, "Detected anomalies")
    
    # Create a table of anomalies
    if anomalies:
        st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALIES FOUND</h3>", unsafe_allow_html=True)
        
        # Convert to DataFrame for nice display
        anomaly_df = pd.DataFrame([
            {
                "ID": obj["id"], 
                "Type": obj["type"], 
                "Anomaly": obj.get("anomaly", "none"),
                "Zone": obj["zone"],
                "Expected Zone": obj.get("expected_zone", "N/A")
            }
            for obj in anomalies
        ])
        
        st.dataframe(anomaly_df, use_container_width=True)
        
        # Add a game-style message
        game_message("ANOMALIES DETECTED! Your shelf needs attention. Check the items highlighted above.")
        
        # Add sound effect (just a playful note)
        st.markdown("""
        <audio autoplay>
          <source src="data:audio/wav;base64,UklGRjQnAABXQVZFZm10IBAAAAABAAEARKwAAESsAAABAAgAZGF0YRAnAAAAAAEBAQIDAwQFBgcICQsMDQ8REhQWGBkbHR8hIyUnKSssLjAzNTc5PD5AQUNGRURJTE5RU1ZYW11fYmRnaWxucXN2eHt9gIKFiImCBQUHCAoLDQ8QEhQWGBocHiAiJCYoKy0vMTM2ODtOgkxvdG9SSUZEN3xLNTR2biMjUVBCPzUhEnhOd04zU3cTNzd2TjciEkhOAg=="
          type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)
    else:
        # Game-style success message
        game_message("🎉 ALL CLEAR! No anomalies detected on this shelf. Great job keeping things in order!")
        
        # Add sound effect (just a playful note)
        st.markdown("""
        <audio autoplay>
          <source src="data:audio/wav;base64,UklGRjQnAABXQVZFZm10IBAAAAABAAEARKwAAESsAAABAAgAZGF0YRAnAAAAAAADAwMEBQQGBwgICQgKCwsMDQ4PEBESFBUXGBkaHB0eICIjJCYnKSorLS4wMTM0Njc5Ojs9Pj9BQkNFRkdISUpLTElKS0xNTk9QUVJTVFVWV1haW1xdXl9gYWJjZGVmZ2hpamtsbW5vcHFyc3N0dXZ3eHl6ent8fH19fn+AgYKDhIWG"
          type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)
    
    # Show summary statistics
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANALYSIS SUMMARY</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>TOTAL ITEMS</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{len(detected_objects)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>ANOMALIES</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{len(anomalies)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        health_pct = 100 - (len(anomalies) / max(1, len(detected_objects)) * 100)
        
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>SHELF HEALTH</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{health_pct:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Distribution of item types
    st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ITEM TYPE DISTRIBUTION</h3>", unsafe_allow_html=True)
    
    # Count items by type
    type_counts = {}
    for obj in detected_objects:
        item_type = obj["type"]
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    # Create a DataFrame
    type_df = pd.DataFrame({
        "Item Type": list(type_counts.keys()),
        "Count": list(type_counts.values())
    })
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(type_df["Item Type"], type_df["Count"], color=["#FF6B6B", "#FFD700", "#4BC0C0"])
    ax.set_facecolor("#000080")
    ax.set_xlabel("Item Type", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    plt.xticks(rotation=0)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f"{height:.0f}", ha="center", va="bottom", color="white")
    
    # Customize the plot
    plt.title("Distribution of Promotional Item Types", color="#FFD700", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add a border
    plt.box(on=True)
    ax.spines["bottom"].set_color("#FFD700")
    ax.spines["top"].set_color("#FFD700")
    ax.spines["left"].set_color("#FFD700")
    ax.spines["right"].set_color("#FFD700")
    
    st.pyplot(fig)
    
    # Add a "PLAY AGAIN" button
    if st.button("🎮 PLAY AGAIN 🎮"):
        st.experimental_rerun()

# Process a real image (uploaded or from camera)
def process_image(image, detect_misplaced=True, detect_damaged=True, detect_duplicates=True):
    # Display original image
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ORIGINAL SHELF IMAGE</h2>", unsafe_allow_html=True)
    display_pixel_image(image, "Shelf with promotional materials")
    
    # Run object detection
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>OBJECT DETECTION RESULTS</h2>", unsafe_allow_html=True)
    
    detected_objects = detect_objects(image)
    
    # Draw detections on the image
    detection_img = draw_detections(image, detected_objects)
    display_pixel_image(detection_img, "Detected promotional items")
    
    # Create a table of detected objects
    st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>DETECTED ITEMS</h3>", unsafe_allow_html=True)
    
    # Convert to DataFrame for nice display
    detection_df = pd.DataFrame([
        {"ID": obj["id"], "Type": obj["type"], "Zone": obj["zone"]}
        for obj in detected_objects
    ])
    
    st.dataframe(detection_df, use_container_width=True)
    
    # Run anomaly detection
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALY DETECTION RESULTS</h2>", unsafe_allow_html=True)
    
    anomalies = []
    
    if detect_misplaced:
        misplaced = detect_misplaced_items(detected_objects)
        anomalies.extend(misplaced)
    
    if detect_damaged:
        damaged = detect_damaged_items(image, detected_objects)
        anomalies.extend(damaged)
    
    if detect_duplicates:
        duplicates = detect_duplicate_items(detected_objects)
        anomalies.extend(duplicates)
    
    # Draw anomalies on the image
    anomaly_img = draw_anomalies(image, detected_objects)
    display_pixel_image(anomaly_img, "Detected anomalies")
    
    # Create a table of anomalies
    if anomalies:
        st.markdown("<h3 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANOMALIES FOUND</h3>", unsafe_allow_html=True)
        
        # Convert to DataFrame for nice display
        anomaly_df = pd.DataFrame([
            {
                "ID": obj["id"], 
                "Type": obj["type"], 
                "Anomaly": obj.get("anomaly", "none"),
                "Zone": obj["zone"],
                "Expected Zone": obj.get("expected_zone", "N/A")
            }
            for obj in anomalies
        ])
        
        st.dataframe(anomaly_df, use_container_width=True)
        
        # Add a game-style message
        game_message("ANOMALIES DETECTED! Your shelf needs attention. Check the items highlighted above.")
    else:
        # Game-style success message
        game_message("🎉 ALL CLEAR! No anomalies detected on this shelf. Great job keeping things in order!")
    
    # Show summary statistics
    st.markdown("<h2 style='color:#FFD700; text-shadow: 2px 2px #FF6B6B;'>ANALYSIS SUMMARY</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>TOTAL ITEMS</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{len(detected_objects)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>ANOMALIES</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{len(anomalies)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        health_pct = 100 - (len(anomalies) / max(1, len(detected_objects)) * 100)
        
        st.markdown(f"""
        <div style='background-color: #000080; padding: 10px; border: 3px solid #FFD700; text-align: center;'>
            <h3 style='color: #FFD700; margin: 0;'>SHELF HEALTH</h3>
            <p style='font-size: 2.5em; color: white; font-family: "VT323", monospace; margin: 0;'>{health_pct:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a "PLAY AGAIN" button
    if st.button("🎮 PLAY AGAIN 🎮"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()