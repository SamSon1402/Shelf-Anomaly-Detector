import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64
from typing import List, Dict, Any, Optional, Tuple

def draw_detections(image: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw detected objects with bounding boxes on the image.
    
    Args:
        image: The input image
        objects: List of detected objects
        
    Returns:
        Image with bounding boxes drawn
    """
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

def draw_anomalies(image: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw anomalies with highlighted bounding boxes.
    
    Args:
        image: The input image
        objects: List of objects, including those with anomalies
        
    Returns:
        Image with anomalies highlighted
    """
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
            
        # For duplicate items, draw a line to the duplicate
        if obj["anomaly"] == "duplicate" and "duplicate_id" in obj:
            # Find the duplicate object
            duplicate_found = False
            for dup_obj in objects:
                if dup_obj["id"] == obj["duplicate_id"]:
                    dup_x1, dup_y1, dup_x2, dup_y2 = dup_obj["bbox"]
                    
                    # Calculate centers
                    center_x1 = (x1 + x2) // 2
                    center_y1 = (y1 + y2) // 2
                    center_x2 = (dup_x1 + dup_x2) // 2
                    center_y2 = (dup_y1 + dup_y2) // 2
                    
                    # Draw dashed line between duplicates
                    draw_dashed_line(result, (center_x1, center_y1), (center_x2, center_y2), color)
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                # Draw a question mark if duplicate not found
                cv2.putText(result, "?", (x2 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, color, 2)
    
    return result

def draw_dashed_line(
    image: np.ndarray, 
    pt1: Tuple[int, int], 
    pt2: Tuple[int, int], 
    color: Tuple[int, int, int], 
    dash_length: int = 10
) -> None:
    """
    Draw a dashed line on an image.
    
    Args:
        image: The image to draw on
        pt1: First point (x, y)
        pt2: Second point (x, y)
        color: Line color (B, G, R)
        dash_length: Length of each dash
        
    Returns:
        None (modifies image in place)
    """
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    
    if dashes == 0:
        return
        
    x_step = (pt2[0] - pt1[0]) / dashes
    y_step = (pt2[1] - pt1[1]) / dashes
    
    for i in range(dashes):
        if i % 2 == 0:  # Draw only even segments
            start = (int(pt1[0] + i * x_step), int(pt1[1] + i * y_step))
            end = (int(pt1[0] + (i + 1) * x_step), int(pt1[1] + (i + 1) * y_step))
            cv2.line(image, start, end, color, 2)

def display_pixel_image(image: np.ndarray, caption: str = "") -> str:
    """
    Convert an image to a base64 string for display with pixel styling.
    
    Args:
        image: The image to convert
        caption: Optional caption for the image
        
    Returns:
        HTML string for displaying the image with pixel styling
    """
    # Encode the image
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('ascii')
    
    # Create HTML with pixel styling
    html = f"""
    <figure>
        <img src="data:image/png;base64,{img_str}" class="pixel-border" style="width:100%">
        <figcaption style="text-align:center; font-family:'VT323', monospace; margin-top:5px">{caption}</figcaption>
    </figure>
    """
    
    return html

def create_item_type_chart(objects: List[Dict[str, Any]]) -> plt.Figure:
    """
    Create a bar chart showing the distribution of item types.
    
    Args:
        objects: List of detected objects
        
    Returns:
        Matplotlib figure object
    """
    # Count items by type
    type_counts = {}
    for obj in objects:
        item_type = obj["type"]
        type_counts[item_type] = type_counts.get(item_type, 0) + 1
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create bars with retro colors
    bars = ax.bar(
        list(type_counts.keys()), 
        list(type_counts.values()), 
        color=["#FF6B6B", "#FFD700", "#4BC0C0"]
    )
    
    # Style the chart with retro gaming aesthetic
    ax.set_facecolor("#000080")  # Navy blue background
    ax.set_xlabel("Item Type", color="white")
    ax.set_ylabel("Count", color="white")
    ax.tick_params(colors="white")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f"{height:.0f}", 
            ha="center", 
            va="bottom", 
            color="white"
        )
    
    # Customize the plot
    plt.title("Distribution of Promotional Item Types", color="#FFD700", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add a border
    plt.box(on=True)
    for spine in ax.spines.values():
        spine.set_color("#FFD700")  # Golden yellow border
    
    return fig

def create_anomaly_pie_chart(objects: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    """
    Create a pie chart showing the distribution of anomaly types.
    
    Args:
        objects: List of objects with anomaly information
        
    Returns:
        Matplotlib figure object or None if no anomalies
    """
    # Count anomalies by type
    anomaly_counts = {}
    for obj in objects:
        if obj.get("anomaly", "none") != "none":
            anomaly_type = obj["anomaly"]
            anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1
    
    # Add normal items
    normal_count = sum(1 for obj in objects if obj.get("anomaly", "none") == "none")
    if normal_count > 0:
        anomaly_counts["normal"] = normal_count
    
    # If no data, return None
    if not anomaly_counts:
        return None
        
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Define retro colors
    colors = {
        "misplaced": "#FF6B6B",  # Coral
        "damaged": "#FF4500",    # Red-orange
        "duplicate": "#DA70D6",  # Orchid
        "normal": "#4BC0C0"      # Teal
    }
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        anomaly_counts.values(),
        labels=anomaly_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors.get(key, "#FFFFFF") for key in anomaly_counts.keys()]
    )
    
    # Style the texts
    for text in texts:
        text.set_color('#FFD700')  # Golden yellow text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Style the chart with retro gaming aesthetic
    ax.set_facecolor("#000080")  # Navy blue background
    fig.patch.set_facecolor("#000080")
    
    plt.title("Anomaly Distribution", color="#FFD700", fontsize=14)
    
    return fig

def create_zone_heatmap(objects: List[Dict[str, Any]], image_height: int = 600) -> plt.Figure:
    """
    Create a heatmap showing the distribution of items across shelf zones.
    
    Args:
        objects: List of detected objects
        image_height: Height of the original image
        
    Returns:
        Matplotlib figure object
    """
    # Define zones (assuming standard shelf heights)
    zones = [0, 150, 300, 450, image_height]
    zone_names = ["Top Shelf", "Middle Shelf 1", "Middle Shelf 2", "Bottom Shelf"]
    
    # Count items by type and zone
    item_types = ["poster", "banner", "standee"]
    zone_data = np.zeros((len(zone_names), len(item_types)))
    
    for obj in objects:
        # Determine zone index
        y_center = (obj["bbox"][1] + obj["bbox"][3]) / 2
        zone_idx = 0
        for z in range(len(zones)-1):
            if y_center >= zones[z] and y_center < zones[z+1]:
                zone_idx = z
                break
        
        # Determine type index
        type_idx = item_types.index(obj["type"]) if obj["type"] in item_types else 0
        
        # Increment count
        zone_data[zone_idx, type_idx] += 1
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(zone_data, cmap="YlOrRd")
    
    # Style the chart
    ax.set_xticks(np.arange(len(item_types)))
    ax.set_yticks(np.arange(len(zone_names)))
    ax.set_xticklabels(item_types)
    ax.set_yticklabels(zone_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(zone_names)):
        for j in range(len(item_types)):
            text = ax.text(j, i, int(zone_data[i, j]),
                          ha="center", va="center", color="black")
    
    # Style with retro gaming theme
    ax.set_facecolor("#000080")
    fig.patch.set_facecolor("#000080")
    ax.tick_params(colors="white")
    plt.title("Item Distribution by Zone", color="#FFD700", fontsize=14)
    plt.xlabel("Item Type", color="white")
    plt.ylabel("Shelf Zone", color="white")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Item Count", rotation=-90, va="bottom", color="white")
    cbar.ax.tick_params(colors="white")
    
    fig.tight_layout()
    
    return fig