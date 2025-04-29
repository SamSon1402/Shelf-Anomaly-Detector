import cv2
import numpy as np
from typing import List, Dict, Any

def detect_damaged_items(
    image: np.ndarray, 
    detected_objects: List[Dict[str, Any]],
    edge_ratio_threshold: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Detect damaged promotional items based on visual cues.
    
    This is a simplified algorithm for the MVP. In a real implementation,
    this would likely use a trained model to detect damage.
    
    Args:
        image: The input image
        detected_objects: List of detected objects
        edge_ratio_threshold: Threshold for the edge pixel ratio
        
    Returns:
        List of objects that are detected as damaged
    """
    anomalies = []
    
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        
        # Extract the object region
        obj_img = image[y1:y2, x1:x2]
        
        # Skip if the region is empty (can happen at image boundaries)
        if obj_img.size == 0:
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        
        # Simple damage detection: check for irregular edges or holes
        edges = cv2.Canny(gray, 100, 200)
        
        # Count white pixels in the edge image
        edge_pixels = np.sum(edges > 0)
        
        # Calculate the area
        area = (x2 - x1) * (y2 - y1)
        
        # If there are many edge pixels relative to the size, consider it damaged
        edge_ratio = edge_pixels / area if area > 0 else 0
        
        if edge_ratio > edge_ratio_threshold:
            # Create a copy of the object to avoid modifying the original
            anomaly_obj = obj.copy()
            anomaly_obj["anomaly"] = "damaged"
            anomaly_obj["damage_confidence"] = min(1.0, edge_ratio / (2 * edge_ratio_threshold))
            anomalies.append(anomaly_obj)
            
    return anomalies

def detect_tears(obj_img: np.ndarray) -> bool:
    """
    Detect tears in promotional materials.
    
    Args:
        obj_img: Image of the object
        
    Returns:
        True if tears are detected, False otherwise
    """
    # Convert to grayscale
    gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological operations to find potential tears
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(gray, kernel, iterations=1)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    
    # Calculate the difference between erosion and dilation
    diff = cv2.absdiff(dilation, erosion)
    
    # Threshold the difference
    _, thresholded = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours of potential tears
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Minimum area for a tear
            # Get bounding rect and calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
            
            # Tears typically have high aspect ratio
            if aspect_ratio > 3:
                return True
    
    return False

def detect_color_defects(obj_img: np.ndarray) -> bool:
    """
    Detect color defects in promotional materials.
    
    Args:
        obj_img: Image of the object
        
    Returns:
        True if color defects are detected, False otherwise
    """
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
    
    # Split the channels
    h, s, v = cv2.split(hsv)
    
    # Calculate standard deviation of saturation
    std_s = np.std(s)
    
    # Low saturation variation can indicate color issues
    return std_s < 10