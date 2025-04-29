import cv2
import numpy as np
from typing import List, Dict, Any

def detect_objects(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect promotional items in a shelf image.
    
    This is a simplified detection algorithm for the MVP that uses
    basic computer vision techniques. In a production environment,
    this would be replaced with a trained deep learning model.
    
    Args:
        image: The input shelf image
        
    Returns:
        List of dictionaries containing detected object information
    """
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
            "zone": zone,
            "confidence": 0.95  # Placeholder for a real confidence score
        })
    
    return detected_objects

def apply_nms(objects: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        objects: List of detected objects
        iou_threshold: Intersection over Union threshold
        
    Returns:
        Filtered list of objects after NMS
    """
    # Sort objects by confidence (descending)
    sorted_objects = sorted(objects, key=lambda x: x.get('confidence', 0.0), reverse=True)
    selected_objects = []
    
    while sorted_objects:
        # Select the object with highest confidence
        current = sorted_objects.pop(0)
        selected_objects.append(current)
        
        # Remove objects that overlap significantly with the current object
        remaining_objects = []
        
        for obj in sorted_objects:
            # Calculate IoU
            bbox1 = current['bbox']
            bbox2 = obj['bbox']
            
            # Calculate intersection
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            # Check if there is an intersection
            if x1 < x2 and y1 < y2:
                intersection = (x2 - x1) * (y2 - y1)
                
                # Calculate areas
                area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                
                # Calculate IoU
                iou = intersection / (area1 + area2 - intersection)
                
                # Keep if IoU is below threshold
                if iou < iou_threshold:
                    remaining_objects.append(obj)
            else:
                remaining_objects.append(obj)
        
        sorted_objects = remaining_objects
    
    return selected_objects

def filter_detections(objects: List[Dict[str, Any]], min_confidence: float = 0.7) -> List[Dict[str, Any]]:
    """
    Filter detected objects based on confidence score.
    
    Args:
        objects: List of detected objects
        min_confidence: Minimum confidence threshold
        
    Returns:
        Filtered list of objects
    """
    return [obj for obj in objects if obj.get('confidence', 0.0) >= min_confidence]

def classify_detected_objects(image: np.ndarray, bboxes: List[List[int]]) -> List[Dict[str, Any]]:
    """
    Classify detected objects based on visual features.
    
    This is a simplified classification for the MVP.
    In a real implementation, this would use a trained classifier.
    
    Args:
        image: The input image
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        
    Returns:
        List of classified objects with type information
    """
    classified_objects = []
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Extract the object region
        obj_img = image[y1:y2, x1:x2]
        
        # Calculate the aspect ratio
        aspect_ratio = (x2 - x1) / (y2 - y1)
        
        # Use aspect ratio and size for a simple classification
        if aspect_ratio > 1.5:
            obj_type = "banner"
        elif aspect_ratio < 0.7:
            obj_type = "standee"
        else:
            obj_type = "poster"
        
        # Determine the shelf zone
        shelf_zones = [0, 150, 300, 450, 600]
        zone = 0
        for z in range(len(shelf_zones)-1):
            if y1 >= shelf_zones[z] and y1 < shelf_zones[z+1]:
                zone = shelf_zones[z]
                break
        
        classified_objects.append({
            "id": f"{obj_type}_{i}",
            "type": obj_type,
            "bbox": bbox,
            "zone": zone,
            "confidence": 0.95  # Placeholder
        })
    
    return classified_objects