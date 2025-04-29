import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

def detect_duplicate_items(
    detected_objects: List[Dict[str, Any]], 
    threshold_distance: float = 50
) -> List[Dict[str, Any]]:
    """
    Detect duplicate promotional items that are placed too close together.
    
    Args:
        detected_objects: List of detected objects
        threshold_distance: Maximum distance between centers to consider as duplicates
        
    Returns:
        List of objects that are detected as duplicates
    """
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
            center1 = calculate_center(obj1["bbox"])
            center2 = calculate_center(obj2["bbox"])
            
            # Calculate distance between centers
            distance = calculate_distance(center1, center2)
            
            # If close enough, mark as duplicates
            if distance < threshold_distance:
                # Create copies to avoid modifying the original objects
                dup1 = obj1.copy()
                dup2 = obj2.copy()
                
                dup1["anomaly"] = "duplicate"
                dup2["anomaly"] = "duplicate"
                
                # Store the duplicate pair info
                dup1["duplicate_id"] = obj2["id"]
                dup2["duplicate_id"] = obj1["id"]
                
                # Store the distance
                dup1["duplicate_distance"] = distance
                dup2["duplicate_distance"] = distance
                
                anomalies.append(dup1)
                anomalies.append(dup2)
                
    return anomalies

def calculate_center(bbox: List[int]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Tuple (center_x, center_y)
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value (0-1)
    """
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Check if there is an intersection
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Calculate IoU
    iou = intersection / (area1 + area2 - intersection)
    
    return iou

def detect_duplicates_by_appearance(
    image: np.ndarray,
    detected_objects: List[Dict[str, Any]],
    similarity_threshold: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Detect duplicate items based on visual appearance.
    
    This is more accurate than just distance-based detection
    but more computationally expensive.
    
    Args:
        image: The input image
        detected_objects: List of detected objects
        similarity_threshold: Threshold for considering objects as duplicates
        
    Returns:
        List of objects detected as duplicates
    """
    anomalies = []
    
    # Extract features from each object
    object_features = []
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        obj_img = image[y1:y2, x1:x2]
        
        # Skip if the region is empty (can happen at image boundaries)
        if obj_img.size == 0:
            object_features.append(None)
            continue
            
        # Convert to grayscale and resize for consistent comparison
        gray = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        
        # Compute features (using histogram as a simple feature)
        hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        object_features.append(hist)
    
    # Compare each pair of objects
    for i, obj1 in enumerate(detected_objects):
        if object_features[i] is None:
            continue
            
        for j, obj2 in enumerate(detected_objects):
            if i >= j or object_features[j] is None:
                continue
                
            # Skip different types
            if obj1["type"] != obj2["type"]:
                continue
                
            # Calculate similarity
            similarity = cv2.compareHist(object_features[i], object_features[j], cv2.HISTCMP_CORREL)
            
            # If similar enough, mark as duplicates
            if similarity > similarity_threshold:
                # Calculate distance to further validate
                center1 = calculate_center(obj1["bbox"])
                center2 = calculate_center(obj2["bbox"])
                distance = calculate_distance(center1, center2)
                
                # Consider as duplicates if they're similar and close
                if distance < 150:  # Allow higher distance for appearance-based detection
                    # Create copies to avoid modifying the original objects
                    dup1 = obj1.copy()
                    dup2 = obj2.copy()
                    
                    dup1["anomaly"] = "duplicate"
                    dup2["anomaly"] = "duplicate"
                    
                    dup1["duplicate_id"] = obj2["id"]
                    dup2["duplicate_id"] = obj1["id"]
                    
                    dup1["similarity"] = similarity
                    dup2["similarity"] = similarity
                    
                    dup1["duplicate_distance"] = distance
                    dup2["duplicate_distance"] = distance
                    
                    anomalies.append(dup1)
                    anomalies.append(dup2)
    
    return anomalies