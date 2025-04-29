import json
from typing import List, Dict, Any, Optional

def detect_misplaced_items(
    detected_objects: List[Dict[str, Any]], 
    layout_rules: Optional[Dict[str, List[int]]] = None
) -> List[Dict[str, Any]]:
    """
    Detect misplaced promotional items based on expected positions.
    
    Args:
        detected_objects: List of detected objects with position and type information
        layout_rules: Dictionary mapping item types to their expected zone heights
                     If None, default rules will be used
    
    Returns:
        List of objects that are detected as misplaced
    """
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
            # Create a copy of the object to avoid modifying the original
            anomaly_obj = obj.copy()
            anomaly_obj["anomaly"] = "misplaced"
            anomaly_obj["expected_zone"] = expected_zones[0]  # Just use the first expected zone
            anomalies.append(anomaly_obj)
            
    return anomalies

def load_layout_rules(filepath: str) -> Dict[str, List[int]]:
    """
    Load layout rules from a JSON file.
    
    Args:
        filepath: Path to the JSON file containing rules
        
    Returns:
        Dictionary with layout rules
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading layout rules: {e}")
        # Return default rules
        return {
            "poster": [0, 150],
            "banner": [150, 300],
            "standee": [300, 450]
        }

def calculate_misplacement_severity(
    obj: Dict[str, Any], 
    expected_zone: int
) -> float:
    """
    Calculate the severity of misplacement based on distance from expected zone.
    
    Args:
        obj: The misplaced object
        expected_zone: The expected zone height
        
    Returns:
        Severity score (0-1, higher means more severe)
    """
    # Get the center y-coordinate of the object
    y1, y2 = obj["bbox"][1], obj["bbox"][3]
    center_y = (y1 + y2) / 2
    
    # Calculate distance from the expected zone center
    expected_center = expected_zone + 75  # Assuming zones are 150px tall
    distance = abs(center_y - expected_center)
    
    # Normalize to 0-1 (assuming max distance would be 450px, the height of the shelf)
    severity = min(1.0, distance / 450)
    
    return severity