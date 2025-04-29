import numpy as np
import cv2
import random
from typing import List, Dict, Tuple, Optional, Any

def generate_shelf_image(width: int = 800, height: int = 600) -> np.ndarray:
    """
    Generate a synthetic shelf image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Numpy array containing the shelf image
    """
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

def generate_promo_item(item_type: str, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Generate a synthetic promotional item image.
    
    Args:
        item_type: Type of promotional item ("poster", "banner", or "standee")
        size: Size (width, height) of the item in pixels
        
    Returns:
        Numpy array containing the item image
    """
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

def place_promo_items(
    shelf_img: np.ndarray, 
    items_config: List[Dict[str, Any]], 
    add_anomaly: bool = False, 
    anomaly_type: Optional[str] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Place promotional items on a shelf image.
    
    Args:
        shelf_img: Base shelf image
        items_config: List of item configurations with type, position, and zone
        add_anomaly: Whether to add an anomaly
        anomaly_type: Type of anomaly to add ("misplaced", "damaged", or "duplicate")
        
    Returns:
        Tuple containing:
        - Modified shelf image with items
        - List of item data including positions and anomaly information
    """
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

def generate_dataset(num_normal: int = 3, num_anomalies: int = 3) -> List[Dict[str, Any]]:
    """
    Generate a complete synthetic dataset of shelf images.
    
    Args:
        num_normal: Number of normal (no anomaly) images to generate
        num_anomalies: Number of anomaly images to generate
        
    Returns:
        List of dictionaries containing generated images and metadata
    """
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