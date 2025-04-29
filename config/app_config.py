"""
Configuration settings for the Shelf Integrity Monitor application.
"""

# Application settings
APP_TITLE = "Shelf Integrity Monitor"
APP_ICON = "🕹️"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Shelf configuration
DEFAULT_SHELF_WIDTH = 800
DEFAULT_SHELF_HEIGHT = 600
SHELF_ZONES = [0, 150, 300, 450, 600]  # Y-coordinates of shelf zones
ZONE_NAMES = ["Top Shelf", "Middle Shelf 1", "Middle Shelf 2", "Bottom Shelf"]

# Promotional item types
ITEM_TYPES = ["poster", "banner", "standee"]

# Default item sizes (width, height)
ITEM_SIZES = {
    "poster": (100, 100),
    "banner": (150, 75),
    "standee": (80, 120)
}

# Anomaly types
ANOMALY_TYPES = ["misplaced", "damaged", "duplicate"]

# Colors for visualization
COLORS = {
    # Item type colors
    "poster": (0, 255, 0),    # Green
    "banner": (255, 0, 0),    # Blue
    "standee": (0, 0, 255),   # Red
    
    # Anomaly colors
    "misplaced": (0, 165, 255),  # Orange
    "damaged": (0, 0, 255),      # Red
    "duplicate": (255, 0, 255),  # Magenta
    
    # UI colors
    "background": "#000080",     # Navy blue
    "primary": "#FFD700",        # Golden yellow
    "secondary": "#FF6B6B",      # Coral
    "text": "#FFFFFF"            # White
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    "object_detection": {
        "min_contour_area": 500,     # Minimum area for a valid contour
        "confidence_threshold": 0.7,  # Minimum confidence for a valid detection
        "nms_threshold": 0.5          # Non-maximum suppression threshold
    },
    "anomaly_detection": {
        "misplaced": {
            # No specific thresholds needed for misplacement detection
        },
        "damaged": {
            "edge_ratio_threshold": 0.1,  # Ratio of edge pixels to area
            "irregularity_threshold": 0.2  # Threshold for shape irregularity
        },
        "duplicate": {
            "distance_threshold": 50,      # Maximum distance between centers (pixels)
            "appearance_threshold": 0.8    # Minimum similarity threshold (0-1)
        }
    }
}

# Default layout rules for misplacement detection
DEFAULT_LAYOUT_RULES = {
    "poster": [0, 150],   # Posters expected on top shelves
    "banner": [150, 300], # Banners on middle shelves
    "standee": [300, 450] # Standees on bottom shelves
}

# Synthetic data generation
SYNTHETIC_DATA_CONFIG = {
    "num_normal_samples": 3,
    "num_anomaly_samples": 3,
    "min_items_per_shelf": 3,
    "max_items_per_shelf": 5
}

# Performance optimization
PERFORMANCE = {
    "image_processing": {
        "max_width": 1200,       # Maximum width for processing
        "resize_factor": 0.75    # Resize factor for large images
    }
}