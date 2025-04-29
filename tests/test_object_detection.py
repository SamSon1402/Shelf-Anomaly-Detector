import unittest
import sys
import os
import numpy as np
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.object_detection import detect_objects, filter_detections, apply_nms
from modules.data_generator import generate_shelf_image, generate_promo_item, place_promo_items

class TestObjectDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a synthetic shelf image with items for testing
        self.shelf_img = generate_shelf_image(width=600, height=400)
        
        # Create item configurations
        self.items_config = [
            {"type": "poster", "position": [50, 20], "zone": 0},
            {"type": "banner", "position": [200, 170], "zone": 150},
            {"type": "standee", "position": [350, 320], "zone": 300}
        ]
        
        # Generate an image with items
        self.test_image, self.ground_truth = place_promo_items(
            self.shelf_img, self.items_config
        )
    
    def test_detect_objects(self):
        """Test that object detection finds all items"""
        # Run object detection
        detected_objects = detect_objects(self.test_image)
        
        # Check that we detect at least the number of items we placed
        self.assertGreaterEqual(len(detected_objects), len(self.items_config))
        
        # Check that detected objects have required attributes
        for obj in detected_objects:
            self.assertIn("id", obj)
            self.assertIn("type", obj)
            self.assertIn("bbox", obj)
            self.assertIn("zone", obj)
            
            # Check bbox structure
            self.assertEqual(len(obj["bbox"]), 4)
            x1, y1, x2, y2 = obj["bbox"]
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
    
    def test_filter_detections(self):
        """Test filtering detections by confidence"""
        # Create test detections with various confidence scores
        test_detections = [
            {"id": "poster_1", "type": "poster", "confidence": 0.9, "bbox": [10, 10, 50, 50]},
            {"id": "banner_1", "type": "banner", "confidence": 0.8, "bbox": [100, 10, 150, 50]},
            {"id": "standee_1", "type": "standee", "confidence": 0.6, "bbox": [200, 10, 250, 50]},
            {"id": "poster_2", "type": "poster", "confidence": 0.5, "bbox": [300, 10, 350, 50]}
        ]
        
        # Filter with default threshold (0.7)
        filtered = filter_detections(test_detections)
        
        # Should keep only the first two items
        self.assertEqual(len(filtered), 2)
        self.assertIn(test_detections[0], filtered)
        self.assertIn(test_detections[1], filtered)
        
        # Filter with custom threshold
        filtered_custom = filter_detections(test_detections, min_confidence=0.6)
        
        # Should keep the first three items
        self.assertEqual(len(filtered_custom), 3)
        self.assertIn(test_detections[0], filtered_custom)
        self.assertIn(test_detections[1], filtered_custom)
        self.assertIn(test_detections[2], filtered_custom)
    
    def test_apply_nms(self):
        """Test non-maximum suppression"""
        # Create test detections with overlapping bounding boxes
        test_detections = [
            {"id": "poster_1", "type": "poster", "confidence": 0.9, "bbox": [10, 10, 60, 60]},
            {"id": "poster_2", "type": "poster", "confidence": 0.8, "bbox": [15, 15, 65, 65]},  # Overlaps with poster_1
            {"id": "banner_1", "type": "banner", "confidence": 0.7, "bbox": [100, 10, 150, 60]},
            {"id": "banner_2", "type": "banner", "confidence": 0.6, "bbox": [105, 15, 155, 65]}  # Overlaps with banner_1
        ]
        
        # Apply NMS
        nms_result = apply_nms(test_detections, iou_threshold=0.5)
        
        # Should keep only the highest confidence detection for each overlapping group
        self.assertEqual(len(nms_result), 2)
        self.assertIn(test_detections[0], nms_result)  # poster_1 (0.9)
        self.assertIn(test_detections[2], nms_result)  # banner_1 (0.7)
        
        # With a higher IOU threshold, it should keep more detections
        nms_result_high_thresh = apply_nms(test_detections, iou_threshold=0.9)
        self.assertGreater(len(nms_result_high_thresh), len(nms_result))

    def test_detection_on_empty_image(self):
        """Test detection on an empty shelf image"""
        # Create an empty shelf
        empty_shelf = generate_shelf_image(width=600, height=400)
        
        # Run detection
        detected_objects = detect_objects(empty_shelf)
        
        # Should find minimal or no objects
        self.assertLessEqual(len(detected_objects), 2)  # May detect shelf lines
    
    def test_detection_consistency(self):
        """Test consistency of detection on identical images"""
        # Run detection twice on the same image
        result1 = detect_objects(self.test_image)
        result2 = detect_objects(self.test_image)
        
        # Should get the same number of objects
        self.assertEqual(len(result1), len(result2))
        
        # Create a slightly modified image
        modified_image = self.test_image.copy()
        modified_image[0:10, 0:10] = [255, 255, 255]  # Modify a small region
        
        # Detection should be stable despite small changes
        result3 = detect_objects(modified_image)
        self.assertEqual(len(result1), len(result3))

if __name__ == '__main__':
    unittest.main()