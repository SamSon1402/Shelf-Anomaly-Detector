import unittest
import sys
import os
import numpy as np
import cv2

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.anomaly_detection import detect_misplaced_items, detect_damaged_items, detect_duplicate_items
from modules.data_generator import generate_shelf_image, generate_promo_item, place_promo_items

class TestAnomalyDetection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a synthetic shelf image for testing
        self.shelf_img = generate_shelf_image(width=600, height=400)
        
        # Create normal item configurations
        self.normal_items = [
            {"type": "poster", "position": [50, 20], "zone": 0},
            {"type": "banner", "position": [200, 170], "zone": 150},
            {"type": "standee", "position": [350, 320], "zone": 300}
        ]
        
        # Generate normal image
        self.normal_image, self.normal_data = place_promo_items(
            self.shelf_img, self.normal_items
        )
        
        # Create misplaced item configuration
        self.misplaced_items = [
            {"type": "poster", "position": [50, 20], "zone": 0},
            {"type": "banner", "position": [200, 170], "zone": 150},
            {"type": "standee", "position": [350, 20], "zone": 0}  # Standee in wrong zone
        ]
        
        # Generate misplaced image
        self.misplaced_image, self.misplaced_data = place_promo_items(
            self.shelf_img, self.misplaced_items
        )
        
        # Create duplicate item configuration
        self.duplicate_items = [
            {"type": "poster", "position": [50, 20], "zone": 0},
            {"type": "poster", "position": [80, 20], "zone": 0},  # Duplicate poster
            {"type": "banner", "position": [200, 170], "zone": 150}
        ]
        
        # Generate duplicate image
        self.duplicate_image, self.duplicate_data = place_promo_items(
            self.shelf_img, self.duplicate_items
        )
        
        # Create damaged item configuration
        self.damaged_items = self.normal_items.copy()
        
        # Generate damaged image (we'll modify it after generating)
        self.damaged_image, self.damaged_data = place_promo_items(
            self.shelf_img, self.damaged_items
        )
        
        # Add damage to the first item
        x1, y1, x2, y2 = self.damaged_data[0]["bbox"]
        # Add white rectangle and scratches to simulate damage
        cv2.rectangle(self.damaged_image, (x1+10, y1+10), (x1+30, y1+30), (255, 255, 255), -1)
        for _ in range(10):
            x_start = np.random.randint(x1, x2-10)
            y_start = np.random.randint(y1, y2-10)
            x_end = x_start + np.random.randint(5, 20)
            y_end = y_start + np.random.randint(5, 20)
            cv2.line(self.damaged_image, (x_start, y_start), (x_end, y_end), (0, 0, 0), 2)
    
    def test_detect_misplaced_items(self):
        """Test misplacement detection"""
        # Define layout rules
        layout_rules = {
            "poster": [0],
            "banner": [150],
            "standee": [300]
        }
        
        # Test on normal data (should find no anomalies)
        misplaced_normal = detect_misplaced_items(self.normal_data, layout_rules)
        self.assertEqual(len(misplaced_normal), 0)
        
        # Test on misplaced data (should find the standee)
        misplaced_anomalies = detect_misplaced_items(self.misplaced_data, layout_rules)
        self.assertEqual(len(misplaced_anomalies), 1)
        self.assertEqual(misplaced_anomalies[0]["type"], "standee")
        self.assertEqual(misplaced_anomalies[0]["anomaly"], "misplaced")
        self.assertEqual(misplaced_anomalies[0]["expected_zone"], 300)
        
        # Test with default rules
        default_misplaced = detect_misplaced_items(self.misplaced_data)
        self.assertGreaterEqual(len(default_misplaced), 1)
    
    def test_detect_damaged_items(self):
        """Test damage detection"""
        # Test on normal data (should find minimal or no damage)
        damaged_normal = detect_damaged_items(self.normal_image, self.normal_data)
        self.assertLessEqual(len(damaged_normal), 1)  # Might find false positives
        
        # Test on damaged data (should find the damaged item)
        damaged_anomalies = detect_damaged_items(self.damaged_image, self.damaged_data)
        self.assertGreaterEqual(len(damaged_anomalies), 1)
        
        # Should identify the correct anomaly type
        found_damage = False
        for anomaly in damaged_anomalies:
            if anomaly["anomaly"] == "damaged":
                found_damage = True
                break
        self.assertTrue(found_damage)
    
    def test_detect_duplicate_items(self):
        """Test duplicate detection"""
        # Test on normal data (should find no duplicates)
        dup_normal = detect_duplicate_items(self.normal_data)
        self.assertEqual(len(dup_normal), 0)
        
        # Test on duplicate data (should find the duplicate posters)
        dup_anomalies = detect_duplicate_items(self.duplicate_data)
        self.assertEqual(len(dup_anomalies), 2)  # Should find both items in the pair
        
        # Should be a pair of posters
        poster_dups = [a for a in dup_anomalies if a["type"] == "poster"]
        self.assertEqual(len(poster_dups), 2)
        
        # Each should reference the other's ID
        self.assertEqual(poster_dups[0]["duplicate_id"], poster_dups[1]["id"])
        self.assertEqual(poster_dups[1]["duplicate_id"], poster_dups[0]["id"])
        
        # Test with custom threshold
        custom_dup = detect_duplicate_items(self.duplicate_data, threshold_distance=10)
        self.assertEqual(len(custom_dup), 0)  # Too small, shouldn't find duplicates
    
    def test_combined_anomaly_detection(self):
        """Test detecting multiple types of anomalies"""
        # Create a shelf with multiple anomalies
        complex_items = [
            {"type": "poster", "position": [50, 20], "zone": 0},
            {"type": "poster", "position": [80, 20], "zone": 0},  # Duplicate
            {"type": "banner", "position": [200, 320], "zone": 300},  # Misplaced
            {"type": "standee", "position": [350, 320], "zone": 300}
        ]
        
        complex_image, complex_data = place_promo_items(self.shelf_img, complex_items)
        
        # Add damage to the standee
        x1, y1, x2, y2 = complex_data[3]["bbox"]
        cv2.rectangle(complex_image, (x1+10, y1+10), (x1+30, y1+30), (255, 255, 255), -1)
        
        # Run all anomaly detections
        layout_rules = {
            "poster": [0],
            "banner": [150],
            "standee": [300]
        }
        
        misplaced = detect_misplaced_items(complex_data, layout_rules)
        damaged = detect_damaged_items(complex_image, complex_data)
        duplicates = detect_duplicate_items(complex_data)
        
        # Combine all anomalies
        all_anomalies = misplaced + damaged + duplicates
        
        # Should find at least 3 anomalies (1 misplaced, 1 damaged, 2 duplicates)
        self.assertGreaterEqual(len(all_anomalies), 4)
        
        # Should contain different types of anomalies
        anomaly_types = set(a["anomaly"] for a in all_anomalies)
        self.assertGreaterEqual(len(anomaly_types), 2)  # At least 2 different types

if __name__ == '__main__':
    unittest.main()