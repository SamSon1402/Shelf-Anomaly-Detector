import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os
from PIL import Image

def resize_image(image: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """
    Resize an image if it's larger than max_width.
    
    Args:
        image: Input image
        max_width: Maximum width
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if width > max_width:
        # Calculate new dimensions
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        
        # Resize the image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance an image for better object detection.
    
    Args:
        image: Input image
        
    Returns:
        Enhanced image
    """
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Apply contrast enhancement
    alpha = 1.2  # Contrast factor
    beta = 0.1    # Brightness factor
    enhanced = cv2.convertScaleAbs(img_float, alpha=alpha, beta=beta)
    
    # Apply sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to uint8
    result = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
    
    return result

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image colors.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    # Create a copy to avoid modifying the original
    result = image.copy()
    
    # Split the channels
    b, g, r = cv2.split(result)
    
    # Normalize each channel
    b_norm = cv2.equalizeHist(b)
    g_norm = cv2.equalizeHist(g)
    r_norm = cv2.equalizeHist(r)
    
    # Merge the channels back
    result = cv2.merge((b_norm, g_norm, r_norm))
    
    return result

def remove_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the background from an image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple containing:
        - Foreground image with transparent background
        - Binary mask (255 for foreground, 0 for background)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create mask with a bit of morphology to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Create BGRA image (with alpha channel)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel based on mask
    bgra[:, :, 3] = mask
    
    return bgra, mask

def extract_shelf_lines(image: np.ndarray) -> np.ndarray:
    """
    Extract horizontal shelf lines from an image.
    
    Args:
        image: Input image
        
    Returns:
        Image with only the shelf lines
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=100, 
        minLineLength=image.shape[1]//2, maxLineGap=20
    )
    
    # Create a blank image
    line_image = np.zeros_like(image)
    
    if lines is not None:
        # Filter for horizontal lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return line_image

def adjust_brightness_contrast(
    image: np.ndarray, 
    brightness: float = 0, 
    contrast: float = 1
) -> np.ndarray:
    """
    Adjust brightness and contrast of an image.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-1 to 1)
        contrast: Contrast adjustment (0 to 3)
        
    Returns:
        Adjusted image
    """
    # Convert to float for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Apply brightness and contrast adjustments
    adjusted = contrast * img_float + brightness
    
    # Clip values to valid range
    adjusted = np.clip(adjusted, 0, 1)
    
    # Convert back to uint8
    result = (adjusted * 255).astype(np.uint8)
    
    return result

def perspective_transform(
    image: np.ndarray, 
    points: np.ndarray
) -> np.ndarray:
    """
    Apply a perspective transform to an image.
    
    Args:
        image: Input image
        points: Four corner points of the shelf
        
    Returns:
        Transformed image
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define the destination points (rectangular)
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Convert source points to float32
    src_points = points.astype(np.float32)
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transformation
    result = cv2.warpPerspective(image, matrix, (width, height))
    
    return result

def convert_from_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image (numpy array)
    """
    # Convert PIL Image to RGB (if it's not already)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    np_image = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    
    return cv_image

def convert_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image to PIL format.
    
    Args:
        cv_image: OpenCV image (numpy array)
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def check_image_quality(image: np.ndarray) -> Tuple[bool, str]:
    """
    Check if an image has sufficient quality for analysis.
    
    Args:
        image: Input image
        
    Returns:
        Tuple containing:
        - Boolean indicating if the image is usable
        - String message with quality assessment
    """
    # Check if image is valid
    if image is None or image.size == 0:
        return False, "Invalid image"
    
    # Check dimensions
    height, width = image.shape[:2]
    if width < 100 or height < 100:
        return False, "Image is too small (minimum 100x100 pixels)"
    
    # Check if image is too dark
    brightness = np.mean(image)
    if brightness < 30:
        return False, "Image is too dark for reliable detection"
    
    # Check if image is blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, "Image is too blurry for reliable detection"
    
    # All checks passed
    return True, "Image quality is sufficient for analysis"