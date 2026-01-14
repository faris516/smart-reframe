import cv2
import numpy as np
from ..config import SHORTS_TEXT_EDGE_THRESHOLD

class TextDetector:
    def __init__(self, processing_width=320):
        self.width = processing_width

    def has_text(self, frame: np.ndarray) -> bool:
        """
        Detects if the frame has significant text density in top/bottom regions.
        Speed-optimized: Resizes -> Grayscale -> Canny -> Density Check.
        """
        h, w = frame.shape[:2]
        ratio = self.width / w
        new_h = int(h * ratio)
        
        # Resize for performance
        small = cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Define Regions of Interest (ROI)
        # 1. Bands (Top 20%, Bottom 20%) - Common for credits/lyrics
        roi_h = int(new_h * 0.2)
        top_roi = gray[0:roi_h, :]
        bottom_roi = gray[new_h-roi_h:new_h, :]
        
        # 2. Center (Middle 60%) - For standalone titles/screens
        # We start below the top ROI and end before the bottom ROI
        center_roi = gray[roi_h:new_h-roi_h, :]
        
        # Edge Detection
        # Thresholds: 100, 200 standard for Canny
        edges_top = cv2.Canny(top_roi, 100, 200)
        edges_bottom = cv2.Canny(bottom_roi, 100, 200)
        edges_center = cv2.Canny(center_roi, 100, 200)
        
        # Calculate density
        # Avoid division by zero
        density_top = np.count_nonzero(edges_top) / max(edges_top.size, 1)
        density_bottom = np.count_nonzero(edges_bottom) / max(edges_bottom.size, 1)
        density_center = np.count_nonzero(edges_center) / max(edges_center.size, 1)
        
        # Logic:
        # A. Bands: Use standard threshold (0.15)
        # B. Center: Use stricter threshold (0.30) to avoid confused faces/texture as text
        
        from ..config import SHORTS_TEXT_EDGE_THRESHOLD, SHORTS_TEXT_CENTER_THRESHOLD
        
        if density_top > SHORTS_TEXT_EDGE_THRESHOLD:
            return True
        if density_bottom > SHORTS_TEXT_EDGE_THRESHOLD:
            return True
        if density_center > SHORTS_TEXT_CENTER_THRESHOLD:
            return True
            
    
    def detect_text_regions(self, frame: np.ndarray) -> list:
        """
        Returns bounding boxes (x,y,w,h) of high-density edge regions likely to be text.
        """
        h_orig, w_orig = frame.shape[:2]
        ratio = self.width / w_orig
        new_h = int(h_orig * ratio)
        
        small = cv2.resize(frame, (self.width, new_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Aggressive Edge Detection + Dilation to merge letters into blocks
        edges = cv2.Canny(gray, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)) # Wide kernel for joining words
        dilated = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filters
            aspect = w / h
            density = cv2.contourArea(cnt) / (w * h)
            
            # Text is usually wider than tall, and somewhat dense
            if w > 20 and h > 8 and aspect > 1.5 and density > 0.3:
                 # Scale back to original
                 scale = 1.0 / ratio
                 bx = int(x * scale)
                 by = int(y * scale)
                 bw = int(w * scale)
                 bh = int(h * scale)
                 text_boxes.append((bx, by, bw, bh))
                 
        return text_boxes
