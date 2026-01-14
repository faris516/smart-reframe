import cv2
import numpy as np
from ..config import OUTPUT_WIDTH, OUTPUT_HEIGHT, SPLIT_BORDER_THICKNESS, SPLIT_BORDER_COLOR

class SplitScreenCompositor:
    def __init__(self, output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT):
        self.output_width = output_width
        self.output_height = output_height
        self.half_height = output_height // 2

    def create_split_screen(self, frame: np.ndarray, subject1_box: tuple, subject2_box: tuple) -> np.ndarray:
        """
        Creates a vertical split screen with subject1 on top and subject2 on bottom.
        Both subjects are centered in their respective 16:9 (approx) halves.
        """
        top_crop = self._get_centered_crop(frame, subject1_box)
        bottom_crop = self._get_centered_crop(frame, subject2_box)
        
        # Add border
        t = SPLIT_BORDER_THICKNESS
        
        # If we want a divider, we can draw it on one of the crops or insert a black bar
        # Simple approach: Draw a line at the bottom of top_crop
        cv2.line(top_crop, (0, self.half_height-1), (self.output_width, self.half_height-1), SPLIT_BORDER_COLOR, t)
        
        # Stack
        combined = np.vstack((top_crop, bottom_crop))
        
        return combined

    def _get_centered_crop(self, frame: np.ndarray, bbox: tuple) -> np.ndarray:
        """Extracts a crop centered on the bbox matching half the output output dimensions."""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        
        # Target dimensions for half screen
        target_w = self.output_width
        target_h = self.half_height
        
        # Calculate source crop size (zoom to fit height or width?)
        # We want to fill the half-screen.
        # Let's try to maintain aspect ratio of target (target_w / target_h = 1080 / 960 ~= 1.125)
        # Source video is 16:9 usually.
        
        # Simple logic: Crop a box of size (target_w, target_h) * scale from the source
        # We want the subject to be visible.
        # Lets assume we crop relative to source height to keep resolution?
        
        # Scale to match source resolution semantics?
        # Actually safer: Crop a region that has the aspect ratio of the half-screen.
        # Aspect Ratio = output_width / half_height
        ar = target_w / target_h # e.g. 1080 / 960 = 1.125
        
        # Determine crop height based on subject size? or just fixed framing?
        # Let's define crop height as, say, 1.5x the subject height, clamped to video height?
        # Or just use a fixed "Head and Shoulders" frame?
        
        # Dynamic approach:
        # Crop height should be enough to cover the face + padding.
        # Let's say crop_h = 3 * face_h
        crop_h = h * 3
        # Clamp min/max
        crop_h = max(crop_h, target_h) # At least output resolution? No, source might be smaller.
        crop_h = min(crop_h, frame.shape[0])
        
        crop_w = int(crop_h * ar)
        
        # Center the crop on cx, cy
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        # Clamp to frame bounds
        if x1 < 0:
            x2 -= x1 # shift right
            x1 = 0
        if y1 < 0:
            y2 -= y1 # shift down
            y1 = 0
        if x2 > frame.shape[1]:
            x1 -= (x2 - frame.shape[1]) # shift left
            x2 = frame.shape[1]
        if y2 > frame.shape[0]:
            y1 -= (y2 - frame.shape[0]) # shift up
            y2 = frame.shape[0]
            
        # Verify valid crop (if shifting pushed out of bounds again)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
             return np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Resize to target half-screen size
        return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
