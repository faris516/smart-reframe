import cv2
import numpy as np
from typing import Optional, Tuple

class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """
        Detects significant motion in the frame.
        Returns: (centroid_x, centroid_y, area) or None
        """
        # Downscale for performance
        h, w = frame.shape[:2]
        target_w = 512
        scale = w / target_w
        if w > target_w:
            small_w = target_w
            small_h = int(h / scale)
            processing_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            processing_frame = frame
            
        gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return None
            
        if self.prev_gray.shape != gray.shape:
             self.prev_gray = gray
             return None
            
        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill holes
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate moments to find centroid
        M = cv2.moments(thresh)
        result = None
        
        # Min area 500 on 512px width is reasonable (~1% of width squared?) 
        # 512*512 = 262k pixels. 500 is very small (0.2%).
        # Let's keep 500 but note it's on downscaled image.
        if M["m00"] > 500: 
            cX = int((M["m10"] / M["m00"]) * scale)
            cY = int((M["m01"] / M["m00"]) * scale)
            real_area = M["m00"] * (scale * scale)
            result = (cX, cY, real_area)
            
        self.prev_gray = gray
        return result
