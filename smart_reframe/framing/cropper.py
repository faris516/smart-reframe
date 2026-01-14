import numpy as np
from typing import List, Tuple, Optional
from ..config import TARGET_ASPECT_RATIO, OUTPUT_WIDTH, OUTPUT_HEIGHT

class FocusSelector:
    def select_focus(self, 
                    faces: List[dict], 
                    people: List[dict], 
                    motion: Optional[Tuple[int, int, float]], 
                    active_speaker_idx: Optional[int], 
                    frame_dims: Tuple[int, int]) -> Tuple[float, float]:
        """
        Determines the focus point (x, y) based on priority rules.
        """
        h, w = frame_dims
        center = (w / 2, h / 2)
        
        # 1. Active Speaker
        if active_speaker_idx is not None and 0 <= active_speaker_idx < len(faces):
            f = faces[active_speaker_idx]
            bbox = f['bbox'] # x, y, w, h
            return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

        # 2. Single Face
        if len(faces) == 1:
            f = faces[0]
            bbox = f['bbox']
            return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

        # 3. Top Ranked Face 
        # (Input 'faces' is assumed to be sorted by importance/persistence)
        if len(faces) > 1:
            f = faces[0]
            bbox = f['bbox']
            return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

        # 4. Largest Person
        if people:
            people_sorted = sorted(people, key=lambda p: p['bbox'][2] * p['bbox'][3], reverse=True)
            p = people_sorted[0]
            bbox = p['bbox']
            # Focus on upper body? YOLO bbox is full body usually.
            # Center is safe. Maybe bias upwards slightly for headroom?
            # Let's stick to center for now to avoid complexity errors.
            return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

        # 5. Frame Center (Safer fallback for music videos than motion)
        # Music videos are usually center-composed. If we lose the subject, 
        # snapping to the center is better than chasing random background motion.
        return center

        # 6. Motion Centroid (Deprecated as primary target)
        # Only used if we really want to track pure motion (e.g. sports ball), 
        # but for this specific "Star" use case, it's distraction.
        # if motion:
        #    return (motion[0], motion[1])

class DynamicCropper:
    def __init__(self, source_width: int, source_height: int, aspect_ratio: float = TARGET_ASPECT_RATIO):
        self.source_width = source_width
        self.source_height = source_height
        self.aspect_ratio = aspect_ratio
        self.output_height = OUTPUT_HEIGHT # Default output height from config

    def get_crop_coords(self, focus_point: Tuple[int, int], current_height: Optional[int] = None) -> Tuple[int, int, int, int]:
        """
        Calculates crop coordinates centered on focus_point.
        """
        cx, cy = focus_point
        
        # Use provided height (dynamic zoom) or default
        out_h = int(current_height) if current_height else self.output_height
        out_w = int(out_h * self.aspect_ratio)
        
        # Calculate top-left corner
        x1 = int(cx - out_w // 2)
        y1 = int(cy - out_h // 2)
        
        # Clamp to bounds
        # X Axis
        if x1 < 0:
            x1 = 0
        elif x1 + out_w > self.source_width:
            x1 = self.source_width - out_w
            
        # Y Axis
        if y1 < 0:
             y1 = 0
        elif y1 + out_h > self.source_height:
             y1 = self.source_height - out_h
             
        # Recalculate x2, y2 based on clamped x1, y1
        x2 = x1 + out_w
        y2 = y1 + out_h
        
        return x1, y1, x2, y2
